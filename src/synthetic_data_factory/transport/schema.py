"""Canonical Arrow schema for :class:`InstructionSample` and conversion helpers.

Type-drift policy
-----------------
``pyarrow.Table.cast`` is permissive. It turns ``int64`` into ``"7"`` and
reinterprets naive timestamps as UTC. :func:`conform` therefore checks every
column's *source* type against a whitelist before casting:

* text columns: ``string``, ``large_string`` or ``string_view``;
* ``metadata``: a map whose keys and values are text;
* ``created_at_utc``: a timestamp whose timezone is UTC, in any unit. Casting
  to ``us`` is *safe*, so sub-microsecond data is rejected, never truncated.

``keys_sorted=True`` is a claim written into the file. :func:`conform`
verifies that map keys are strictly increasing within each row (which also
means they are unique) instead of trusting the flag.

Copy semantics
--------------
Python objects to Arrow always copies. Arrow to Polars is zero-copy for
fixed-width columns (``created_at_utc`` shares its buffer). Text columns are
transcoded, because Polars stores strings as views and exports
``large_string``.
"""

# Deferred annotations: pyarrow-stubs makes Array/ChunkedArray generic for the
# type checker, but the runtime classes are not subscriptable.
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Final, cast, overload

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

from synthetic_data_factory.schemas import InstructionSample

SCHEMA_ID_KEY: Final = b"sdf.schema"
SCHEMA_ID: Final = b"instruction_sample/v1"

# keys_sorted passed positionally: the runtime keyword is ``keys_sorted`` but
# pyarrow-stubs spells it ``key_sorted``; position 3 is correct for both.
METADATA_TYPE: Final = pa.map_(pa.string(), pa.string(), True)
TIMESTAMP_TYPE: Final = pa.timestamp("us", tz="UTC")
TEXT_COLUMNS: Final = ("sample_id", "system_prompt", "instruction", "response", "domain")

INSTRUCTION_SAMPLE_SCHEMA: Final = pa.schema(
    [
        *(pa.field(name, pa.string(), nullable=False) for name in TEXT_COLUMNS),
        # Non-null (stricter than a bare map): the model always has a dict, so
        # an absent map would be a third state ("unknown") the contract lacks.
        pa.field("metadata", METADATA_TYPE, nullable=False),
        pa.field("created_at_utc", TIMESTAMP_TYPE, nullable=False),
    ],
    metadata={SCHEMA_ID_KEY: SCHEMA_ID},
)

_UTC_ZONES: Final = frozenset({"UTC", "Etc/UTC", "+00:00"})
_ENTRY_LIST_TYPE: Final = pa.list_(
    pa.field(
        "entries",
        pa.struct([pa.field("key", pa.string(), nullable=False), pa.field("value", pa.string())]),
        nullable=False,
    )
)

type ArrowData = pa.Table | pa.RecordBatch


class SchemaMismatchError(ValueError):
    """Arrow data does not conform to :data:`INSTRUCTION_SAMPLE_SCHEMA`."""


def schema_matches(actual: pa.Schema) -> bool:
    """Semantic equality with :data:`INSTRUCTION_SAMPLE_SCHEMA`.

    Column names and order, nullability, schema metadata and full semantic
    types must all match. ``DataType.equals`` enforces ``keys_sorted``, nested
    nullability, item types and timezones, and ignores only nested child
    *names*. That is deliberate: Parquet renames a map's ``entries`` struct
    after its column, so ``Schema.equals`` would report a round-trip as drift.
    """
    expected = INSTRUCTION_SAMPLE_SCHEMA
    return (
        actual.names == expected.names
        and actual.metadata == expected.metadata
        and all(
            a.nullable == e.nullable and a.type.equals(e.type)
            for a, e in zip(actual, expected, strict=True)
        )
    )


def _is_text(data_type: pa.DataType) -> bool:
    return bool(
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_string_view(data_type)
    )


def _source_type_ok(name: str, data_type: pa.DataType) -> bool:
    if name == "metadata":
        return bool(
            pa.types.is_map(data_type)
            and _is_text(data_type.key_type)
            and _is_text(data_type.item_type)
        )
    if name == "created_at_utc":
        return bool(pa.types.is_timestamp(data_type) and data_type.tz in _UTC_ZONES)
    return _is_text(data_type)


def _require_strictly_sorted_keys(column: pa.ChunkedArray[Any]) -> None:
    # Cast to list<struct> first: MapArray.keys ignores slice offsets, but
    # list_flatten and list_parent_indices respect them.
    entries: pa.ChunkedArray[Any] = column.cast(_ENTRY_LIST_TYPE)
    keys = pc.struct_field(pc.list_flatten(entries), "key")
    if len(keys) < 2:
        return
    parents = np.asarray(pc.list_parent_indices(entries))
    ascending = np.asarray(pc.less(keys[:-1], keys[1:]))
    if np.any((parents[:-1] == parents[1:]) & ~ascending):
        raise SchemaMismatchError("metadata keys must be strictly increasing within each row")


@overload
def conform(data: pa.Table, *, verify_sorted_keys: bool = True) -> pa.Table: ...
@overload
def conform(data: pa.RecordBatch, *, verify_sorted_keys: bool = True) -> pa.RecordBatch: ...
def conform(data: ArrowData, *, verify_sorted_keys: bool = True) -> ArrowData:
    """Return ``data`` with exactly the canonical schema, or raise :class:`SchemaMismatchError`.

    Columns may come in any order and use any whitelisted source type. Extra,
    missing or duplicated columns, disallowed types, nulls in non-null fields,
    sub-microsecond timestamps and (unless ``verify_sorted_keys=False``)
    unsorted or duplicate map keys are rejected. Skipping the key check is only
    for callers that decode maps into dicts straight away and police
    duplicates themselves; such data must never be written back under the
    ``keys_sorted`` claim.
    """
    names = data.schema.names
    expected = INSTRUCTION_SAMPLE_SCHEMA.names
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise SchemaMismatchError(f"expected columns {sorted(expected)}, got {sorted(names)}")
    for field in data.schema:
        if not _source_type_ok(field.name, field.type):
            raise SchemaMismatchError(f"column {field.name!r} has disallowed type {field.type}")
    try:
        conformed = data.select(expected).cast(INSTRUCTION_SAMPLE_SCHEMA)
    # ArrowInvalid subclasses ValueError; a null cast into a non-null field
    # raises a *plain* ValueError, so catching ArrowInvalid alone lets it escape.
    except (ValueError, pa.ArrowTypeError) as exc:
        raise SchemaMismatchError(str(exc)) from exc
    if verify_sorted_keys:
        metadata = conformed.column("metadata")
        if isinstance(metadata, pa.Array):  # RecordBatch columns are plain arrays
            metadata = pa.chunked_array([metadata])
        _require_strictly_sorted_keys(metadata)
    return conformed


def samples_to_record_batch(samples: Sequence[InstructionSample]) -> pa.RecordBatch:
    """Columnar encoding of validated samples. Metadata keys are emitted sorted."""
    columns: list[pa.Array[Any]] = [
        pa.array([getattr(s, name) for s in samples], type=pa.string()) for name in TEXT_COLUMNS
    ]
    columns.append(pa.array([sorted(s.metadata.items()) for s in samples], type=METADATA_TYPE))
    columns.append(pa.array([s.created_at_utc for s in samples], type=TIMESTAMP_TYPE))
    return pa.RecordBatch.from_arrays(columns, schema=INSTRUCTION_SAMPLE_SCHEMA)


def samples_to_table(samples: Sequence[InstructionSample]) -> pa.Table:
    return pa.Table.from_batches([samples_to_record_batch(samples)])


def iter_rows(data: ArrowData) -> Iterator[dict[str, object]]:
    """Plain-Python rows of conformed data, with ``metadata`` decoded into a ``dict``."""
    for row in conform(data).to_pylist():
        row["metadata"] = dict(cast(list[tuple[str, str]], row["metadata"]))
        yield row


def iter_samples(data: ArrowData) -> Iterator[InstructionSample]:
    """Strictly re-validated samples. Raises ``ValidationError`` on the first bad row."""
    for row in iter_rows(data):
        yield InstructionSample.model_validate(row)


def table_to_samples(data: ArrowData) -> list[InstructionSample]:
    return list(iter_samples(data))


def table_to_polars(table: pa.Table) -> pl.DataFrame:
    """Conformed Polars view of ``table`` (zero-copy for fixed-width columns)."""
    return cast(pl.DataFrame, pl.from_arrow(conform(table), rechunk=False))


def polars_to_table(frame: pl.DataFrame) -> pa.Table:
    """Canonical Arrow table from a Polars frame, with full :func:`conform` checks."""
    return conform(frame.to_arrow())


def samples_to_polars(samples: Sequence[InstructionSample]) -> pl.DataFrame:
    return table_to_polars(samples_to_table(samples))


def polars_to_samples(frame: pl.DataFrame) -> list[InstructionSample]:
    return table_to_samples(polars_to_table(frame))
