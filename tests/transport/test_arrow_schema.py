"""Canonical Arrow schema: type stability, microsecond/UTC invariants, conform, Polars."""

from __future__ import annotations  # pyarrow-stubs generics are not subscriptable at runtime

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from synthetic_data_factory.schemas import InstructionSample
from synthetic_data_factory.transport.schema import (
    INSTRUCTION_SAMPLE_SCHEMA,
    METADATA_TYPE,
    SCHEMA_ID,
    SCHEMA_ID_KEY,
    SchemaMismatchError,
    conform,
    polars_to_samples,
    polars_to_table,
    samples_to_polars,
    samples_to_record_batch,
    samples_to_table,
    schema_matches,
    table_to_polars,
    table_to_samples,
)

S = INSTRUCTION_SAMPLE_SCHEMA
T0 = datetime(2026, 3, 1, 12, 30, 45, 123456, tzinfo=UTC)

texts = st.text(min_size=1, max_size=40).filter(lambda s: bool(s.strip()) and "\x00" not in s)
keys = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,11}", fullmatch=True)


@st.composite
def samples(draw: st.DrawFn) -> InstructionSample:
    return InstructionSample(
        sample_id=str(draw(st.uuids(version=4))),
        system_prompt=draw(texts),
        instruction=draw(texts),
        response=draw(texts),
        domain=draw(st.from_regex(r"[a-z0-9]{1,8}", fullmatch=True)),
        metadata=draw(st.dictionaries(keys, texts, max_size=6)),
        created_at_utc=draw(st.datetimes(timezones=st.just(UTC))),
    )


def make(n: int, **overrides: Any) -> InstructionSample:
    fields: dict[str, Any] = {
        "sample_id": str(uuid.UUID(int=n + 1, version=4)),
        "system_prompt": "sys",
        "instruction": f"instruction {n}",
        "response": f"response {n}",
        "domain": "test",
        "metadata": {"source": "unit", "batch": str(n)},
        "created_at_utc": T0 + timedelta(microseconds=n),
    }
    fields.update(overrides)
    return InstructionSample(**fields)


def parquet_round_trip(table: pa.Table) -> pa.Table:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return pq.read_table(pa.BufferReader(sink.getvalue()))


def replace_column(table: pa.Table, name: str, column: pa.Array[Any]) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(index, pa.field(name, column.type), column)


# --------------------------------------------------------------------------- #
# Schema definition                                                           #
# --------------------------------------------------------------------------- #


def test_schema_is_exactly_the_contract() -> None:
    expected = [
        ("sample_id", pa.string(), False),
        ("system_prompt", pa.string(), False),
        ("instruction", pa.string(), False),
        ("response", pa.string(), False),
        ("domain", pa.string(), False),
        ("metadata", pa.map_(pa.string(), pa.string(), True), False),
        ("created_at_utc", pa.timestamp("us", tz="UTC"), False),
    ]
    assert [(f.name, f.type, f.nullable) for f in S] == expected
    assert METADATA_TYPE.keys_sorted
    assert S.metadata == {SCHEMA_ID_KEY: SCHEMA_ID}


# --------------------------------------------------------------------------- #
# Round trips                                                                 #
# --------------------------------------------------------------------------- #


@given(st.lists(samples(), max_size=8))
def test_models_round_trip_through_arrow_exactly(batch: list[InstructionSample]) -> None:
    table = samples_to_table(batch)
    assert table.schema.equals(S, check_metadata=True)
    assert table_to_samples(table) == batch
    assert table_to_samples(samples_to_record_batch(batch)) == batch


@settings(max_examples=60)
@given(st.lists(samples(), min_size=1, max_size=8))
def test_models_round_trip_through_parquet_without_type_drift(
    batch: list[InstructionSample],
) -> None:
    restored = parquet_round_trip(samples_to_table(batch))
    assert schema_matches(restored.schema)
    assert table_to_samples(restored) == batch


@pytest.mark.parametrize(
    "moment",
    [
        datetime.min.replace(tzinfo=UTC),
        datetime.max.replace(tzinfo=UTC),  # 9999-12-31 23:59:59.999999
        datetime(1970, 1, 1, tzinfo=UTC) - timedelta(microseconds=1),  # negative epoch offset
        T0,
    ],
    ids=["min", "max", "pre-epoch", "typical"],
)
def test_microsecond_timestamps_survive_parquet_and_stay_utc(moment: datetime) -> None:
    restored = table_to_samples(
        parquet_round_trip(samples_to_table([make(0, created_at_utc=moment)]))
    )
    assert restored[0].created_at_utc == moment
    assert restored[0].created_at_utc.microsecond == moment.microsecond
    assert restored[0].created_at_utc.tzinfo is UTC


def test_metadata_is_written_sorted_and_compared_order_insensitively() -> None:
    sample = make(0, metadata={"zeta": "1", "alpha": "2", "mid": "3"})
    table = samples_to_table([sample])
    assert table.column("metadata").to_pylist() == [[("alpha", "2"), ("mid", "3"), ("zeta", "1")]]
    assert table_to_samples(table) == [sample]


def test_empty_batch_round_trips() -> None:
    table = samples_to_table([])
    assert table.num_rows == 0
    assert table.schema.equals(S, check_metadata=True)
    assert table_to_samples(parquet_round_trip(table)) == []


def test_conform_is_idempotent_and_preserves_type_identity() -> None:
    table = samples_to_table([make(n) for n in range(3)])
    once = conform(table)
    assert conform(once).equals(once)
    assert once.schema.equals(S, check_metadata=True)
    assert isinstance(conform(samples_to_record_batch([make(0)])), pa.RecordBatch)


def test_rows_that_conform_structurally_are_still_model_validated() -> None:
    table = replace_column(samples_to_table([make(0)]), "instruction", pa.array(["   "]))
    with pytest.raises(ValidationError):
        table_to_samples(table)


# --------------------------------------------------------------------------- #
# conform: accepted source types                                              #
# --------------------------------------------------------------------------- #


def test_conform_accepts_reordered_columns_and_alternative_string_layouts() -> None:
    table = samples_to_table([make(0), make(1)])
    alt = pa.table(
        {
            "created_at_utc": table.column("created_at_utc"),
            "metadata": table.column("metadata").cast(pa.map_(pa.large_string(), pa.string_view())),
            **{n: table.column(n).cast(pa.large_string()) for n in ("sample_id", "domain")},
            **{
                n: table.column(n).cast(pa.string_view())
                for n in ("system_prompt", "instruction", "response")
            },
        }
    )
    assert conform(alt).equals(table)


@pytest.mark.parametrize("tz", ["UTC", "Etc/UTC", "+00:00"])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_conform_accepts_utc_timestamps_in_any_unit_when_lossless(
    tz: str, unit: Literal["s", "ms", "us", "ns"]
) -> None:
    table = samples_to_table([make(0, created_at_utc=datetime(2026, 1, 1, tzinfo=UTC))])
    ts = table.column("created_at_utc").cast(pa.timestamp(unit, tz=tz))
    assert conform(replace_column(table, "created_at_utc", ts.combine_chunks())).equals(table)


def test_conform_accepts_unflagged_but_sorted_maps() -> None:
    table = samples_to_table([make(0)])
    unflagged = table.column("metadata").cast(pa.map_(pa.string(), pa.string()))
    assert conform(replace_column(table, "metadata", unflagged.combine_chunks())).equals(table)


# --------------------------------------------------------------------------- #
# conform: rejected drift                                                     #
# --------------------------------------------------------------------------- #


def _drifted(name: str, column: pa.Array[Any]) -> pa.Table:
    return replace_column(samples_to_table([make(0)]), name, column)


@pytest.mark.parametrize(
    ("name", "column", "message"),
    [
        ("sample_id", pa.array([7]), "disallowed type"),
        ("domain", pa.array([b"test"]), "disallowed type"),
        (
            "created_at_utc",
            pa.array([datetime(2026, 1, 1)], type=pa.timestamp("us")),
            "disallowed type",
        ),
        (
            "created_at_utc",
            pa.array([T0], type=pa.timestamp("us", tz="Europe/Warsaw")),
            "disallowed type",
        ),
        ("created_at_utc", pa.array([T0.date()]), "disallowed type"),
        (
            "metadata",
            pa.array([[("k", 1)]], type=pa.map_(pa.string(), pa.int64())),
            "disallowed type",
        ),
        (
            "metadata",
            pa.array(
                [[("k", "v")]], type=pa.list_(pa.struct([("k", pa.string()), ("v", pa.string())]))
            ),
            "disallowed type",
        ),
        ("instruction", pa.array([None], type=pa.string()), "null"),
        ("created_at_utc", pa.array([1], type=pa.timestamp("ns", tz="UTC")), "timestamp"),
        (
            "metadata",
            pa.array([[("b", "1"), ("a", "2")]], type=METADATA_TYPE),
            "strictly increasing",
        ),
        (
            "metadata",
            pa.array([[("a", "1"), ("a", "2")]], type=METADATA_TYPE),
            "strictly increasing",
        ),
    ],
    ids=[
        "int-to-string",
        "binary-to-string",
        "naive-timestamp",
        "non-utc-zone",
        "date-not-timestamp",
        "map-int-values",
        "list-not-map",
        "null-in-required",
        "sub-microsecond",
        "unsorted-keys",
        "duplicate-keys",
    ],
)
def test_conform_rejects_type_drift(name: str, column: pa.Array[Any], message: str) -> None:
    with pytest.raises(SchemaMismatchError, match=message):
        conform(_drifted(name, column))


def test_conform_rejects_missing_extra_and_duplicate_columns() -> None:
    table = samples_to_table([make(0)])
    with pytest.raises(SchemaMismatchError, match="expected columns"):
        conform(table.drop_columns(["domain"]))
    with pytest.raises(SchemaMismatchError, match="expected columns"):
        conform(table.append_column("quality", pa.array([0.9])))
    duplicated = pa.Table.from_arrays(
        [*table.columns, table.column("domain")], names=[*S.names, "domain"]
    )
    with pytest.raises(SchemaMismatchError, match="expected columns"):
        conform(duplicated)


def test_sorted_key_check_respects_slices() -> None:
    # The unsorted row is outside the slice; MapArray.keys would see it, list_flatten does not.
    column = pa.array([[("b", "1"), ("a", "2")], [("a", "1"), ("b", "2")], []], type=METADATA_TYPE)
    table = pa.Table.from_arrays(
        [
            *samples_to_table([make(n) for n in range(3)]).columns[:5],
            column,
            samples_to_table([make(n) for n in range(3)]).column("created_at_utc"),
        ],
        schema=S,
    )
    assert conform(table.slice(1)).num_rows == 2
    with pytest.raises(SchemaMismatchError):
        conform(table.slice(0, 1))


def test_key_order_is_checked_within_rows_not_across_them() -> None:
    rows = [make(0, metadata={"z": "1"}), make(1, metadata={"a": "1", "b": "2"})]
    assert table_to_samples(samples_to_table(rows)) == rows


def test_verify_sorted_keys_can_be_skipped_for_decode_only_callers() -> None:
    unsorted = _drifted("metadata", pa.array([[("b", "1"), ("a", "2")]], type=METADATA_TYPE))
    assert conform(unsorted, verify_sorted_keys=False).num_rows == 1


# --------------------------------------------------------------------------- #
# schema_matches                                                              #
# --------------------------------------------------------------------------- #


def test_schema_matches_tolerates_parquet_map_entry_renaming() -> None:
    restored = parquet_round_trip(samples_to_table([make(0)])).schema
    assert not restored.equals(S, check_metadata=True)  # entries struct renamed by Parquet
    assert schema_matches(restored)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: pa.schema(list(reversed(list(s))), metadata=s.metadata),
        lambda s: s.remove_metadata(),
        lambda s: s.set(0, s.field(0).with_nullable(True)),
        lambda s: s.set(5, pa.field("metadata", pa.map_(pa.string(), pa.string()), nullable=False)),
        lambda s: s.set(
            6, pa.field("created_at_utc", pa.timestamp("us", tz="Etc/UTC"), nullable=False)
        ),
        lambda s: s.set(
            6, pa.field("created_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False)
        ),
    ],
    ids=["order", "metadata", "nullability", "keys-sorted", "tz-spelling", "unit"],
)
def test_schema_matches_rejects_semantic_drift(mutate: Any) -> None:
    assert not schema_matches(mutate(S))


# --------------------------------------------------------------------------- #
# Polars                                                                      #
# --------------------------------------------------------------------------- #


def test_polars_dtypes_are_stable() -> None:
    frame = samples_to_polars([make(0)])
    assert frame.dtypes == [
        *[pl.String] * 5,
        pl.Map(pl.String, pl.String),
        pl.Datetime(time_unit="us", time_zone="UTC"),
    ]


@given(st.lists(samples(), max_size=6))
def test_polars_round_trip_is_exact(batch: list[InstructionSample]) -> None:
    table = samples_to_table(batch)
    assert polars_to_table(table_to_polars(table)).equals(table)
    assert polars_to_samples(samples_to_polars(batch)) == batch


def test_arrow_to_polars_is_zero_copy_for_fixed_width_columns() -> None:
    table = samples_to_table([make(n) for n in range(64)]).combine_chunks()
    original = table.column("created_at_utc").chunks[0].buffers()[1]
    restored = (
        polars_to_table(table_to_polars(table)).column("created_at_utc").chunks[0].buffers()[1]
    )
    assert original is not None and restored is not None
    assert restored.address == original.address


def test_polars_frames_with_unsorted_metadata_are_rejected() -> None:
    frame = pl.DataFrame(
        {name: ["x"] for name in ("system_prompt", "instruction", "response")}
        | {
            "sample_id": [str(uuid.UUID(int=1, version=4))],
            "domain": ["test"],
            "metadata": pl.Series([[{"key": "b", "value": "1"}, {"key": "a", "value": "2"}]]).cast(
                pl.Map(pl.String, pl.String)
            ),
            "created_at_utc": [T0],
        }
    )
    with pytest.raises(SchemaMismatchError, match="strictly increasing"):
        polars_to_table(frame)
