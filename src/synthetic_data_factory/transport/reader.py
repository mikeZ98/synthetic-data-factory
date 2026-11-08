"""Bounded-memory streaming reader for JSONL, Parquet and in-memory sources.

Memory bounds
-------------
* **JSONL** is read one line at a time with ``readline(max_line_bytes + 1)``.
  A line longer than ``max_line_bytes`` is rejected as ``oversized`` and its
  remainder skipped in bounded chunks, so one pathological line cannot load an
  unbounded string. Peak memory is O(batch_size * max_line_bytes).
* **Parquet** goes through ``ParquetFile.iter_batches``, which decodes
  row groups incrementally. Peak memory is O(row group + batch).
* **In-memory** iterables (sync or async) are consumed lazily.

Reads are synchronous and run on the event loop between batch yields. This is
deliberate. Offloading to threads would let a cancelled read keep running
against a file the generator has already closed. Per-batch blocking is
bounded by ``batch_size``. Every source is closed deterministically when the
consumer stops early (``aclose``), not left to garbage collection.

Error tolerance
---------------
Malformed *records* never interrupt the stream. They are counted per
:class:`RejectReason` in :attr:`StreamingDatasetReader.stats`. Malformed
*sources* fail fast: a missing file, or a Parquet file whose column set or
types cannot conform, raises before the first batch.
"""

import asyncio
import contextlib
import functools
from collections import Counter
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import ValidationError

from synthetic_data_factory.schemas import InstructionSample
from synthetic_data_factory.transport.schema import (
    INSTRUCTION_SAMPLE_SCHEMA,
    SchemaMismatchError,
    conform,
)

DEFAULT_BATCH_SIZE: Final = 1_024
DEFAULT_MAX_LINE_BYTES: Final = 8 * 1024 * 1024

RawRecord = InstructionSample | Mapping[str, object] | str | bytes
RawBatches = AsyncGenerator[list[RawRecord], None]


class RejectReason(StrEnum):
    OVERSIZED = "oversized"  # JSONL line longer than max_line_bytes
    INVALID_JSON = "invalid_json"  # not parseable as JSON (incl. invalid UTF-8)
    SCHEMA = "schema"  # parsed, but violates the InstructionSample contract
    NULL_FIELD = "null_field"  # Parquet row with a null in a required column
    DUPLICATE_METADATA_KEY = "duplicate_metadata_key"  # Parquet map with repeated keys
    UNREPRESENTABLE = "unrepresentable"  # Parquet batch that cannot cast losslessly (e.g. ns)


@dataclass(slots=True)
class ReaderStats:
    records: int = 0  # records surfaced by the source, including rejected ones
    accepted: int = 0  # records that validated (only counted by ``batches()``)
    blank_lines: int = 0
    batches: int = 0
    rejected: Counter[RejectReason] = field(default_factory=Counter)

    @property
    def total_rejected(self) -> int:
        return self.rejected.total()


def classify(error: ValidationError) -> RejectReason:
    return (
        RejectReason.INVALID_JSON
        if any(item["type"] == "json_invalid" for item in error.errors())
        else RejectReason.SCHEMA
    )


def parse_record(record: RawRecord) -> InstructionSample:
    """Validate one raw record (model, JSON text/bytes, or mapping). Raises ``ValidationError``."""
    if isinstance(record, InstructionSample):
        return record
    if isinstance(record, str | bytes):
        return InstructionSample.model_validate_json(record)
    return InstructionSample.model_validate(record)


async def _aclose(iterator: object) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        await aclose()


class StreamingDatasetReader:
    """Single-use async reader. Build it with :meth:`jsonl`, :meth:`parquet` or :meth:`memory`."""

    def __init__(
        self,
        open_source: Callable[["StreamingDatasetReader"], RawBatches],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    ) -> None:
        for name, value in (("batch_size", batch_size), ("max_line_bytes", max_line_bytes)):
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self.batch_size: Final = batch_size
        self.max_line_bytes: Final = max_line_bytes
        self.stats: Final = ReaderStats()
        self._open_source: Final = open_source
        self._consumed = False

    @classmethod
    def jsonl(
        cls,
        path: Path | str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    ) -> "StreamingDatasetReader":
        resolved = Path(path)
        return cls(
            lambda reader: reader._jsonl(resolved),
            batch_size=batch_size,
            max_line_bytes=max_line_bytes,
        )

    @classmethod
    def parquet(
        cls, path: Path | str, *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> "StreamingDatasetReader":
        resolved = Path(path)
        return cls(lambda reader: reader._parquet(resolved), batch_size=batch_size)

    @classmethod
    def memory(
        cls,
        records: Iterable[RawRecord] | AsyncIterable[RawRecord],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> "StreamingDatasetReader":
        return cls(lambda reader: reader._memory(records), batch_size=batch_size)

    # ------------------------------------------------------------------ #
    # Public streams                                                      #
    # ------------------------------------------------------------------ #

    async def raw_batches(self) -> RawBatches:
        """Batches of unvalidated records, each of at most ``batch_size``."""
        if self._consumed:
            raise RuntimeError("StreamingDatasetReader is single-use")
        self._consumed = True
        async with contextlib.aclosing(self._open_source(self)) as stream:
            async for batch in stream:
                self.stats.batches += 1
                yield batch
                await asyncio.sleep(0)

    async def records(self) -> AsyncGenerator[RawRecord, None]:
        """Flat stream of unvalidated records: a ``CurationPipeline`` source."""
        async with contextlib.aclosing(self.raw_batches()) as batches:
            async for batch in batches:
                for record in batch:
                    yield record

    async def batches(self) -> AsyncGenerator[list[InstructionSample], None]:
        """Validated samples. Rejects are counted, never raised. Empty batches are skipped."""
        async with contextlib.aclosing(self.raw_batches()) as raw_batches:
            async for raw in raw_batches:
                valid: list[InstructionSample] = []
                for record in raw:
                    try:
                        valid.append(parse_record(record))
                    except ValidationError as error:
                        self.stats.rejected[classify(error)] += 1
                self.stats.accepted += len(valid)
                if valid:
                    yield valid

    # ------------------------------------------------------------------ #
    # Sources                                                             #
    # ------------------------------------------------------------------ #

    async def _jsonl(self, path: Path) -> RawBatches:
        limit = self.max_line_bytes
        with path.open("rb") as handle:
            batch: list[RawRecord] = []
            while line := handle.readline(limit + 1):
                if len(line) > limit and not line.endswith(b"\n"):
                    while line and not line.endswith(b"\n"):
                        line = handle.readline(limit + 1)
                    self.stats.records += 1
                    self.stats.rejected[RejectReason.OVERSIZED] += 1
                    continue
                record = line.strip()
                if not record:
                    self.stats.blank_lines += 1
                    continue
                self.stats.records += 1
                batch.append(record)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    async def _parquet(self, path: Path) -> RawBatches:
        with pq.ParquetFile(path) as parquet:
            # Structural check up front: columns and types must be conformable.
            conform(parquet.schema_arrow.empty_table(), verify_sorted_keys=False)
            for batch in parquet.iter_batches(batch_size=self.batch_size):
                self.stats.records += batch.num_rows
                rows = self._parquet_rows(batch)
                if rows:
                    yield rows

    def _parquet_rows(self, batch: pa.RecordBatch) -> list[RawRecord]:
        required = functools.reduce(
            pc.and_, [pc.is_valid(batch.column(name)) for name in INSTRUCTION_SAMPLE_SCHEMA.names]
        )
        complete = batch.filter(required)
        if dropped := batch.num_rows - complete.num_rows:
            self.stats.rejected[RejectReason.NULL_FIELD] += dropped
        try:
            # Foreign writers may store unsorted map keys; rows become dicts,
            # so only *duplicate* keys (which a dict would hide) are rejected.
            conformed = conform(complete, verify_sorted_keys=False)
        except SchemaMismatchError:
            self.stats.rejected[RejectReason.UNREPRESENTABLE] += complete.num_rows
            return []
        rows: list[RawRecord] = []
        for row in conformed.to_pylist():
            items = cast(list[tuple[str, str]], row["metadata"])
            metadata = dict(items)
            if len(metadata) != len(items):
                self.stats.rejected[RejectReason.DUPLICATE_METADATA_KEY] += 1
                continue
            row["metadata"] = metadata
            rows.append(row)
        return rows

    async def _memory(self, source: Iterable[RawRecord] | AsyncIterable[RawRecord]) -> RawBatches:
        stream = source if isinstance(source, AsyncIterable) else _as_async(source)
        batch: list[RawRecord] = []
        try:
            async for record in stream:
                self.stats.records += 1
                batch.append(record)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        finally:
            await _aclose(stream)
        if batch:
            yield batch


async def _as_async(records: Iterable[RawRecord]) -> AsyncIterator[RawRecord]:
    for record in records:
        yield record
