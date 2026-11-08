"""StreamingDatasetReader: chunking, EOF, error tolerance, deterministic close, memory bounds."""

from __future__ import annotations  # pyarrow-stubs generics are not subscriptable at runtime

import asyncio
import json
import os
import tracemalloc
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthetic_data_factory.core import CurationPipeline, MinHashLSHIndex
from synthetic_data_factory.schemas import InstructionSample, MinHashConfig
from synthetic_data_factory.transport.reader import (
    RawRecord,
    RejectReason,
    StreamingDatasetReader,
    parse_record,
)
from synthetic_data_factory.transport.schema import (
    INSTRUCTION_SAMPLE_SCHEMA,
    METADATA_TYPE,
    SchemaMismatchError,
    samples_to_table,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
R = RejectReason


def record(n: int, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "sample_id": str(uuid.UUID(int=n + 1, version=4)),
        "system_prompt": "sys",
        "instruction": f"instruction {n}",
        "response": f"response {n}",
        "domain": "test",
        "metadata": {"n": str(n)},
        "created_at_utc": (T0 + timedelta(microseconds=n)).isoformat(),
    }
    fields.update(overrides)
    return fields


def sample(n: int) -> InstructionSample:
    return InstructionSample.model_validate_json(json.dumps(record(n)))


def jsonl_line(n: int, **overrides: Any) -> bytes:
    return json.dumps(record(n, **overrides)).encode() + b"\n"


def write(path: Path, *chunks: bytes) -> Path:
    path.write_bytes(b"".join(chunks))
    return path


async def collect(reader: StreamingDatasetReader) -> list[list[InstructionSample]]:
    return [batch async for batch in reader.batches()]


def open_fds() -> int:
    return len(os.listdir("/dev/fd"))


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("option", ["batch_size", "max_line_bytes"])
@pytest.mark.parametrize("value", [0, -1, True])
def test_rejects_invalid_options(tmp_path: Path, option: str, value: int) -> None:
    with pytest.raises(ValueError, match=option):
        StreamingDatasetReader.jsonl(tmp_path / "x.jsonl", **{option: value})


@pytest.mark.asyncio
async def test_reader_is_single_use() -> None:
    reader = StreamingDatasetReader.memory([])
    assert await collect(reader) == []
    with pytest.raises(RuntimeError, match="single-use"):
        await collect(reader)


# --------------------------------------------------------------------------- #
# JSONL                                                                       #
# --------------------------------------------------------------------------- #


@settings(max_examples=40, deadline=None)
@given(n=st.integers(0, 60), batch_size=st.integers(1, 17), trailing_newline=st.booleans())
def test_jsonl_chunking_is_exact(
    tmp_path_factory: pytest.TempPathFactory, n: int, batch_size: int, trailing_newline: bool
) -> None:
    body = b"".join(jsonl_line(i) for i in range(n))
    if not trailing_newline:
        body = body.rstrip(b"\n")  # EOF without a final newline
    path = write(tmp_path_factory.mktemp("jsonl") / "data.jsonl", body)
    reader = StreamingDatasetReader.jsonl(path, batch_size=batch_size)
    batches = asyncio.run(collect(reader))
    assert [len(b) for b in batches] == [batch_size] * (n // batch_size) + (
        [n % batch_size] if n % batch_size else []
    )
    assert [s for b in batches for s in b] == [sample(i) for i in range(n)]
    assert (reader.stats.records, reader.stats.accepted, reader.stats.total_rejected) == (n, n, 0)


@pytest.mark.asyncio
async def test_jsonl_tolerates_malformed_records_without_interrupting(tmp_path: Path) -> None:
    path = write(
        tmp_path / "mixed.jsonl",
        jsonl_line(0),
        b"\n   \n",  # blank lines are skipped, not rejected
        b"{not json\n",
        b'{"sample_id": "\xff\xfe"}\n',  # invalid UTF-8
        jsonl_line(1, sample_id="not-a-uuid"),
        jsonl_line(2, created_at_utc=datetime(2026, 1, 1).isoformat()),  # naive timestamp
        jsonl_line(3).replace(b"\n", b"\r\n"),  # CRLF
        b"[1, 2, 3]\n",
        jsonl_line(4),
    )
    reader = StreamingDatasetReader.jsonl(path, batch_size=2)
    batches = await collect(reader)
    assert [s for b in batches for s in b] == [sample(0), sample(3), sample(4)]
    assert reader.stats.rejected == {R.INVALID_JSON: 2, R.SCHEMA: 3}
    assert (reader.stats.records, reader.stats.blank_lines, reader.stats.accepted) == (8, 2, 3)


@pytest.mark.asyncio
async def test_oversized_lines_are_rejected_in_bounded_chunks(tmp_path: Path) -> None:
    limit = 1_024
    exact = jsonl_line(1, instruction="x" * 600)
    assert len(exact) <= limit + 1
    path = write(
        tmp_path / "big.jsonl",
        jsonl_line(0),
        b'{"instruction": "' + b"y" * (limit * 5) + b'"}\n',  # oversized mid-file
        exact,  # exactly at the limit is fine
        jsonl_line(2),
        b"z" * (limit + 10),  # oversized at EOF, no newline
    )
    reader = StreamingDatasetReader.jsonl(path, max_line_bytes=limit)
    batches = await collect(reader)
    assert [s.sample_id for b in batches for s in b] == [sample(n).sample_id for n in range(3)]
    assert reader.stats.rejected == {R.OVERSIZED: 2}
    assert reader.stats.records == 5


@pytest.mark.asyncio
async def test_line_of_exactly_max_bytes_without_newline_is_accepted(tmp_path: Path) -> None:
    line = jsonl_line(0).rstrip(b"\n")
    path = write(tmp_path / "edge.jsonl", line)
    reader = StreamingDatasetReader.jsonl(path, max_line_bytes=len(line))
    assert await collect(reader) == [[sample(0)]]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"", b"\n\n  \n"], ids=["empty", "blank-only"])
async def test_empty_inputs_yield_nothing(tmp_path: Path, content: bytes) -> None:
    reader = StreamingDatasetReader.jsonl(write(tmp_path / "e.jsonl", content))
    assert await collect(reader) == []
    assert reader.stats.batches == 0


@pytest.mark.asyncio
async def test_missing_file_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await collect(StreamingDatasetReader.jsonl(tmp_path / "absent.jsonl"))


@pytest.mark.asyncio
async def test_all_rejected_batches_are_skipped(tmp_path: Path) -> None:
    path = write(tmp_path / "bad.jsonl", b"{}\n{}\n", jsonl_line(0))
    reader = StreamingDatasetReader.jsonl(path, batch_size=2)
    assert await collect(reader) == [[sample(0)]]
    assert reader.stats.batches == 2  # raw batches read
    assert reader.stats.rejected == {R.SCHEMA: 2}


@pytest.mark.asyncio
async def test_early_break_closes_the_file_deterministically(tmp_path: Path) -> None:
    path = write(tmp_path / "many.jsonl", *(jsonl_line(n) for n in range(100)))
    baseline = open_fds()
    stream = StreamingDatasetReader.jsonl(path, batch_size=10).batches()
    first = await anext(stream)
    assert len(first) == 10
    assert open_fds() == baseline + 1  # the file is open mid-stream
    await stream.aclose()
    assert open_fds() == baseline


@pytest.mark.asyncio
async def test_records_feed_the_curation_pipeline(tmp_path: Path) -> None:
    path = write(
        tmp_path / "feed.jsonl",
        *(jsonl_line(n) for n in range(30)),
        b"{broken\n",
        jsonl_line(0),  # exact duplicate (same id) of the first record
    )
    reader = StreamingDatasetReader.jsonl(path, batch_size=7)
    kept: list[InstructionSample] = []
    index = MinHashLSHIndex(MinHashConfig(threshold=0.95, num_perm=64, shingle_k=2))
    metrics = await CurationPipeline(index, sink=kept.extend, batch_size=8).run(reader.records())
    assert (metrics.raw_count, metrics.schema_valid_count, metrics.deduped_count) == (32, 31, 30)
    assert kept == [sample(n) for n in range(30)]


def test_jsonl_streaming_memory_is_bounded(tmp_path: Path) -> None:
    pad = "p" * 400
    path = write(tmp_path / "large.jsonl", *(jsonl_line(n, response=pad) for n in range(20_000)))
    file_size = path.stat().st_size

    async def drain() -> int:
        rows = 0
        async for batch in StreamingDatasetReader.jsonl(path, batch_size=100).raw_batches():
            rows += len(batch)
        return rows

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        assert asyncio.run(drain()) == 20_000
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert file_size > 9_000_000
    assert peak < file_size / 20, f"peak {peak} B for a {file_size} B file"


# --------------------------------------------------------------------------- #
# Parquet                                                                     #
# --------------------------------------------------------------------------- #


def write_parquet(path: Path, table: pa.Table, **options: Any) -> Path:
    pq.write_table(table, path, **options)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 300, 1_000, 5_000])
async def test_parquet_chunking_and_exact_round_trip(tmp_path: Path, batch_size: int) -> None:
    expected = [sample(n) for n in range(2_345)]
    path = write_parquet(tmp_path / "d.parquet", samples_to_table(expected), row_group_size=1_000)
    reader = StreamingDatasetReader.parquet(path, batch_size=batch_size)
    batches = await collect(reader)
    assert all(len(b) <= batch_size for b in batches)
    assert [s for b in batches for s in b] == expected
    assert reader.stats.records == reader.stats.accepted == 2_345


def _foreign(columns: dict[str, pa.Array[Any]], nullable: bool = True) -> pa.Table:
    return pa.Table.from_arrays(
        list(columns.values()),
        schema=pa.schema([pa.field(n, c.type, nullable=nullable) for n, c in columns.items()]),
    )


def _base_columns(n: int) -> dict[str, pa.Array[Any]]:
    table = samples_to_table([sample(i) for i in range(n)])
    return {name: table.column(name).combine_chunks() for name in INSTRUCTION_SAMPLE_SCHEMA.names}


@pytest.mark.asyncio
async def test_parquet_rows_with_nulls_or_duplicate_keys_are_rejected_individually(
    tmp_path: Path,
) -> None:
    columns = _base_columns(5)
    columns["instruction"] = pa.array(["a", None, "c", "d", "e"])
    columns["metadata"] = pa.array(
        [
            [("n", "0")],
            [("n", "1")],
            [("k", "1"), ("k", "2")],
            [("b", "1"), ("a", "2")],
            [("n", "4")],
        ],
        type=pa.map_(pa.string(), pa.string()),  # foreign writer: keys not flagged sorted
    )
    path = write_parquet(tmp_path / "foreign.parquet", _foreign(columns))
    reader = StreamingDatasetReader.parquet(path)
    batches = await collect(reader)
    ids = [s.sample_id for b in batches for s in b]
    assert ids == [sample(n).sample_id for n in (0, 3, 4)]  # unsorted-but-unique keys are fine
    assert batches[0][1].metadata == {"a": "2", "b": "1"}
    assert reader.stats.rejected == {R.NULL_FIELD: 1, R.DUPLICATE_METADATA_KEY: 1}


@pytest.mark.asyncio
async def test_unrepresentable_batch_is_rejected_and_stream_continues(tmp_path: Path) -> None:
    columns = _base_columns(4)
    nanos = [
        int((T0 - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 10**9) + n
        for n in (0, 1000, 2001, 3000)
    ]
    columns["created_at_utc"] = pa.array(nanos, type=pa.timestamp("ns", tz="UTC"))
    path = write_parquet(tmp_path / "ns.parquet", _foreign(columns), coerce_timestamps=None)
    reader = StreamingDatasetReader.parquet(path, batch_size=2)
    batches = await collect(reader)
    assert [s.sample_id for b in batches for s in b] == [sample(0).sample_id, sample(1).sample_id]
    assert reader.stats.rejected == {R.UNREPRESENTABLE: 2}  # batch holding the 2001 ns row


@pytest.mark.asyncio
async def test_model_invalid_parquet_rows_are_counted(tmp_path: Path) -> None:
    columns = _base_columns(3)
    columns["domain"] = pa.array(["test", "NOT A SLUG", "test"])
    reader = StreamingDatasetReader.parquet(
        write_parquet(tmp_path / "m.parquet", _foreign(columns, False))
    )
    assert [len(b) for b in await collect(reader)] == [2]
    assert reader.stats.rejected == {R.SCHEMA: 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["extra", "missing", "type"])
async def test_structurally_incompatible_parquet_fails_fast(tmp_path: Path, change: str) -> None:
    columns: dict[str, pa.Array[Any]] = dict(_base_columns(2))
    if change == "extra":
        columns["score"] = pa.array([0.1, 0.2])
    elif change == "missing":
        del columns["domain"]
    else:
        columns["sample_id"] = pa.array([1, 2])
    path = write_parquet(tmp_path / "s.parquet", _foreign(columns))
    reader = StreamingDatasetReader.parquet(path)
    with pytest.raises(SchemaMismatchError):
        await collect(reader)
    assert reader.stats.records == 0


@pytest.mark.asyncio
async def test_parquet_early_break_closes_the_file(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "d.parquet", samples_to_table([sample(n) for n in range(50)]))
    baseline = open_fds()
    stream = StreamingDatasetReader.parquet(path, batch_size=5).batches()
    assert len(await anext(stream)) == 5
    await stream.aclose()
    assert open_fds() == baseline


def _synthetic_parquet(path: Path, rows: int) -> int:
    """Write ``rows`` conforming rows (built column-wise, no models) and return decoded bytes."""
    table = pa.Table.from_arrays(
        [
            pa.array([str(uuid.UUID(int=n + 1, version=4)) for n in range(rows)]),
            pa.array(["sys"] * rows),
            pa.array([f"instruction {n}" for n in range(rows)]),
            pa.array([f"{n:08d}" * 64 for n in range(rows)]),  # ~0.5 KiB, poorly compressible
            pa.array(["test"] * rows),
            pa.array([[("n", str(n))] for n in range(rows)], type=METADATA_TYPE),
            pa.array(
                [T0 + timedelta(microseconds=n) for n in range(rows)],
                type=pa.timestamp("us", tz="UTC"),
            ),
        ],
        schema=INSTRUCTION_SAMPLE_SCHEMA,
    )
    pq.write_table(table, path, row_group_size=1_000, compression="zstd")
    return table.nbytes


def _peak_arrow_bytes(path: Path) -> int:
    peak = 0

    async def drain() -> None:
        nonlocal peak
        async for _ in StreamingDatasetReader.parquet(path, batch_size=500).raw_batches():
            peak = max(peak, pa.total_allocated_bytes())

    baseline = pa.total_allocated_bytes()
    asyncio.run(drain())
    return peak - baseline


def test_parquet_streaming_memory_does_not_grow_with_file_size(tmp_path: Path) -> None:
    small = _synthetic_parquet(tmp_path / "small.parquet", 10_000)
    large = _synthetic_parquet(tmp_path / "large.parquet", 40_000)
    assert large > 4 * small * 0.99 and large > 20_000_000
    peak_small = _peak_arrow_bytes(tmp_path / "small.parquet")
    peak_large = _peak_arrow_bytes(tmp_path / "large.parquet")
    # O(row group + batch): a 4x larger file must not raise the peak materially.
    assert peak_large < peak_small * 1.5 + 512 * 1024, (peak_small, peak_large)
    assert peak_large < large / 8, f"arrow peak {peak_large} B vs {large} B decoded"


# --------------------------------------------------------------------------- #
# In-memory sources                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_memory_source_accepts_all_raw_record_kinds() -> None:
    raw: Sequence[RawRecord] = [
        sample(0),
        record(1) | {"created_at_utc": T0 + timedelta(microseconds=1)},
        json.dumps(record(2)),
        json.dumps(record(3)).encode(),
        {"bogus": True},
    ]
    reader = StreamingDatasetReader.memory(raw, batch_size=2)
    assert [s for b in await collect(reader) for s in b] == [sample(n) for n in range(4)]
    assert reader.stats.rejected == {R.SCHEMA: 1}


@pytest.mark.asyncio
async def test_memory_async_generator_is_closed_on_early_exit() -> None:
    closed = False

    async def source() -> AsyncIterator[RawRecord]:
        nonlocal closed
        try:
            for n in range(1_000):
                yield sample(n)
        finally:
            closed = True

    stream = StreamingDatasetReader.memory(source(), batch_size=3).records()
    assert parse_record(await anext(stream)) == sample(0)
    await stream.aclose()
    assert closed


@pytest.mark.asyncio
async def test_memory_async_iterator_without_aclose_is_supported() -> None:
    class Plain:
        def __init__(self) -> None:
            self._items = iter([sample(0), sample(1)])

        def __aiter__(self) -> Plain:
            return self

        async def __anext__(self) -> RawRecord:
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration from None

    assert await collect(StreamingDatasetReader.memory(Plain())) == [[sample(0), sample(1)]]


@pytest.mark.asyncio
async def test_metadata_map_type_survives_reader(tmp_path: Path) -> None:
    columns = _base_columns(1)
    columns["metadata"] = pa.array([[("a", "1"), ("b", "2")]], type=METADATA_TYPE)
    reader = StreamingDatasetReader.parquet(
        write_parquet(tmp_path / "md.parquet", _foreign(columns, False))
    )
    assert (await collect(reader))[0][0].metadata == {"a": "1", "b": "2"}
