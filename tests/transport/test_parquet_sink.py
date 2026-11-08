"""AtomicParquetSink: atomic swap, rollback, cancellation hygiene, exact round trips."""

from __future__ import annotations  # pyarrow-stubs generics are not subscriptable at runtime

import asyncio
import itertools
import os
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthetic_data_factory.core import CurationPipeline, MinHashLSHIndex
from synthetic_data_factory.schemas import ExportSpecification, InstructionSample, MinHashConfig
from synthetic_data_factory.transport.parquet_sink import (
    AtomicParquetSink,
    ExportValidationError,
    SinkState,
)
from synthetic_data_factory.transport.schema import (
    INSTRUCTION_SAMPLE_SCHEMA,
    SchemaMismatchError,
    conform,
    polars_to_samples,
    polars_to_table,
    samples_to_polars,
    samples_to_table,
    schema_matches,
    table_to_samples,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
TMP_RE = re.compile(r"\.tmp\.[0-9a-f]{32}\.parquet")
RGS = 1_000


def make(n: int) -> InstructionSample:
    return InstructionSample(
        sample_id=str(uuid.UUID(int=n + 1, version=4)),
        system_prompt="sys",
        instruction=f"instruction number {n} " + "lorem " * (n % 7),
        response=f"response {n}",
        domain="test",
        metadata={"b": str(n), "a": "x"},
        created_at_utc=T0 + timedelta(microseconds=n * 997),
    )


SAMPLES = [make(n) for n in range(3_500)]
TABLE = samples_to_table(SAMPLES)


def spec(path: str = "out/data.parquet", **overrides: Any) -> ExportSpecification:
    fields: dict[str, Any] = {"output_path": path, "compression": "zstd", "row_group_size": RGS}
    fields.update(overrides)
    return ExportSpecification(**fields)


def stray_temp_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.name.startswith(".tmp.")]


def group_sizes(path: Path) -> list[int]:
    metadata = pq.ParquetFile(path).metadata
    return [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]


# --------------------------------------------------------------------------- #
# Round trips                                                                 #
# --------------------------------------------------------------------------- #


def test_round_trip_via_pyarrow_and_polars_scan_is_exact(tmp_path: Path) -> None:
    with AtomicParquetSink(spec(), root=tmp_path) as sink:
        for start in range(0, len(SAMPLES), 333):
            sink.write(SAMPLES[start : start + 333])
    assert sink.state is SinkState.COMMITTED
    assert sink.report is not None
    assert (sink.report.rows, sink.report.row_groups) == (3_500, 4)
    assert sink.report.size_bytes == sink.path.stat().st_size
    table = pq.read_table(sink.path)
    assert schema_matches(table.schema)
    assert table_to_samples(table) == SAMPLES
    assert polars_to_samples(pl.scan_parquet(sink.path).collect()) == SAMPLES
    assert stray_temp_files(tmp_path) == []


@pytest.mark.parametrize("codec", ["zstd", "snappy", "gzip"])
def test_compression_codec_is_applied(tmp_path: Path, codec: str) -> None:
    with AtomicParquetSink(spec(compression=codec), root=tmp_path) as sink:
        sink.write(SAMPLES[:10])
    column_chunk = pq.ParquetFile(sink.path).metadata.row_group(0).column(0)
    assert column_chunk.compression == codec.upper()


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=1_700), max_size=6))
def test_row_groups_are_full_except_the_last(
    tmp_path_factory: pytest.TempPathFactory, chunk_sizes: list[int]
) -> None:
    root = tmp_path_factory.mktemp("layout")
    offsets = [0, *itertools.accumulate(chunk_sizes)]
    total = min(offsets[-1], TABLE.num_rows)
    with AtomicParquetSink(spec(), root=root) as sink:
        for start, end in itertools.pairwise(offsets):
            sink.write_arrow(TABLE.slice(start, max(0, min(end, total) - start)))
    sizes = group_sizes(sink.path)
    assert sum(sizes) == total
    assert all(size == RGS for size in sizes[:-1])
    assert all(0 < size <= RGS for size in sizes[-1:])
    # conform() maps Parquet's renamed map-entries struct back to the canonical type
    assert conform(pq.read_table(sink.path)).equals(TABLE.slice(0, total))


def test_write_arrow_accepts_polars_and_record_batches(tmp_path: Path) -> None:
    with AtomicParquetSink(spec(), root=tmp_path) as sink:
        sink.write_arrow(polars_to_table(samples_to_polars(SAMPLES[:5])))
        sink.write_arrow(TABLE.slice(5, 5).to_batches()[0])
    assert table_to_samples(pq.read_table(sink.path)) == SAMPLES[:10]


def test_empty_commit_produces_valid_zero_row_file(tmp_path: Path) -> None:
    with AtomicParquetSink(spec(), root=tmp_path) as sink:
        sink.write([])
        sink.write_arrow(pa.RecordBatch.from_pylist([], schema=INSTRUCTION_SAMPLE_SCHEMA))
        sink.write_arrow(TABLE.slice(0, 0))
    table = pq.read_table(sink.path)
    assert table.num_rows == 0
    assert schema_matches(table.schema)


def test_parent_directories_are_created(tmp_path: Path) -> None:
    with AtomicParquetSink(spec("a/b/c/out.parquet"), root=tmp_path) as sink:
        sink.write(SAMPLES[:1])
    assert (tmp_path / "a" / "b" / "c" / "out.parquet").is_file()


# --------------------------------------------------------------------------- #
# Atomicity                                                                   #
# --------------------------------------------------------------------------- #


def test_atomic_swap_old_file_stays_readable_until_commit(tmp_path: Path) -> None:
    with AtomicParquetSink(spec(), root=tmp_path) as old:
        old.write(SAMPLES[:3])
    sink = AtomicParquetSink(spec(), root=tmp_path).open()
    sink.write(SAMPLES[:2_500])
    temp = sink.temp_path
    assert temp is not None
    assert TMP_RE.fullmatch(temp.name)
    assert temp.parent == sink.path.parent  # same directory: rename stays atomic
    assert temp.is_file()
    assert table_to_samples(pq.read_table(sink.path)) == SAMPLES[:3]  # readers still see old
    sink.commit()
    assert table_to_samples(pq.read_table(sink.path)) == SAMPLES[:2_500]
    assert sink.temp_path is None
    assert stray_temp_files(tmp_path) == []


def test_commit_fsyncs_file_and_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    with AtomicParquetSink(spec(), root=tmp_path) as sink:
        sink.write(SAMPLES[:1])
    assert len(calls) == 2


@pytest.fixture
def previous_file(tmp_path: Path) -> Iterator[Path]:
    with AtomicParquetSink(spec(), root=tmp_path) as sink:
        sink.write(SAMPLES[:3])
    yield sink.path
    assert table_to_samples(pq.read_table(sink.path)) == SAMPLES[:3], "old file must be intact"
    assert stray_temp_files(tmp_path) == [], "no dangling .tmp files"


def test_exception_in_body_rolls_back(previous_file: Path, tmp_path: Path) -> None:
    sink = AtomicParquetSink(spec(), root=tmp_path)
    with pytest.raises(KeyError), sink:
        sink.write(SAMPLES[:1_500])
        raise KeyError("caller bug")
    assert sink.state is SinkState.ABORTED


def test_writer_failure_mid_stream_auto_aborts(
    previous_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("simulated disk failure")

    sink = AtomicParquetSink(spec(), root=tmp_path).open()
    monkeypatch.setattr(pq.ParquetWriter, "write_table", fail)
    with pytest.raises(OSError, match="simulated disk failure"):
        sink.write(SAMPLES[:RGS])
    assert sink.state is SinkState.ABORTED
    with pytest.raises(RuntimeError, match="aborted"):
        sink.commit()


def test_writer_construction_failure_leaves_nothing(
    previous_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_writer(*args: object, **kwargs: object) -> None:
        raise OSError("cannot initialise writer")

    monkeypatch.setattr(pq, "ParquetWriter", broken_writer)
    with pytest.raises(OSError, match="cannot initialise"):
        AtomicParquetSink(spec(), root=tmp_path).open()


def test_fsync_failure_during_commit_rolls_back(
    previous_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(fd: int) -> None:
        raise OSError("fsync failed")

    sink = AtomicParquetSink(spec(), root=tmp_path).open()
    sink.write(SAMPLES[:10])
    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="fsync failed"):
        sink.commit()
    assert sink.state is SinkState.ABORTED


def test_footer_failure_during_abort_does_not_mask_cleanup(
    previous_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = AtomicParquetSink(spec(), root=tmp_path).open()
    sink.write(SAMPLES[:10])

    def fail(self: object) -> None:
        raise OSError("footer write failed")

    monkeypatch.setattr(pq.ParquetWriter, "close", fail)
    sink.abort()
    monkeypatch.undo()
    assert sink.state is SinkState.ABORTED


@pytest.mark.asyncio
async def test_cancellation_mid_pipeline_leaves_no_temp_files(tmp_path: Path) -> None:
    gate = asyncio.Event()

    async def endless() -> AsyncIterator[InstructionSample]:
        for n in itertools.count():
            if n == 1_500:
                gate.set()
            yield make(n)

    sink = AtomicParquetSink(spec(), root=tmp_path)

    async def run() -> None:
        with sink:
            index = MinHashLSHIndex(MinHashConfig(threshold=0.99, num_perm=16, shingle_k=2))
            await CurationPipeline(index, sink=sink, batch_size=250).run(endless())

    task = asyncio.create_task(run())
    await asyncio.wait_for(gate.wait(), 10)
    assert stray_temp_files(tmp_path) != []  # a partial temp file exists mid-run
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sink.state is SinkState.ABORTED
    assert stray_temp_files(tmp_path) == []
    assert not sink.path.exists()


# --------------------------------------------------------------------------- #
# Round-trip validation                                                       #
# --------------------------------------------------------------------------- #


class _FakeRowGroup:
    def __init__(self, rows: int) -> None:
        self.num_rows = rows


class _FakeMetadata:
    def __init__(self, groups: list[int]) -> None:
        self._groups = groups
        self.num_rows = sum(groups)
        self.num_row_groups = len(groups)

    def row_group(self, i: int) -> _FakeRowGroup:
        return _FakeRowGroup(self._groups[i])


def _fake_parquet_file(schema: pa.Schema, groups: list[int]) -> type:
    class FakeParquetFile:
        def __init__(self, path: object) -> None:
            self.schema_arrow = schema
            self.metadata = _FakeMetadata(groups)

        def __enter__(self) -> FakeParquetFile:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    return FakeParquetFile


@pytest.mark.parametrize(
    ("schema", "groups", "message"),
    [
        (INSTRUCTION_SAMPLE_SCHEMA.remove_metadata(), [10], "schema drift"),
        (INSTRUCTION_SAMPLE_SCHEMA, [9], "row count"),
        (INSTRUCTION_SAMPLE_SCHEMA, [4, 6], "row groups"),
        (INSTRUCTION_SAMPLE_SCHEMA, [RGS + 1, 0], "row groups"),
    ],
    ids=["schema", "rows", "short-group", "oversized-group"],
)
def test_verification_rejects_drift_and_rolls_back(
    previous_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: pa.Schema,
    groups: list[int],
    message: str,
) -> None:
    sink = AtomicParquetSink(spec(), root=tmp_path).open()
    sink.write(SAMPLES[:10])
    monkeypatch.setattr(pq, "ParquetFile", _fake_parquet_file(schema, groups))
    with pytest.raises(ExportValidationError, match=message):
        sink.commit()
    monkeypatch.undo()
    assert sink.state is SinkState.ABORTED


# --------------------------------------------------------------------------- #
# Lifecycle                                                                   #
# --------------------------------------------------------------------------- #


def test_lifecycle_misuse_is_rejected(tmp_path: Path) -> None:
    sink = AtomicParquetSink(spec(), root=tmp_path)
    assert sink.state is SinkState.NEW and sink.temp_path is None
    with pytest.raises(RuntimeError, match="expected open"):
        sink.write(SAMPLES[:1])
    sink.open()
    with pytest.raises(RuntimeError, match="opened only once"):
        sink.open()
    sink.commit()
    with pytest.raises(RuntimeError, match="committed"):
        sink.write_arrow(TABLE.slice(0, 1))
    sink.abort()  # no-op once committed
    assert sink.path.exists()


def test_abort_before_open_is_terminal(tmp_path: Path) -> None:
    sink = AtomicParquetSink(spec(), root=tmp_path)
    sink.abort()
    sink.abort()
    assert sink.state is SinkState.ABORTED
    with pytest.raises(RuntimeError, match="opened only once"):
        sink.open()


def test_write_arrow_rejects_non_conforming_data_without_aborting(tmp_path: Path) -> None:
    with AtomicParquetSink(spec(), root=tmp_path) as sink:
        with pytest.raises(SchemaMismatchError):
            sink.write_arrow(TABLE.drop_columns(["domain"]))
        assert sink.state is SinkState.OPEN
        sink.write(SAMPLES[:1])
    assert pq.read_table(sink.path).num_rows == 1


@pytest.mark.asyncio
async def test_sink_plugs_into_curation_pipeline(tmp_path: Path) -> None:
    async def source() -> AsyncIterator[InstructionSample]:
        for sample in [*SAMPLES[:50], *SAMPLES[:50]]:  # second half is exact duplicates
            yield sample

    index = MinHashLSHIndex(MinHashConfig(threshold=0.9, num_perm=64, shingle_k=2))
    with AtomicParquetSink(spec(), root=tmp_path) as sink:
        metrics = await CurationPipeline(index, sink=sink, batch_size=16).run(source())
    assert metrics.raw_count == 100
    assert metrics.deduped_count == pq.read_table(sink.path).num_rows == 50
