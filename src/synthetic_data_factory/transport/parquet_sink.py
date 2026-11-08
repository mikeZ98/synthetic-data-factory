"""Atomic, streaming Parquet sink.

Protocol
--------
1. ``open``: create ``<target dir>/.tmp.<uuid4>.parquet`` with ``O_EXCL`` and
   start a ``ParquetWriter`` on it. The leading dot hides the partial file from
   dataset globbing; Arrow datasets skip ``.``/``_`` prefixed files.
2. ``write``: append batches to a buffer. Whenever it holds at least
   ``row_group_size`` rows, write the full row groups and keep the remainder
   (slices are zero-copy). Every row group except the last therefore has
   exactly ``row_group_size`` rows, and buffered memory stays below
   ``row_group_size`` plus one incoming batch.
3. ``commit``: write the remainder, close the writer, ``fsync`` the file, and
   re-read the footer to check the Arrow schema (including metadata), the row
   count and the row-group layout. Then ``os.replace`` onto the target (atomic
   on POSIX) and ``fsync`` the directory so the rename is durable.
4. Any failure before the rename, a writer error, or an exception or
   ``CancelledError`` inside ``with`` triggers ``abort``: the temp file is
   removed and the target is untouched. Readers see either the old file or the
   complete new one, never a partial write.

``fsync`` on macOS does not flush the drive cache (that requires
``F_FULLFSYNC``). Atomicity is unaffected; power-loss durability is the
platform's ``fsync`` guarantee.
"""

import contextlib
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final, Self

import pyarrow as pa
import pyarrow.parquet as pq

from synthetic_data_factory.schemas import ExportSpecification, InstructionSample
from synthetic_data_factory.transport.schema import (
    INSTRUCTION_SAMPLE_SCHEMA,
    ArrowData,
    conform,
    samples_to_record_batch,
    schema_matches,
)

TMP_PREFIX: Final = ".tmp."


class SinkState(StrEnum):
    NEW = "new"
    OPEN = "open"
    COMMITTED = "committed"
    ABORTED = "aborted"


class ExportValidationError(RuntimeError):
    """The written file does not round-trip the expected schema, row count or layout."""


@dataclass(frozen=True, slots=True)
class ExportReport:
    path: Path
    rows: int
    row_groups: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _OpenFile:
    tmp: Path
    handle: BinaryIO
    writer: pq.ParquetWriter


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class AtomicParquetSink:
    """Streams samples into a Parquet file that appears atomically on commit.

    Use it as a context manager: a clean exit commits and an exception aborts.
    It is also a :data:`~synthetic_data_factory.core.pipeline.Sink`, so it can
    be passed straight to :class:`~synthetic_data_factory.core.CurationPipeline`.
    """

    def __init__(self, spec: ExportSpecification, *, root: Path | str = ".") -> None:
        self.spec: Final = spec
        self.path: Final = Path(root) / spec.output_path
        self.report: ExportReport | None = None
        self._state = SinkState.NEW
        self._open: _OpenFile | None = None  # set exactly while state is OPEN
        self._buffer: list[pa.RecordBatch] = []
        self._buffered = 0
        self._rows = 0

    @property
    def state(self) -> SinkState:
        return self._state

    @property
    def temp_path(self) -> Path | None:
        return None if self._open is None else self._open.tmp

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def open(self) -> Self:
        if self._state is not SinkState.NEW:
            raise RuntimeError(f"sink is {self._state}; a sink can be opened only once")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / f"{TMP_PREFIX}{uuid.uuid4().hex}.parquet"
        handle = tmp.open("xb")
        try:
            writer = pq.ParquetWriter(
                handle,
                INSTRUCTION_SAMPLE_SCHEMA,
                compression=self.spec.compression,
                store_schema=True,
            )
        except BaseException:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
        self._open = _OpenFile(tmp, handle, writer)
        self._state = SinkState.OPEN
        return self

    def commit(self) -> ExportReport:
        opened = self._resources()
        try:
            if self._buffered:
                self._flush(final=True)
            opened.writer.close()
            opened.handle.flush()
            os.fsync(opened.handle.fileno())
            opened.handle.close()
            report = self._verify(opened.tmp)
            os.replace(opened.tmp, self.path)
        except BaseException:
            self.abort()
            raise
        self._open = None
        self._state = SinkState.COMMITTED
        self.report = report
        _fsync_directory(self.path.parent)
        return report

    def abort(self) -> None:
        """Discard everything written. Idempotent, and a no-op once committed."""
        if self._state in (SinkState.COMMITTED, SinkState.ABORTED):
            return
        opened, self._open = self._open, None
        self._buffer, self._buffered = [], 0
        self._state = SinkState.ABORTED
        if opened is None:
            return
        try:
            # The data is being discarded; a failing footer write must not
            # mask the original error or skip the unlink below.
            with contextlib.suppress(Exception):
                opened.writer.close()
            # A failed close leaves ``is_open`` True, and ``ParquetWriter.__del__``
            # would retry the footer write against the closed handle at GC time.
            opened.writer.is_open = False
        finally:
            try:
                opened.handle.close()
            finally:
                opened.tmp.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.abort()

    # ------------------------------------------------------------------ #
    # Writes                                                              #
    # ------------------------------------------------------------------ #

    def write(self, samples: Sequence[InstructionSample]) -> None:
        self._resources()
        if samples:
            self._append(samples_to_record_batch(samples))

    def __call__(self, samples: Sequence[InstructionSample]) -> None:
        self.write(samples)

    def write_arrow(self, data: ArrowData) -> None:
        """Append Arrow data after full :func:`conform` checks (for example from Polars)."""
        self._resources()
        conformed = conform(data)
        batches = conformed.to_batches() if isinstance(conformed, pa.Table) else [conformed]
        for batch in batches:
            self._append(batch)

    def _append(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        self._buffer.append(batch)
        self._buffered += batch.num_rows
        if self._buffered >= self.spec.row_group_size:
            self._flush(final=False)

    def _flush(self, *, final: bool) -> None:
        writer = self._resources().writer
        size = self.spec.row_group_size
        table = pa.Table.from_batches(self._buffer, schema=INSTRUCTION_SAMPLE_SCHEMA)
        count = table.num_rows if final else table.num_rows // size * size
        try:
            writer.write_table(table.slice(0, count), row_group_size=size)
        except BaseException:
            self.abort()
            raise
        remainder = table.slice(count)
        self._buffer, self._buffered = remainder.to_batches(), remainder.num_rows
        self._rows += count

    # ------------------------------------------------------------------ #
    # Verification and helpers                                            #
    # ------------------------------------------------------------------ #

    def _verify(self, tmp: Path) -> ExportReport:
        with pq.ParquetFile(tmp) as parquet:
            schema, metadata = parquet.schema_arrow, parquet.metadata
            group_sizes = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
        problems = []
        if not schema_matches(schema):
            problems.append(f"schema drift: {schema} != {INSTRUCTION_SAMPLE_SCHEMA}")
        if metadata.num_rows != self._rows:
            problems.append(f"row count {metadata.num_rows} != {self._rows} written")
        size = self.spec.row_group_size
        if any(n != size for n in group_sizes[:-1]) or any(n > size for n in group_sizes[-1:]):
            problems.append(f"row groups {group_sizes} violate row_group_size={size}")
        if problems:
            raise ExportValidationError("; ".join(problems))
        return ExportReport(
            path=self.path,
            rows=metadata.num_rows,
            row_groups=metadata.num_row_groups,
            size_bytes=tmp.stat().st_size,
        )

    def _resources(self) -> _OpenFile:
        if self._open is None:
            raise RuntimeError(f"sink is {self._state}, expected {SinkState.OPEN}")
        return self._open
