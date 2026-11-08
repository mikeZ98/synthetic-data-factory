"""Stateful fuzzing of AtomicParquetSink: atomic visibility and zero temp-file leaks.

After every step of an arbitrary interleaving of opens, writes, commits,
aborts and injected I/O failures:

* the target holds exactly the last *committed* content (or does not exist);
* at most one ``.tmp.*`` file exists while a sink is open, none otherwise;
* a failed write or commit leaves the sink ABORTED and the target untouched.
"""

import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq
import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from synthetic_data_factory.schemas import ExportSpecification, InstructionSample
from synthetic_data_factory.transport.parquet_sink import AtomicParquetSink, SinkState

T0 = datetime(2026, 1, 1, tzinfo=UTC)
RGS = 1_000
SPEC = ExportSpecification(output_path="out/data.parquet", compression="zstd", row_group_size=RGS)
POOL = [
    InstructionSample(
        sample_id=str(uuid.UUID(int=n + 1, version=4)),
        system_prompt="sys",
        instruction=f"instruction {n}",
        response="ok",
        domain="fuzz",
        created_at_utc=T0 + timedelta(microseconds=n),
    )
    for n in range(4_000)
]
FUZZ_ROOT = Path(".cache") / "fuzz"


class SinkMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        FUZZ_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=FUZZ_ROOT))
        self.sink: AtomicParquetSink | None = None
        self.pending: list[str] = []
        self.committed: list[str] | None = None
        self.cursor = 0

    def _open(self) -> bool:
        return self.sink is not None and self.sink.state is SinkState.OPEN

    def _take(self, n: int) -> list[InstructionSample]:
        batch = [POOL[(self.cursor + i) % len(POOL)] for i in range(n)]
        self.cursor += n
        return batch

    @precondition(lambda self: not self._open())
    @rule()
    def open_new_sink(self) -> None:
        self.sink = AtomicParquetSink(SPEC, root=self.root).open()
        self.pending = []

    @precondition(lambda self: self._open())
    @rule(n=st.integers(0, 1_300))
    def write(self, n: int) -> None:
        assert self.sink is not None
        batch = self._take(n)
        self.sink.write(batch)
        self.pending += [s.sample_id for s in batch]

    @precondition(lambda self: self._open())
    @rule()
    def commit(self) -> None:
        assert self.sink is not None
        report = self.sink.commit()
        assert report.rows == len(self.pending)
        self.committed = self.pending

    @precondition(lambda self: self._open())
    @rule()
    def abort(self) -> None:
        assert self.sink is not None
        self.sink.abort()
        assert self.sink.state is SinkState.ABORTED

    @precondition(lambda self: self._open())
    @rule()
    def write_fails_mid_stream(self) -> None:
        assert self.sink is not None and self.sink._open is not None
        writer = self.sink._open.writer
        with (
            mock.patch.object(writer, "write_table", side_effect=OSError("disk failure")),
            pytest.raises(OSError, match="disk failure"),
        ):
            self.sink.write(self._take(RGS))  # a full row group forces a flush
        assert self.sink.state is SinkState.ABORTED

    @precondition(lambda self: self._open())
    @rule()
    def commit_fails_at_fsync(self) -> None:
        assert self.sink is not None
        with (
            mock.patch("os.fsync", side_effect=OSError("fsync failure")),
            pytest.raises(OSError, match="fsync failure"),
        ):
            self.sink.commit()
        assert self.sink.state is SinkState.ABORTED

    @invariant()
    def target_holds_exactly_the_last_commit(self) -> None:
        target = self.root / SPEC.output_path
        if self.committed is None:
            assert not target.exists()
        else:
            assert (
                pq.read_table(target, columns=["sample_id"]).column(0).to_pylist() == self.committed
            )

    @invariant()
    def temp_files_never_leak(self) -> None:
        temps = [p for p in self.root.rglob(".tmp.*")]
        assert len(temps) == (1 if self._open() else 0)

    def teardown(self) -> None:
        if self.sink is not None:
            self.sink.abort()
        assert [p for p in self.root.rglob(".tmp.*")] == []
        shutil.rmtree(self.root)


TestSinkMachine = SinkMachine.TestCase
TestSinkMachine.settings = settings(max_examples=40, stateful_step_count=20, deadline=None)


def test_fuzz_root_is_inside_the_cache() -> None:
    assert FUZZ_ROOT.parts[0] == ".cache"
    assert not os.path.isabs(FUZZ_ROOT)
