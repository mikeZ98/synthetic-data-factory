"""CurationPipeline: state machine invariants, backpressure, cancellation, zero-leak drain."""

import asyncio
import itertools
import json
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthetic_data_factory.core.deduplicator import MinHashLSHIndex, canonical_order
from synthetic_data_factory.core.pipeline import (
    TERMINAL_STATES,
    TRANSITIONS,
    CurationPipeline,
    InvalidTransitionError,
    PipelineSnapshot,
    PipelineState,
    PipelineStateMachine,
    RawRecord,
)
from synthetic_data_factory.schemas import InstructionSample, MinHashConfig

S = PipelineState
CONFIG = MinHashConfig(threshold=0.7, num_perm=64, shingle_k=3)
T0 = datetime(2026, 1, 1, tzinfo=UTC)
# Completed runs: full cycles then K. Failed runs: full cycles, a (possibly
# partial) cycle prefix, then F. Every run starts IDLE -> INGESTING.
HISTORY_RE = re.compile(r"I(?:NCD)+K|I(?:NCD)*N(?:C(?:D)?)?F")
LETTER = {
    S.IDLE: "I",
    S.INGESTING: "N",
    S.CURATING: "C",
    S.DEDUPLICATING: "D",
    S.COMPLETED: "K",
    S.FAILED: "F",
}
TIMEOUT = 10.0


def make_record(n: int, *, text: str | None = None) -> dict[str, object]:
    return {
        "sample_id": str(uuid.UUID(int=n + 1, version=4)),
        "system_prompt": "You are helpful.",
        "instruction": text if text is not None else " ".join(f"w{n}x{i}" for i in range(12)),
        "response": "ok",
        "domain": "test",
        "created_at_utc": T0 + timedelta(seconds=n),
    }


FIRST_TEXT = " ".join(f"w0x{i}" for i in range(12))


def to_json(record: dict[str, object]) -> str:
    return json.dumps(
        {k: v.isoformat() if isinstance(v, datetime) else v for k, v in record.items()}
    )


def encoded(history: Sequence[PipelineState]) -> str:
    return "".join(LETTER[state] for state in history)


async def from_list(records: Sequence[RawRecord]) -> AsyncIterator[RawRecord]:
    for record in records:
        yield record


class Collector:
    def __init__(self) -> None:
        self.batches: list[list[InstructionSample]] = []

    async def __call__(self, batch: Sequence[InstructionSample]) -> None:
        self.batches.append(list(batch))

    @property
    def samples(self) -> list[InstructionSample]:
        return [s for batch in self.batches for s in batch]


def assert_no_leaked_tasks() -> None:
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def until(predicate: Callable[[], bool], *, limit: int = 10_000) -> None:
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


# --------------------------------------------------------------------------- #
# State machine                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("source", "target"), list(itertools.product(S, S)))
def test_transition_table_is_enforced_exhaustively(source: S, target: S) -> None:
    machine = PipelineStateMachine()
    machine._history[-1] = source  # place the machine in an arbitrary state
    if target in TRANSITIONS[source]:
        machine.transition(target)
        assert machine.state is target
    else:
        with pytest.raises(InvalidTransitionError, match="illegal transition"):
            machine.transition(target)
        assert machine.state is source


def test_terminal_states_have_no_exits_and_every_state_can_fail() -> None:
    assert set(TRANSITIONS) == set(S)
    for state in S:
        if state in TERMINAL_STATES:
            assert TRANSITIONS[state] == frozenset()
        else:
            assert S.FAILED in TRANSITIONS[state]


def test_history_is_an_immutable_append_only_view() -> None:
    machine = PipelineStateMachine()
    machine.transition(S.INGESTING)
    history = machine.history
    machine.transition(S.CURATING)
    assert history == (S.IDLE, S.INGESTING)
    assert machine.history == (S.IDLE, S.INGESTING, S.CURATING)


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["queue_maxsize", "batch_size"])
@pytest.mark.parametrize("value", [0, -1, True])
def test_rejects_invalid_sizes(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        CurationPipeline(MinHashLSHIndex(CONFIG), sink=Collector(), **{field: value})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Happy paths                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_validates_deduplicates_and_reports_metrics() -> None:
    sample = InstructionSample.model_validate(make_record(1))
    records: list[RawRecord] = [
        make_record(0),
        sample,  # already a model
        to_json(make_record(2)),  # JSON text
        to_json(make_record(3)).encode(),  # JSON bytes
        {**make_record(4), "sample_id": "not-a-uuid"},  # schema-invalid
        "{not json",  # malformed JSON
        make_record(5, text=FIRST_TEXT),  # exact duplicate of record 0, created later
    ]
    sink = Collector()
    pipeline = CurationPipeline(MinHashLSHIndex(CONFIG), sink=sink, batch_size=3)
    metrics = await pipeline.run(from_list(records))
    assert (metrics.raw_count, metrics.schema_valid_count, metrics.deduped_count) == (7, 5, 4)
    assert [s.sample_id for s in sink.samples] == [
        str(uuid.UUID(int=n + 1, version=4)) for n in range(4)
    ]
    assert pipeline.state is S.COMPLETED
    assert HISTORY_RE.fullmatch(encoded(pipeline.history))
    assert pipeline.snapshot().batches == 3
    assert_no_leaked_tasks()


@pytest.mark.asyncio
async def test_empty_source_completes_one_empty_cycle_without_calling_sink() -> None:
    sink = Collector()
    pipeline = CurationPipeline(MinHashLSHIndex(CONFIG), sink=sink)
    metrics = await pipeline.run(from_list([]))
    assert pipeline.history == (S.IDLE, S.INGESTING, S.CURATING, S.DEDUPLICATING, S.COMPLETED)
    assert sink.batches == []
    assert metrics.raw_count == 0


@pytest.mark.asyncio
async def test_sync_sink_and_sync_hook_are_supported() -> None:
    received: list[InstructionSample] = []
    snapshots: list[PipelineSnapshot] = []
    pipeline = CurationPipeline(
        MinHashLSHIndex(CONFIG),
        sink=received.extend,
        metrics_hook=snapshots.append,
        batch_size=2,
    )
    await pipeline.run(from_list([make_record(n) for n in range(5)]))
    assert len(received) == 5
    assert snapshots[-1].state is S.COMPLETED
    assert snapshots[-1].metrics.deduped_count == 5
    assert snapshots[-1].batches == 3


@pytest.mark.asyncio
async def test_pipeline_is_single_use() -> None:
    pipeline = CurationPipeline(MinHashLSHIndex(CONFIG), sink=Collector())
    await pipeline.run(from_list([]))
    with pytest.raises(InvalidTransitionError):
        await pipeline.run(from_list([]))
    assert pipeline.state is S.COMPLETED


@pytest.mark.asyncio
async def test_source_without_aclose_is_supported() -> None:
    class PlainIterator:
        def __init__(self) -> None:
            self._items = iter([make_record(0), make_record(1)])

        def __aiter__(self) -> "PlainIterator":
            return self

        async def __anext__(self) -> RawRecord:
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration from None

    metrics = await CurationPipeline(MinHashLSHIndex(CONFIG), sink=Collector()).run(PlainIterator())
    assert metrics.deduped_count == 2


@pytest.mark.asyncio
async def test_sink_receives_each_batch_in_canonical_order() -> None:
    sink = Collector()
    records = [make_record(n) for n in reversed(range(6))]
    await CurationPipeline(MinHashLSHIndex(CONFIG), sink=sink, batch_size=3).run(from_list(records))
    for batch in sink.batches:
        assert batch == sorted(batch, key=canonical_order)


# --------------------------------------------------------------------------- #
# Backpressure and non-blocking metrics                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ingestion_buffer_is_bounded_under_a_stalled_consumer() -> None:
    queue_maxsize, batch_size = 4, 2
    produced = 0
    gate = asyncio.Event()

    async def source() -> AsyncIterator[RawRecord]:
        nonlocal produced
        for n in range(100):
            produced += 1
            yield make_record(n)

    async def stalled_sink(batch: Sequence[InstructionSample]) -> None:
        await gate.wait()

    pipeline = CurationPipeline(
        MinHashLSHIndex(CONFIG),
        sink=stalled_sink,
        queue_maxsize=queue_maxsize,
        batch_size=batch_size,
    )
    task = asyncio.create_task(pipeline.run(source()))
    await until(lambda: produced > 0)
    for _ in range(200):
        await asyncio.sleep(0)
    # consumed batch + full queue + the one record the producer holds while blocked in put()
    assert produced <= batch_size + queue_maxsize + 1
    gate.set()
    metrics = await asyncio.wait_for(task, TIMEOUT)
    assert metrics.raw_count == 100
    assert_no_leaked_tasks()


@pytest.mark.asyncio
async def test_slow_hook_never_blocks_processing_and_gets_terminal_snapshot() -> None:
    gate = asyncio.Event()
    delivered: list[PipelineSnapshot] = []

    async def slow_hook(snapshot: PipelineSnapshot) -> None:
        delivered.append(snapshot)
        await gate.wait()

    sink = Collector()
    pipeline = CurationPipeline(
        MinHashLSHIndex(CONFIG), sink=sink, metrics_hook=slow_hook, batch_size=1
    )
    task = asyncio.create_task(pipeline.run(from_list([make_record(n) for n in range(20)])))
    await until(lambda: pipeline.state is S.COMPLETED)
    assert len(sink.samples) == 20  # all processing done while the hook is stuck
    assert len(delivered) == 1
    gate.set()
    await asyncio.wait_for(task, TIMEOUT)
    assert delivered[-1].state is S.COMPLETED
    assert delivered[-1].metrics.deduped_count == 20
    assert len(delivered) < len(pipeline.history)  # intermediate snapshots were conflated
    assert_no_leaked_tasks()


@pytest.mark.asyncio
async def test_failing_hook_is_isolated() -> None:
    def broken_hook(snapshot: PipelineSnapshot) -> None:
        raise RuntimeError("hook exploded")

    pipeline = CurationPipeline(MinHashLSHIndex(CONFIG), sink=Collector(), metrics_hook=broken_hook)
    metrics = await pipeline.run(from_list([make_record(0)]))
    assert metrics.deduped_count == 1
    assert pipeline.state is S.COMPLETED
    assert pipeline.hook_errors >= 1


# --------------------------------------------------------------------------- #
# Failure, cancellation, and zero-leak drain                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_source_failure_moves_to_failed_and_reraises_original() -> None:
    closed = False
    snapshots: list[PipelineSnapshot] = []

    async def failing_source() -> AsyncIterator[RawRecord]:
        nonlocal closed
        try:
            for n in range(3):
                yield make_record(n)
            raise ValueError("source broke")
        finally:
            closed = True

    pipeline = CurationPipeline(
        MinHashLSHIndex(CONFIG), sink=Collector(), metrics_hook=snapshots.append, batch_size=100
    )
    with pytest.raises(ValueError, match="source broke") as excinfo:
        await pipeline.run(failing_source())
    assert isinstance(excinfo.value.__cause__, BaseExceptionGroup)
    assert pipeline.state is S.FAILED
    assert snapshots[-1].state is S.FAILED
    assert closed
    assert HISTORY_RE.fullmatch(encoded(pipeline.history))
    assert_no_leaked_tasks()


@pytest.mark.asyncio
async def test_sink_failure_cancels_producer_and_closes_source() -> None:
    closed = False

    async def endless() -> AsyncIterator[RawRecord]:
        nonlocal closed
        try:
            for n in itertools.count():
                yield make_record(n)
        finally:
            closed = True

    async def failing_sink(batch: Sequence[InstructionSample]) -> None:
        raise OSError("disk full")

    pipeline = CurationPipeline(
        MinHashLSHIndex(CONFIG), sink=failing_sink, queue_maxsize=2, batch_size=2
    )
    with pytest.raises(OSError, match="disk full"):
        await asyncio.wait_for(pipeline.run(endless()), TIMEOUT)
    assert pipeline.state is S.FAILED
    assert closed
    assert_no_leaked_tasks()


@pytest.mark.asyncio
async def test_cancellation_mid_run_leaves_zero_orphaned_tasks() -> None:
    closed = False
    sink_entered = asyncio.Event()

    async def endless() -> AsyncIterator[RawRecord]:
        nonlocal closed
        try:
            for n in itertools.count():
                yield make_record(n)
        finally:
            closed = True

    async def hanging_sink(batch: Sequence[InstructionSample]) -> None:
        sink_entered.set()
        await asyncio.Event().wait()  # never completes

    pipeline = CurationPipeline(
        MinHashLSHIndex(CONFIG),
        sink=hanging_sink,
        metrics_hook=lambda snapshot: None,
        queue_maxsize=2,
        batch_size=2,
    )
    task = asyncio.create_task(pipeline.run(endless()))
    await asyncio.wait_for(sink_entered.wait(), TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pipeline.state is S.FAILED
    assert HISTORY_RE.fullmatch(encoded(pipeline.history))
    assert closed
    assert_no_leaked_tasks()


@pytest.mark.asyncio
async def test_cancellation_during_final_flush_keeps_completed() -> None:
    async def hook(snapshot: PipelineSnapshot) -> None:
        if snapshot.state is S.COMPLETED:
            await asyncio.Event().wait()  # stall the terminal delivery

    pipeline = CurationPipeline(MinHashLSHIndex(CONFIG), sink=Collector(), metrics_hook=hook)
    task = asyncio.create_task(pipeline.run(from_list([make_record(0)])))
    await until(lambda: pipeline.state is S.COMPLETED)
    for _ in range(50):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pipeline.state is S.COMPLETED  # processing finished; no illegal COMPLETED -> FAILED
    assert_no_leaked_tasks()


# --------------------------------------------------------------------------- #
# Deadlock freedom (property)                                                 #
# --------------------------------------------------------------------------- #


@settings(max_examples=60, deadline=None)
@given(
    n_records=st.integers(0, 40),
    queue_maxsize=st.integers(1, 5),
    batch_size=st.integers(1, 7),
    duplicate_every=st.integers(1, 5),
    yield_in_sink=st.booleans(),
    with_hook=st.booleans(),
)
def test_every_configuration_terminates_with_consistent_metrics(
    n_records: int,
    queue_maxsize: int,
    batch_size: int,
    duplicate_every: int,
    yield_in_sink: bool,
    with_hook: bool,
) -> None:
    shared_text = "the same shared instruction text for duplicates"
    records = [
        make_record(n, text=shared_text if n % duplicate_every == 0 else None)
        for n in range(n_records)
    ]
    snapshots: list[PipelineSnapshot] = []

    async def scenario() -> None:
        received: list[InstructionSample] = []

        async def sink(batch: Sequence[InstructionSample]) -> None:
            received.extend(batch)
            if yield_in_sink:
                await asyncio.sleep(0)

        pipeline = CurationPipeline(
            MinHashLSHIndex(CONFIG),
            sink=sink,
            metrics_hook=snapshots.append if with_hook else None,
            queue_maxsize=queue_maxsize,
            batch_size=batch_size,
        )
        metrics = await asyncio.wait_for(pipeline.run(from_list(records)), TIMEOUT)
        duplicates = sum(1 for n in range(n_records) if n % duplicate_every == 0)
        expected_kept = n_records - max(0, duplicates - 1)
        assert (metrics.raw_count, metrics.schema_valid_count) == (n_records, n_records)
        assert metrics.deduped_count == len(received) == expected_kept
        assert len({s.sample_id for s in received}) == len(received)
        assert pipeline.snapshot().batches == n_records // batch_size + 1
        assert HISTORY_RE.fullmatch(encoded(pipeline.history))
        assert pipeline.history[-1] is S.COMPLETED
        assert_no_leaked_tasks()

    asyncio.run(scenario())
    if with_hook:
        assert snapshots[-1].state is S.COMPLETED
