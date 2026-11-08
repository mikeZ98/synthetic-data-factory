"""Asynchronous curation pipeline: bounded ingestion, validation, and deduplication.

Topology (one event loop, structured concurrency)
-------------------------------------------------
::

    source --(producer task)--> bounded Queue --(run() body)--> validate --> dedup --> sink
                                                      \\--> LatestValue --(emitter task)--> hook

* **Backpressure / bounded memory.** The producer awaits ``Queue.put``, so at
  most ``queue_maxsize`` raw records are buffered, plus one in-flight batch of
  at most ``batch_size``.
* **No orphaned tasks.** The producer and emitter are children of an
  ``asyncio.TaskGroup``. On any exit path (success, failure, cancellation) the
  group cancels and awaits every child before ``run`` returns or raises. An
  async-generator source is closed with ``aclose()``.
* **Race freedom.** All pipeline state is mutated only from the ``run`` body
  task, and index mutations contain no ``await``. No interleaving can observe
  a half-applied transition.
* **Deadlock freedom.** There is one blocking edge in each direction: producer
  ``put`` blocks on a full queue, consumer ``get`` blocks on an empty one. The
  producer ends with an end-of-stream sentinel *after* closing the source, and
  a producer failure cancels the consumer through the TaskGroup. So a consumer
  blocked on ``get`` always wakes up.
* **Non-blocking metrics.** Snapshots are published into a latest-value slot,
  which never blocks the publisher. A dedicated emitter task delivers them to
  the hook. A slow hook sees conflated (skipped) intermediate snapshots, but
  always receives the terminal one.

State machine
-------------
``IDLE -> INGESTING -> CURATING -> DEDUPLICATING -> (INGESTING | COMPLETED)``,
and ``FAILED`` is reachable from every non-terminal state. One cycle runs per
micro-batch. A pipeline instance runs once.
"""

import asyncio
import functools
import inspect
import logging
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import ValidationError

from synthetic_data_factory.core.deduplicator import MinHashLSHIndex
from synthetic_data_factory.schemas import CurationMetrics, InstructionSample

_LOGGER: Final = logging.getLogger(__name__)

DEFAULT_QUEUE_MAXSIZE: Final = 1_024
DEFAULT_BATCH_SIZE: Final = 256


class PipelineState(StrEnum):
    IDLE = "idle"
    INGESTING = "ingesting"
    CURATING = "curating"
    DEDUPLICATING = "deduplicating"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATES: Final = frozenset({PipelineState.COMPLETED, PipelineState.FAILED})

TRANSITIONS: Final[Mapping[PipelineState, frozenset[PipelineState]]] = {
    PipelineState.IDLE: frozenset({PipelineState.INGESTING, PipelineState.FAILED}),
    PipelineState.INGESTING: frozenset({PipelineState.CURATING, PipelineState.FAILED}),
    PipelineState.CURATING: frozenset({PipelineState.DEDUPLICATING, PipelineState.FAILED}),
    PipelineState.DEDUPLICATING: frozenset(
        {PipelineState.INGESTING, PipelineState.COMPLETED, PipelineState.FAILED}
    ),
    PipelineState.COMPLETED: frozenset(),
    PipelineState.FAILED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    """Raised on a state transition that is not in :data:`TRANSITIONS`."""


class PipelineStateMachine:
    """Explicit, append-only record of pipeline states."""

    __slots__ = ("_history",)

    def __init__(self) -> None:
        self._history: list[PipelineState] = [PipelineState.IDLE]

    @property
    def state(self) -> PipelineState:
        return self._history[-1]

    @property
    def history(self) -> tuple[PipelineState, ...]:
        return tuple(self._history)

    def transition(self, target: PipelineState) -> None:
        if target not in TRANSITIONS[self.state]:
            raise InvalidTransitionError(f"illegal transition {self.state} -> {target}")
        self._history.append(target)


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """Point-in-time view emitted to the metrics hook."""

    state: PipelineState
    metrics: CurationMetrics
    batches: int


RawRecord = InstructionSample | Mapping[str, object] | str | bytes
Sink = Callable[[Sequence[InstructionSample]], Awaitable[None] | None]
MetricsHook = Callable[[PipelineSnapshot], Awaitable[None] | None]


class _EndOfStream:
    __slots__ = ()


_END: Final = _EndOfStream()


async def _call[T](callback: Callable[[T], Awaitable[None] | None], argument: T) -> None:
    result = callback(argument)
    if inspect.isawaitable(result):
        await result


class _LatestValue:
    """Single-slot, overwrite-on-publish channel. ``publish`` never blocks.

    Invariant: the event is set if and only if ``_value`` has been published
    and not yet delivered. ``close`` only raises a flag and must follow the
    ``publish`` of the terminal value. The drainer therefore always delivers
    that terminal value before it returns, even if it was mid-delivery when
    the value was published.
    """

    __slots__ = ("_closed", "_event", "_value")

    def __init__(self, initial: PipelineSnapshot) -> None:
        self._value = initial
        self._closed = False
        self._event = asyncio.Event()

    def publish(self, value: PipelineSnapshot) -> None:
        self._value = value
        self._event.set()

    def close(self) -> None:
        self._closed = True

    async def drain(self, deliver: Callable[[PipelineSnapshot], Awaitable[None]]) -> None:
        while True:
            await self._event.wait()
            self._event.clear()
            await deliver(self._value)
            if self._closed and not self._event.is_set():
                return


class CurationPipeline:
    """Single-use async runner: ingest -> validate -> deduplicate -> sink."""

    def __init__(
        self,
        index: MinHashLSHIndex,
        *,
        sink: Sink,
        metrics_hook: MetricsHook | None = None,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        for name, value in (("queue_maxsize", queue_maxsize), ("batch_size", batch_size)):
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._index: Final = index
        self._sink: Final = sink
        self._hook: Final = metrics_hook
        self._queue_maxsize: Final = queue_maxsize
        self._batch_size: Final = batch_size
        self._machine: Final = PipelineStateMachine()
        self._raw = 0
        self._valid = 0
        self._kept = 0
        self._batches = 0
        self.hook_errors = 0
        self._channel: Final = _LatestValue(self.snapshot())

    @property
    def state(self) -> PipelineState:
        return self._machine.state

    @property
    def history(self) -> tuple[PipelineState, ...]:
        return self._machine.history

    def snapshot(self) -> PipelineSnapshot:
        return PipelineSnapshot(
            state=self.state,
            metrics=CurationMetrics.from_counts(
                raw_count=self._raw, schema_valid_count=self._valid, deduped_count=self._kept
            ),
            batches=self._batches,
        )

    async def run(self, source: AsyncIterable[RawRecord]) -> CurationMetrics:
        """Drive ``source`` to completion and return the final funnel metrics.

        Raises :class:`InvalidTransitionError` if the pipeline has already run.
        On failure the state becomes ``FAILED`` and the original exception is
        re-raised, chained from the TaskGroup's exception group. Cancellation
        moves to ``FAILED`` and propagates ``CancelledError``.
        """
        self._transition(PipelineState.INGESTING)
        queue: asyncio.Queue[RawRecord | _EndOfStream] = asyncio.Queue(self._queue_maxsize)
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._produce(source, queue), name="sdf-producer")
                if self._hook is not None:
                    deliver = functools.partial(self._deliver, self._hook)
                    group.create_task(self._channel.drain(deliver), name="sdf-metrics")
                await self._consume(queue)
                self._transition(PipelineState.COMPLETED)
                self._channel.close()
        except BaseExceptionGroup as failure:
            self._fail()
            if self._hook is not None:
                await self._deliver(self._hook, self.snapshot())
            raise failure.exceptions[0] from failure
        except BaseException:  # CancelledError, KeyboardInterrupt, SystemExit
            self._fail()
            raise
        return self.snapshot().metrics

    def _transition(self, target: PipelineState) -> None:
        self._machine.transition(target)
        self._channel.publish(self.snapshot())

    def _fail(self) -> None:
        # Cancellation can arrive after COMPLETED, while the final metrics
        # snapshot is being flushed. Processing did finish, so the state stays
        # COMPLETED rather than raising InvalidTransitionError over the
        # CancelledError.
        if self.state not in TERMINAL_STATES:
            self._transition(PipelineState.FAILED)

    async def _produce(
        self, source: AsyncIterable[RawRecord], queue: asyncio.Queue[RawRecord | _EndOfStream]
    ) -> None:
        iterator = aiter(source)
        try:
            async for record in iterator:
                await queue.put(record)
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()
        await queue.put(_END)

    async def _consume(self, queue: asyncio.Queue[RawRecord | _EndOfStream]) -> None:
        while True:
            batch, ended = await self._take_batch(queue)
            self._transition(PipelineState.CURATING)
            valid = self._validate(batch)
            self._transition(PipelineState.DEDUPLICATING)
            kept, _ = self._index.deduplicate_batch(valid)
            self._kept += len(kept)
            self._batches += 1
            if kept:
                await _call(self._sink, kept)
            if ended:
                return
            self._transition(PipelineState.INGESTING)
            await asyncio.sleep(0)  # fairness: let the producer and emitter run

    async def _take_batch(
        self, queue: asyncio.Queue[RawRecord | _EndOfStream]
    ) -> tuple[list[RawRecord], bool]:
        batch: list[RawRecord] = []
        while len(batch) < self._batch_size:
            item = await queue.get()
            if isinstance(item, _EndOfStream):
                return batch, True
            batch.append(item)
        return batch, False

    def _validate(self, batch: Sequence[RawRecord]) -> list[InstructionSample]:
        valid: list[InstructionSample] = []
        for record in batch:
            self._raw += 1
            try:
                if isinstance(record, InstructionSample):
                    sample = record
                elif isinstance(record, str | bytes):
                    sample = InstructionSample.model_validate_json(record)
                else:
                    sample = InstructionSample.model_validate(record)
            except ValidationError:
                continue
            valid.append(sample)
        self._valid += len(valid)
        return valid

    async def _deliver(self, hook: MetricsHook, snapshot: PipelineSnapshot) -> None:
        try:
            await _call(hook, snapshot)
        except Exception:
            self.hook_errors += 1
            _LOGGER.exception("metrics hook failed; pipeline continues")
