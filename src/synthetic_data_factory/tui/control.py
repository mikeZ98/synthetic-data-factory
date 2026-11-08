"""Pause, resume and graceful stop for the record stream feeding the pipeline.

Control is applied at the *source*, so the pipeline needs no extra states.

* **Pause** blocks the stream before the next record. The bounded queue
  drains, then the pipeline idles on ``Queue.get``. Pause takes effect within
  at most ``queue_maxsize + batch_size`` records.
* **Stop** ends the stream. The pipeline processes what is already queued,
  completes through its normal end-of-stream path, and the atomic sink
  commits. Nothing is cancelled and no partial file can appear.

Stop is only observed when the source produces its next record. If upstream
is itself blocked (a hung reader), graceful stop cannot complete; that is what
forced cancellation (a second quit or signal) is for.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Sequence

from synthetic_data_factory.schemas import InstructionSample
from synthetic_data_factory.transport.reader import StreamingDatasetReader


class SourceController:
    """Gates a chain of readers. Safe to drive from key bindings and signal handlers."""

    __slots__ = ("_running", "_stopped", "yielded")

    def __init__(self) -> None:
        self._running = asyncio.Event()
        self._running.set()
        self._stopped = False
        self.yielded = 0  # records handed to the pipeline

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    @property
    def stop_requested(self) -> bool:
        return self._stopped

    def pause(self) -> None:
        if not self._stopped:  # pausing after stop would block the final drain
            self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def stop(self) -> None:
        self._stopped = True
        self._running.set()  # wake a paused stream so it can observe the stop

    async def stream(
        self, readers: Sequence[StreamingDatasetReader]
    ) -> AsyncGenerator[InstructionSample, None]:
        """Validated samples from each reader in turn, honouring pause and stop."""
        for reader in readers:
            async with contextlib.aclosing(reader.batches()) as batches:
                async for batch in batches:
                    for sample in batch:
                        await self._running.wait()
                        if self._stopped:
                            return
                        self.yielded += 1
                        yield sample
