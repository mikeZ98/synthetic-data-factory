"""Live curation run in a reactive terminal UI.

Lifecycle
---------
The pipeline runs as a Textual worker on the UI event loop. Index mutations
contain no ``await``, and the pipeline yields between batches, so the UI
repaints at least once per batch. The metrics hook is synchronous and O(1): it
stores the snapshot and pushes plain values into reactive widgets. Pipeline
conflation guarantees that a slow repaint can never stall processing.

Interruption policy:

* ``c`` / ``SIGUSR1``: graceful stop. Already-queued records are processed,
  the atomic sink commits a valid file, and the UI stays open on the result.
* ``q`` / ``ctrl+q`` / ``SIGINT`` / ``SIGTERM``: the same graceful stop, then
  exit with code 0 (1 if the run failed).
* A second ``q`` or signal while stopping forces cancellation. The sink aborts
  (no partial file, no ``.tmp`` left behind) and the exit code is 130.
"""

import asyncio
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Footer, Header, Static
from textual.worker import Worker, WorkerState

from synthetic_data_factory.core.deduplicator import MinHashLSHIndex
from synthetic_data_factory.core.pipeline import CurationPipeline, PipelineSnapshot, PipelineState
from synthetic_data_factory.schemas import CurationMetrics, ExportSpecification, MinHashConfig
from synthetic_data_factory.transport.parquet_sink import AtomicParquetSink
from synthetic_data_factory.transport.reader import StreamingDatasetReader
from synthetic_data_factory.tui.control import SourceController
from synthetic_data_factory.tui.telemetry import CurationTelemetry, compose_telemetry
from synthetic_data_factory.tui.widgets import (
    CurationMetricsHUD,
    JaccardHistogramWidget,
    RejectionBreakdownWidget,
    RollingRate,
    ThroughputGauge,
)

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_FORCED: Final = 130

DEFAULT_BATCH_SIZE: Final = 256
DEFAULT_QUEUE_MAXSIZE: Final = 1_024
SHUTDOWN_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)
STOP_SIGNALS: Final = (signal.SIGUSR1,)

InputFormat = Literal["jsonl", "parquet"]
_SUFFIXES: Final[dict[str, InputFormat]] = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
}


def detect_format(path: Path) -> InputFormat:
    try:
        return _SUFFIXES[path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"{path}: unsupported input type; expected one of {', '.join(sorted(_SUFFIXES))}"
        ) from None


def open_reader(path: Path, *, batch_size: int) -> StreamingDatasetReader:
    if detect_format(path) == "jsonl":
        return StreamingDatasetReader.jsonl(path, batch_size=batch_size)
    return StreamingDatasetReader.parquet(path, batch_size=batch_size)


@dataclass(frozen=True, slots=True)
class RunConfig:
    inputs: tuple[Path, ...]
    export: ExportSpecification
    minhash: MinHashConfig
    root: Path = Path(".")
    batch_size: int = DEFAULT_BATCH_SIZE
    queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE
    auto_exit: bool = False

    @property
    def backlog_capacity(self) -> int:
        # full queue + one in-flight batch + the record a blocked producer holds
        return self.queue_maxsize + self.batch_size + 1


class CurationTUIApp(App[int]):
    """One curation job with live telemetry; waits for ``q`` afterwards unless ``auto_exit``."""

    TITLE = "synthetic-data-factory"
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: auto auto auto auto;
        grid-gutter: 0 1;
    }
    #hud, #throughput, #status { column-span: 2; }
    #status { height: auto; padding: 0 1; color: $text-muted; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("c", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        config: RunConfig,
        *,
        readers: Sequence[StreamingDatasetReader] | None = None,
        install_signal_handlers: bool = True,
        clock: Callable[[], float] = time.monotonic,
        tick_seconds: float = 0.25,
    ) -> None:
        super().__init__()
        self.config: Final = config
        self.controller: Final = SourceController()
        self.readers: Final = (
            list(readers)
            if readers is not None
            else [open_reader(path, batch_size=config.batch_size) for path in config.inputs]
        )
        self.index: Final = MinHashLSHIndex(config.minhash)
        self.sink: Final = AtomicParquetSink(config.export, root=config.root)
        self.outcome: PipelineState | None = None
        self.exit_code: int | None = None
        self.error: BaseException | None = None
        self.telemetry: CurationTelemetry | None = None
        self._install_signals: Final = install_signal_handlers
        self._installed_signals: list[signal.Signals] = []
        self._rate: Final = RollingRate(clock=clock)
        self._tick_seconds: Final = tick_seconds
        self._snapshot: PipelineSnapshot | None = None
        self._exit_when_done = False

    # ------------------------------------------------------------------ #
    # Layout and lifecycle                                                #
    # ------------------------------------------------------------------ #

    def compose(self) -> ComposeResult:
        yield Header()
        yield CurationMetricsHUD(id="hud")
        yield JaccardHistogramWidget(id="histogram")
        yield RejectionBreakdownWidget(id="rejections")
        yield ThroughputGauge(id="throughput")
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(JaccardHistogramWidget).threshold = self.config.minhash.threshold
        self.query_one(ThroughputGauge).capacity = self.config.backlog_capacity
        if self._install_signals:
            self._install_signal_handlers()
        names = ", ".join(str(path) for path in self.config.inputs) or "in-memory source"
        self._set_status(f"Curating {names} -> {self.sink.path}")
        self.run_worker(self._run_pipeline(), name="curation", exit_on_error=False)
        self.set_interval(self._tick_seconds, self.refresh_telemetry)

    def on_unmount(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in self._installed_signals:
            loop.remove_signal_handler(sig)
        self._installed_signals.clear()

    async def _run_pipeline(self) -> CurationMetrics:
        pipeline = CurationPipeline(
            self.index,
            sink=self.sink,
            metrics_hook=self._on_snapshot,
            queue_maxsize=self.config.queue_maxsize,
            batch_size=self.config.batch_size,
        )
        with self.sink:  # commit on success; abort on error or cancellation
            return await pipeline.run(self.controller.stream(self.readers))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state is WorkerState.SUCCESS:
            self.outcome, self.exit_code = PipelineState.COMPLETED, EXIT_OK
            report = self.sink.report
            rows = report.rows if report is not None else 0
            verb = "Stopped early" if self.controller.stop_requested else "Completed"
            self._set_status(f"{verb}: wrote {rows:,} rows to {self.sink.path}. Press q to quit.")
        elif event.state is WorkerState.ERROR:
            self.outcome, self.exit_code = PipelineState.FAILED, EXIT_FAILED
            self.error = event.worker.error
            self._set_status(f"Failed: {type(self.error).__name__}: {self.error}. Press q to quit.")
        elif event.state is WorkerState.CANCELLED:
            self.outcome, self.exit_code = PipelineState.FAILED, EXIT_FORCED
            self._set_status("Cancelled: partial output discarded.")
        else:
            return
        self.refresh_telemetry()
        if self._exit_when_done or self.config.auto_exit or event.state is WorkerState.CANCELLED:
            self.exit(self.exit_code, return_code=self.exit_code)

    # ------------------------------------------------------------------ #
    # Telemetry                                                           #
    # ------------------------------------------------------------------ #

    def _on_snapshot(self, snapshot: PipelineSnapshot) -> None:
        self._snapshot = snapshot
        self.refresh_telemetry()

    def refresh_telemetry(self) -> None:
        telemetry = compose_telemetry(
            self._snapshot,
            (reader.stats for reader in self.readers),
            self.index.stats,
            yielded=self.controller.yielded,
        )
        self.telemetry = telemetry
        hud = self.query_one(CurationMetricsHUD)
        hud.metrics = telemetry.metrics
        hud.state = self.outcome if self.outcome is not None else telemetry.state
        hud.paused = self.controller.paused
        hud.interrupted = self.controller.stop_requested
        self.query_one(RejectionBreakdownWidget).counts = telemetry.rejections
        self.query_one(JaccardHistogramWidget).buckets = telemetry.similarity
        gauge = self.query_one(ThroughputGauge)
        gauge.rate = self._rate.observe(telemetry.processed)
        gauge.backlog = telemetry.backlog

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    # ------------------------------------------------------------------ #
    # Controls                                                            #
    # ------------------------------------------------------------------ #

    def action_toggle_pause(self) -> None:
        if self.outcome is not None or self.controller.stop_requested:
            return
        if self.controller.paused:
            self.controller.resume()
            self._set_status("Resumed.")
        else:
            self.controller.pause()
            self._set_status("Paused at the next record; press p to resume.")
        self.refresh_telemetry()

    def action_cancel(self) -> None:
        """Graceful stop: finish queued records, commit the sink, keep the UI open."""
        if self.outcome is not None or self.controller.stop_requested:
            return
        self.controller.stop()
        self._set_status("Stopping: finishing queued records and committing output...")
        self.refresh_telemetry()

    async def action_quit(self) -> None:
        self.request_shutdown()

    def request_shutdown(self) -> None:
        """Graceful stop then exit; a repeated request while stopping forces cancellation."""
        if self.outcome is not None:
            self.exit(self.exit_code, return_code=self.exit_code or EXIT_OK)
        elif self._exit_when_done:
            self._set_status("Forcing shutdown: aborting output...")
            self.workers.cancel_node(self)  # CancelledError -> sink aborts, no partial file
        else:
            self._exit_when_done = True
            self.action_cancel()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in SHUTDOWN_SIGNALS:
            loop.add_signal_handler(sig, self.request_shutdown)
            self._installed_signals.append(sig)
        for sig in STOP_SIGNALS:
            loop.add_signal_handler(sig, self.action_cancel)
            self._installed_signals.append(sig)
