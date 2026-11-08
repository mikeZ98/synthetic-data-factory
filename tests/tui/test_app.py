"""CurationTUIApp: full sessions, key bindings, signal orchestration, storage hygiene."""

import asyncio
import json
import os
import signal
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from textual.pilot import Pilot

from synthetic_data_factory.core.deduplicator import IndexStats
from synthetic_data_factory.core.pipeline import PipelineSnapshot, PipelineState
from synthetic_data_factory.schemas import (
    CurationMetrics,
    ExportSpecification,
    InstructionSample,
    MinHashConfig,
)
from synthetic_data_factory.transport.reader import (
    ReaderStats,
    RejectReason,
    StreamingDatasetReader,
)
from synthetic_data_factory.transport.schema import samples_to_table, table_to_samples
from synthetic_data_factory.tui.app import (
    EXIT_FAILED,
    EXIT_FORCED,
    EXIT_OK,
    CurationTUIApp,
    RunConfig,
    detect_format,
    open_reader,
)
from synthetic_data_factory.tui.control import SourceController
from synthetic_data_factory.tui.telemetry import compose_telemetry
from synthetic_data_factory.tui.widgets import (
    CurationMetricsHUD,
    JaccardHistogramWidget,
    RejectionBreakdownWidget,
    ThroughputGauge,
)

S = PipelineState
T0 = datetime(2026, 1, 1, tzinfo=UTC)
TIMEOUT = 15.0
UNIQUE = 400
OUTPUT = "out/curated.parquet"


def sample(n: int) -> InstructionSample:
    topic = n % UNIQUE  # n and n + UNIQUE are exact duplicates (different ids)
    return InstructionSample(
        sample_id=str(uuid.UUID(int=n + 1, version=4)),
        system_prompt="sys",
        instruction=f"explain topic {topic} in simple words for a curious student",
        response=f"a careful answer about topic {topic} with several details",
        domain="edu",
        created_at_utc=T0 + timedelta(seconds=n),
    )


def write_jsonl(path: Path, count: int, *, junk: bool = True) -> Path:
    lines = [sample(n).model_dump_json() for n in range(count)]
    if junk:
        lines += ["{broken", json.dumps({"sample_id": "nope"})]
    path.write_text("\n".join(lines) + "\n")
    return path


def config(root: Path, *inputs: Path, **overrides: object) -> RunConfig:
    fields: dict[str, object] = {
        "inputs": inputs,
        "export": ExportSpecification(output_path=OUTPUT, compression="zstd", row_group_size=1_000),
        "minhash": MinHashConfig(threshold=0.9, num_perm=64, shingle_k=3),
        "root": root,
        "batch_size": 64,
        "queue_maxsize": 64,
    }
    fields.update(overrides)
    return RunConfig(**fields)  # type: ignore[arg-type]


async def until(pilot: Pilot[int], predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(TIMEOUT):
        while not predicate():
            await pilot.pause(0.01)


def output_rows(root: Path) -> list[InstructionSample]:
    return table_to_samples(pq.read_table(root / OUTPUT))


def stray_temp_files(root: Path) -> list[Path]:
    return [p for p in root.rglob(".tmp.*")]


def paused(controller: SourceController) -> bool:
    """Fresh read: mypy would otherwise keep a narrowed property value across calls."""
    return controller.paused


def stopping(controller: SourceController) -> bool:
    return controller.stop_requested


def assert_no_orphaned_tasks() -> None:
    assert asyncio.all_tasks() == {asyncio.current_task()}


def hanging_reader(gate: asyncio.Event, count: int = 10) -> StreamingDatasetReader:
    """Yields ``count`` samples, then blocks forever (upstream stall)."""

    async def source() -> AsyncIterator[InstructionSample]:
        for n in range(count):
            yield sample(n)
        gate.set()
        await asyncio.Event().wait()

    return StreamingDatasetReader.memory(source(), batch_size=1)


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "kind"),
    [("a.jsonl", "jsonl"), ("a.NDJSON", "jsonl"), ("a.parquet", "parquet")],
)
def test_detect_format(name: str, kind: str) -> None:
    assert detect_format(Path(name)) == kind


def test_unknown_input_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported input type"):
        open_reader(Path("data.csv"), batch_size=10)


def test_backlog_capacity(tmp_path: Path) -> None:
    assert config(tmp_path).backlog_capacity == 64 + 64 + 1


def test_telemetry_before_first_snapshot() -> None:
    empty_index = IndexStats(0, 1, 1, 0, 0, (0, 0, 0, 0))
    telemetry = compose_telemetry(None, [], empty_index, yielded=0)
    assert telemetry.state is S.IDLE
    assert (telemetry.processed, telemetry.backlog, telemetry.metrics.raw_count) == (0, 0, 0)


def test_telemetry_merges_readers_and_drops_zero_counts() -> None:
    first = ReaderStats(records=10, accepted=8)
    first.rejected.update({RejectReason.SCHEMA: 2, RejectReason.OVERSIZED: 0})
    second = ReaderStats(records=5, accepted=4)
    second.rejected.update({RejectReason.SCHEMA: 1})
    snapshot = PipelineSnapshot(
        state=S.DEDUPLICATING,
        metrics=CurationMetrics.from_counts(raw_count=9, schema_valid_count=9, deduped_count=7),
        batches=2,
    )
    index = IndexStats(7, 8, 8, 30, 0, (20, 5, 3, 2))
    telemetry = compose_telemetry(snapshot, [first, second], index, yielded=11)
    assert (telemetry.metrics.raw_count, telemetry.metrics.schema_valid_count) == (15, 12)
    assert telemetry.metrics.deduped_count == 7
    assert (telemetry.processed, telemetry.backlog) == (9, 2)
    assert telemetry.rejections == {RejectReason.SCHEMA: 3}
    assert telemetry.similarity == (20, 5, 3, 2)


@pytest.mark.asyncio
async def test_controller_semantics() -> None:
    controller = SourceController()
    assert not paused(controller) and not controller.stop_requested
    controller.pause()
    assert paused(controller)
    controller.resume()
    assert not paused(controller)
    controller.pause()
    controller.stop()  # stop wakes a paused stream
    assert not paused(controller) and stopping(controller)
    controller.pause()  # pausing after stop would block the final drain: ignored
    assert not paused(controller)
    reader = StreamingDatasetReader.memory([sample(0), sample(1)])
    assert [s async for s in controller.stream([reader])] == []  # stopped before first record


@pytest.mark.asyncio
async def test_controller_chains_readers_in_order() -> None:
    controller = SourceController()
    readers = [StreamingDatasetReader.memory([sample(n)]) for n in range(3)]
    assert [s async for s in controller.stream(readers)] == [sample(n) for n in range(3)]
    assert controller.yielded == 3


# --------------------------------------------------------------------------- #
# Full sessions                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_full_session_final_screen_matches_completed_metrics(tmp_path: Path) -> None:
    jsonl = write_jsonl(tmp_path / "a.jsonl", 600)  # 400 unique + 200 duplicates + 2 junk
    parquet = tmp_path / "b.parquet"
    pq.write_table(samples_to_table([sample(n) for n in range(600, 700)]), parquet)
    app = CurationTUIApp(config(tmp_path, jsonl, parquet), install_signal_handlers=False)
    async with app.run_test(size=(140, 45)) as pilot:
        await until(pilot, lambda: app.outcome is not None)
        await pilot.pause()
        hud = app.query_one(CurationMetricsHUD)
        assert app.outcome is S.COMPLETED and hud.state is S.COMPLETED
        assert (hud.metrics.raw_count, hud.metrics.schema_valid_count) == (702, 700)
        assert hud.metrics.deduped_count == 400  # 600..699 duplicate topics 200..299
        rejections = app.query_one(RejectionBreakdownWidget).counts
        assert rejections == {RejectReason.INVALID_JSON: 1, RejectReason.SCHEMA: 1}
        assert (
            sum(app.query_one(JaccardHistogramWidget).buckets) == app.index.stats.candidate_checks
        )
        assert app.query_one(JaccardHistogramWidget).threshold == 0.9
        assert app.query_one(ThroughputGauge).capacity == 129
        screen = app.export_screenshot()
        assert "COMPLETED" in screen and "Completed: wrote 400 rows" in screen.replace(
            "&#160;", " "
        )
        await pilot.press("q")
    assert app.return_code == EXIT_OK
    assert len(output_rows(tmp_path)) == 400
    assert stray_temp_files(tmp_path) == []
    assert_no_orphaned_tasks()


@pytest.mark.asyncio
async def test_pause_and_resume_via_p(tmp_path: Path) -> None:
    app = CurationTUIApp(
        config(tmp_path, write_jsonl(tmp_path / "a.jsonl", 6_000, junk=False)),
        install_signal_handlers=False,
    )
    async with app.run_test() as pilot:
        await pilot.press("p")
        hud = app.query_one(CurationMetricsHUD)
        assert paused(app.controller) and hud.paused
        frozen = app.controller.yielded
        for _ in range(20):
            await pilot.pause(0.01)
        assert app.controller.yielded == frozen < 6_000  # the source is gated
        assert "PAUSED" in app.export_screenshot()
        await pilot.press("p")
        assert not paused(app.controller)
        await until(pilot, lambda: app.outcome is not None)
        await pilot.press("p")  # no-op once finished
        assert app.outcome is S.COMPLETED and not paused(app.controller)
        await pilot.press("q")
    assert len(output_rows(tmp_path)) == UNIQUE


@pytest.mark.asyncio
async def test_cancel_via_c_commits_partial_output_and_keeps_ui(tmp_path: Path) -> None:
    app = CurationTUIApp(
        config(tmp_path, write_jsonl(tmp_path / "a.jsonl", 20_000, junk=False)),
        install_signal_handlers=False,
    )
    async with app.run_test() as pilot:
        await until(pilot, lambda: app.controller.yielded > 100)
        await pilot.press("c")
        await pilot.press("c")  # idempotent
        await pilot.press("p")  # pause is ignored while stopping
        assert not app.controller.paused
        await until(pilot, lambda: app.outcome is not None)
        assert app.is_running  # cancel does not quit
        hud = app.query_one(CurationMetricsHUD)
        assert hud.interrupted and "STOPPED" in app.export_screenshot()
        kept = hud.metrics.deduped_count
        await pilot.press("c")  # no-op once finished
        await pilot.press("q")
    assert app.return_code == EXIT_OK
    rows = output_rows(tmp_path)
    assert 0 < len(rows) == kept and app.controller.yielded < 20_000
    assert stray_temp_files(tmp_path) == []


@pytest.mark.asyncio
async def test_quit_while_running_drains_commits_and_exits_zero(tmp_path: Path) -> None:
    app = CurationTUIApp(
        config(tmp_path, write_jsonl(tmp_path / "a.jsonl", 20_000, junk=False)),
        install_signal_handlers=False,
    )
    async with app.run_test() as pilot:
        await until(pilot, lambda: app.controller.yielded > 100)
        await pilot.press("q")
        await until(pilot, lambda: not app.is_running)
    assert app.return_code == EXIT_OK and app.outcome is S.COMPLETED
    assert len(output_rows(tmp_path)) > 0
    assert stray_temp_files(tmp_path) == []


@pytest.mark.asyncio
async def test_second_quit_forces_cancellation_and_discards_output(tmp_path: Path) -> None:
    gate = asyncio.Event()
    app = CurationTUIApp(
        config(tmp_path), readers=[hanging_reader(gate)], install_signal_handlers=False
    )
    async with app.run_test() as pilot:
        await asyncio.wait_for(gate.wait(), TIMEOUT)
        await pilot.press("q")  # graceful stop cannot complete: upstream is stalled
        await pilot.pause(0.05)
        assert app.is_running and app.outcome is None
        await pilot.press("q")  # escalate
        await until(pilot, lambda: not app.is_running)
    assert app.return_code == EXIT_FORCED
    assert app.sink.state.value == "aborted"
    assert_no_orphaned_tasks()
    assert not (tmp_path / OUTPUT).exists()
    assert stray_temp_files(tmp_path) == []


@pytest.mark.asyncio
async def test_failure_is_shown_and_exit_code_is_one(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    pq.write_table(samples_to_table([sample(0)]).drop_columns(["domain"]), bad)
    app = CurationTUIApp(config(tmp_path, bad), install_signal_handlers=False)
    async with app.run_test(size=(140, 40)) as pilot:
        await until(pilot, lambda: app.outcome is not None)
        await pilot.pause()
        assert app.outcome is S.FAILED
        assert app.query_one(CurationMetricsHUD).state is S.FAILED
        assert "SchemaMismatchError" in app.export_screenshot()
        await pilot.press("q")
    assert app.return_code == EXIT_FAILED
    assert not (tmp_path / OUTPUT).exists() and stray_temp_files(tmp_path) == []


@pytest.mark.asyncio
async def test_auto_exit_returns_without_user_input(tmp_path: Path) -> None:
    app = CurationTUIApp(
        config(tmp_path, write_jsonl(tmp_path / "a.jsonl", 50), auto_exit=True),
        install_signal_handlers=False,
    )
    async with app.run_test() as pilot:
        await until(pilot, lambda: not app.is_running)
    assert app.return_code == EXIT_OK
    assert len(output_rows(tmp_path)) == 50


# --------------------------------------------------------------------------- #
# Signals                                                                     #
# --------------------------------------------------------------------------- #


def _signal_is_trapped(sig: signal.Signals) -> bool:
    return signal.getsignal(sig) not in (signal.SIG_DFL, signal.default_int_handler)


@pytest.mark.asyncio
async def test_sigterm_drains_flushes_and_exits_zero(tmp_path: Path) -> None:
    app = CurationTUIApp(config(tmp_path, write_jsonl(tmp_path / "a.jsonl", 20_000, junk=False)))
    async with app.run_test() as pilot:
        assert all(_signal_is_trapped(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1))
        await until(pilot, lambda: app.controller.yielded > 100)
        os.kill(os.getpid(), signal.SIGTERM)
        await until(pilot, lambda: not app.is_running)
    assert app.return_code == EXIT_OK and app.controller.stop_requested
    assert len(output_rows(tmp_path)) > 0
    assert stray_temp_files(tmp_path) == []
    assert not any(_signal_is_trapped(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1))


@pytest.mark.asyncio
async def test_sigusr1_stops_gracefully_but_keeps_ui_open(tmp_path: Path) -> None:
    app = CurationTUIApp(config(tmp_path, write_jsonl(tmp_path / "a.jsonl", 20_000, junk=False)))
    async with app.run_test() as pilot:
        await until(pilot, lambda: app.controller.yielded > 100)
        os.kill(os.getpid(), signal.SIGUSR1)
        await until(pilot, lambda: app.outcome is not None)
        assert app.is_running and app.outcome is S.COMPLETED
        await pilot.press("q")
    assert app.return_code == EXIT_OK
    assert len(output_rows(tmp_path)) > 0


@pytest.mark.asyncio
async def test_repeated_sigint_forces_exit_130(tmp_path: Path) -> None:
    gate = asyncio.Event()
    app = CurationTUIApp(config(tmp_path), readers=[hanging_reader(gate)])
    async with app.run_test() as pilot:
        await asyncio.wait_for(gate.wait(), TIMEOUT)
        assert _signal_is_trapped(signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)
        await pilot.pause(0.05)
        assert app.is_running
        os.kill(os.getpid(), signal.SIGINT)
        await until(pilot, lambda: not app.is_running)
    assert app.return_code == EXIT_FORCED
    assert not (tmp_path / OUTPUT).exists() and stray_temp_files(tmp_path) == []
    assert not _signal_is_trapped(signal.SIGINT)
    assert_no_orphaned_tasks()


@pytest.mark.asyncio
async def test_signal_handlers_can_be_disabled(tmp_path: Path) -> None:
    app = CurationTUIApp(
        config(tmp_path, write_jsonl(tmp_path / "a.jsonl", 5)), install_signal_handlers=False
    )
    async with app.run_test() as pilot:
        assert not _signal_is_trapped(signal.SIGTERM)
        await until(pilot, lambda: app.outcome is not None)
        await pilot.press("q")
    assert app.return_code == EXIT_OK
