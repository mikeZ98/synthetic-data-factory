"""Headless widget tests: rendering invariants and reactive repaint through textual.pilot."""

import html
import io
import math
import re

import pytest
from hypothesis import given
from hypothesis import strategies as st
from rich.console import Console, RenderableType
from textual.app import App, ComposeResult
from textual.pilot import Pilot

from synthetic_data_factory.core.pipeline import PipelineSnapshot, PipelineState
from synthetic_data_factory.schemas import CurationMetrics
from synthetic_data_factory.transport.reader import RejectReason
from synthetic_data_factory.tui.widgets import (
    CurationMetricsHUD,
    JaccardHistogramWidget,
    RejectionBreakdownWidget,
    RollingRate,
    ThroughputGauge,
    state_badge,
)
from synthetic_data_factory.tui.widgets._bars import FULL_BLOCK, horizontal_bar, share
from synthetic_data_factory.tui.widgets.histogram import BUCKET_LABELS, histogram_renderable
from synthetic_data_factory.tui.widgets.hud import EMPTY_METRICS, hud_table
from synthetic_data_factory.tui.widgets.rejections import REASON_LABELS, rejection_table
from synthetic_data_factory.tui.widgets.throughput import (
    MAX_SAMPLES,
    saturation,
    saturation_style,
    throughput_table,
)

S = PipelineState


def plain(renderable: RenderableType, width: int = 110) -> str:
    buffer = io.StringIO()
    Console(width=width, file=buffer, color_system=None).print(renderable)
    return buffer.getvalue()


def metrics(raw: int, valid: int, kept: int) -> CurationMetrics:
    return CurationMetrics.from_counts(raw_count=raw, schema_valid_count=valid, deduped_count=kept)


class WidgetHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield CurationMetricsHUD()
        yield RejectionBreakdownWidget()
        yield JaccardHistogramWidget()
        yield ThroughputGauge()


def screen_text(pilot: Pilot[None]) -> str:
    """Text actually composited onto the headless screen (not just render() output)."""
    svg = pilot.app.export_screenshot()
    runs = re.findall(r"<text[^>]*>(.*?)</text>", svg, flags=re.DOTALL)  # glyph runs only, no CSS
    return html.unescape("".join(runs)).replace("\xa0", " ")


# --------------------------------------------------------------------------- #
# Bars                                                                        #
# --------------------------------------------------------------------------- #


@given(st.floats(allow_nan=True, allow_infinity=True), st.integers(1, 60))
def test_bar_always_has_exact_width(fraction: float, width: int) -> None:
    bar = horizontal_bar(fraction, width)
    assert len(bar) == width
    if math.isnan(fraction) or fraction <= 0:
        assert bar.strip() == ""
    else:
        assert bar.strip() != ""  # any positive value stays visible


def test_bar_resolution_and_clamping() -> None:
    assert horizontal_bar(1.0, 4) == FULL_BLOCK * 4
    assert horizontal_bar(7.5, 4) == FULL_BLOCK * 4
    assert horizontal_bar(0.5, 4) == FULL_BLOCK * 2 + "  "
    assert horizontal_bar(1 / 32, 4) == "▏   "
    assert horizontal_bar(1e-9, 4) == "▏   "


def test_bar_rejects_non_positive_width() -> None:
    with pytest.raises(ValueError, match="width"):
        horizontal_bar(0.5, 0)


def test_share_of_empty_total_is_zero() -> None:
    assert share(3, 0) == 0.0
    assert share(1, 4) == 0.25


# --------------------------------------------------------------------------- #
# HUD                                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("state", "paused", "interrupted", "label"),
    [
        (S.IDLE, False, False, "IDLE"),
        (S.INGESTING, False, False, "INGESTING"),
        (S.CURATING, True, False, "PAUSED"),
        (S.DEDUPLICATING, False, True, "DEDUPLICATING"),
        (S.COMPLETED, True, False, "COMPLETED"),  # pause is meaningless once terminal
        (S.COMPLETED, False, True, "STOPPED"),
        (S.FAILED, False, True, "FAILED"),
    ],
)
def test_state_badge(state: S, paused: bool, interrupted: bool, label: str) -> None:
    assert state_badge(state, paused=paused, interrupted=interrupted).plain == f" {label} "


def test_hud_formats_counts_and_retention() -> None:
    text = plain(hud_table(metrics(12_345, 10_000, 8_000), S.DEDUPLICATING))
    for expected in ("12,345", "10,000", "8,000", "64.8%", "DEDUPLICATING"):
        assert expected in text


@pytest.mark.asyncio
async def test_hud_reacts_to_snapshots() -> None:
    async with WidgetHarness().run_test(size=(120, 40)) as pilot:
        hud = pilot.app.query_one(CurationMetricsHUD)
        assert hud.metrics == EMPTY_METRICS
        assert "IDLE" in screen_text(pilot)
        hud.show_snapshot(
            PipelineSnapshot(state=S.CURATING, metrics=metrics(900, 850, 700), batches=4)
        )
        assert "CURATING" in plain(hud.render())  # render() reflects the change immediately
        await pilot.pause()
        screen = screen_text(pilot)
        assert "900" in screen and "850" in screen and "700" in screen and "77.8%" in screen
        hud.paused = True
        await pilot.pause()
        assert "PAUSED" in screen_text(pilot)
        hud.paused, hud.state, hud.interrupted = False, S.COMPLETED, True
        await pilot.pause()
        assert "STOPPED" in screen_text(pilot)


# --------------------------------------------------------------------------- #
# Rejection breakdown                                                         #
# --------------------------------------------------------------------------- #


def test_rejection_table_lists_every_reason_in_fixed_order() -> None:
    text = plain(rejection_table({RejectReason.SCHEMA: 6, RejectReason.INVALID_JSON: 2}))
    positions = [text.index(label) for label in REASON_LABELS.values()]
    assert positions == sorted(positions)
    assert set(REASON_LABELS) == set(RejectReason)
    assert "75%" in text and "25%" in text and "Total" in text and "8" in text


def test_rejection_table_without_rejections() -> None:
    assert "No rejected records" in plain(rejection_table({}))


@pytest.mark.asyncio
async def test_rejection_widget_snapshots_input_and_validates() -> None:
    async with WidgetHarness().run_test(size=(120, 40)) as pilot:
        widget = pilot.app.query_one(RejectionBreakdownWidget)
        assert "No rejected records" in screen_text(pilot)
        source = {RejectReason.OVERSIZED: 3}
        widget.counts = source
        source[RejectReason.OVERSIZED] = 999  # caller mutation must not leak in
        await pilot.pause()
        screen = screen_text(pilot)
        assert "Oversized line" in screen and "999" not in screen
        with pytest.raises(ValueError, match="non-negative"):
            widget.counts = {RejectReason.SCHEMA: -1}
        assert widget.counts == {RejectReason.OVERSIZED: 3}


# --------------------------------------------------------------------------- #
# Jaccard histogram                                                           #
# --------------------------------------------------------------------------- #


def test_histogram_labels_match_core_bucket_edges() -> None:
    assert BUCKET_LABELS == ("0.00-0.50", "0.50-0.70", "0.70-0.85", "0.85-1.00")


@pytest.mark.parametrize(("threshold", "marked_row"), [(0.9, 3), (0.85, 3), (0.6, 1), (0.1, 0)])
def test_histogram_marks_the_threshold_bucket(threshold: float, marked_row: int) -> None:
    lines = plain(histogram_renderable((5, 0, 2, 10), threshold=threshold)).splitlines()
    rows = [line for line in lines if any(label in line for label in BUCKET_LABELS)]
    assert [i for i, row in enumerate(rows) if "threshold" in row] == [marked_row]
    assert f"dedup threshold {threshold:.2f}" in lines[0]


def test_histogram_without_threshold_or_data() -> None:
    text = plain(histogram_renderable((0, 0, 0, 0)))
    assert "0 verified candidate pairs" in text and "threshold" not in text


@pytest.mark.asyncio
async def test_histogram_widget_reacts_and_validates() -> None:
    async with WidgetHarness().run_test(size=(120, 40)) as pilot:
        widget = pilot.app.query_one(JaccardHistogramWidget)
        widget.buckets = (1, 2, 3, 1_234)
        widget.threshold = 0.85
        await pilot.pause()
        screen = screen_text(pilot)
        assert "1,240 verified candidate pairs" in screen and "1,234" in screen
        for bad in ((1, 2, 3), (1, 2, 3, 4, 5), (1, -1, 0, 0)):
            with pytest.raises(ValueError, match="bucket counts"):
                widget.buckets = bad
        assert widget.buckets == (1, 2, 3, 1_234)


# --------------------------------------------------------------------------- #
# Throughput                                                                  #
# --------------------------------------------------------------------------- #


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_rolling_rate_over_window() -> None:
    clock = FakeClock()
    rate = RollingRate(window=5.0, clock=clock)
    assert rate.rate == 0.0
    assert rate.observe(0) == 0.0  # a single sample has no span
    for _ in range(10):
        clock.now += 1.0
        rate.observe(int((clock.now - 100.0) * 50))  # 50 samples/s
    assert rate.rate == pytest.approx(50.0)
    for _ in range(10):  # stall: rate decays to zero once the window passes
        clock.now += 1.0
        rate.observe(500)
    assert rate.rate == 0.0


def test_rolling_rate_handles_reset_and_zero_span() -> None:
    clock = FakeClock()
    rate = RollingRate(clock=clock)
    rate.observe(100)
    rate.observe(200)  # same timestamp: no span
    assert rate.rate == 0.0
    clock.now += 1.0
    rate.observe(10)  # counter reset
    clock.now += 1.0
    assert rate.observe(30) == pytest.approx(20.0)


def test_rolling_rate_memory_is_bounded() -> None:
    clock = FakeClock()
    rate = RollingRate(window=1e9, clock=clock)
    for n in range(MAX_SAMPLES * 3):
        clock.now += 0.001
        rate.observe(n)
    assert len(rate._samples) == MAX_SAMPLES
    assert rate.rate == pytest.approx(1_000.0)


@pytest.mark.parametrize("window", [0.0, -1.0, math.nan])
def test_rolling_rate_rejects_bad_windows(window: float) -> None:
    with pytest.raises(ValueError, match="window"):
        RollingRate(window=window)


@pytest.mark.parametrize(
    ("backlog", "capacity", "fraction", "style"),
    [
        (0, 10, 0.0, "green3"),
        (5, 10, 0.5, "yellow3"),
        (9, 10, 0.9, "red3"),
        (50, 10, 1.0, "red3"),
        (-3, 10, 0.0, "green3"),
    ],
)
def test_saturation_is_clamped_and_styled(
    backlog: int, capacity: int, fraction: float, style: str
) -> None:
    assert saturation(backlog, capacity) == fraction
    assert saturation_style(fraction) == style


def test_throughput_table_text() -> None:
    text = plain(throughput_table(1234.56, 30, 120))
    assert "1,234.6 samples/s" in text and "30/120 (25%)" in text


@pytest.mark.asyncio
async def test_throughput_gauge_reacts_and_sanitizes() -> None:
    async with WidgetHarness().run_test(size=(120, 40)) as pilot:
        gauge = pilot.app.query_one(ThroughputGauge)
        gauge.capacity, gauge.backlog, gauge.rate = 200, 150, 42.0
        await pilot.pause()
        assert "42.0 samples/s" in screen_text(pilot) and "150/200 (75%)" in screen_text(pilot)
        gauge.rate, gauge.backlog = math.nan, -5
        assert (gauge.rate, gauge.backlog) == (0.0, 0)
        gauge.rate = math.inf
        assert gauge.rate == 0.0
        with pytest.raises(ValueError, match="capacity"):
            gauge.capacity = 0
