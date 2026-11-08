"""Reactive Textual telemetry widgets for the curation pipeline."""

from synthetic_data_factory.tui.widgets.histogram import JaccardHistogramWidget
from synthetic_data_factory.tui.widgets.hud import CurationMetricsHUD, state_badge
from synthetic_data_factory.tui.widgets.rejections import RejectionBreakdownWidget
from synthetic_data_factory.tui.widgets.throughput import RollingRate, ThroughputGauge

__all__ = [
    "CurationMetricsHUD",
    "JaccardHistogramWidget",
    "RejectionBreakdownWidget",
    "RollingRate",
    "ThroughputGauge",
    "state_badge",
]
