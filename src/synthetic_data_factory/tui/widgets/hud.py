"""Headline curation metrics and the pipeline state badge."""

from typing import Final

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from synthetic_data_factory.core.pipeline import TERMINAL_STATES, PipelineSnapshot, PipelineState
from synthetic_data_factory.schemas import CurationMetrics

EMPTY_METRICS: Final = CurationMetrics.from_counts(
    raw_count=0, schema_valid_count=0, deduped_count=0
)

_BADGE_STYLES: Final = {
    PipelineState.IDLE: "bold white on grey35",
    PipelineState.INGESTING: "bold white on dodger_blue3",
    PipelineState.CURATING: "bold black on cyan3",
    PipelineState.DEDUPLICATING: "bold white on medium_purple3",
    PipelineState.COMPLETED: "bold black on green3",
    PipelineState.FAILED: "bold white on red3",
}
_PAUSED_STYLE: Final = "bold black on yellow3"
_STOPPED_STYLE: Final = "bold black on dark_orange"


def state_badge(state: PipelineState, *, paused: bool = False, interrupted: bool = False) -> Text:
    """``PAUSED`` overrides live states, ``STOPPED`` marks an interrupted completion."""
    if paused and state not in TERMINAL_STATES:
        label, style = "PAUSED", _PAUSED_STYLE
    elif interrupted and state is PipelineState.COMPLETED:
        label, style = "STOPPED", _STOPPED_STYLE
    else:
        label, style = state.value.upper(), _BADGE_STYLES[state]
    return Text(f" {label} ", style=style)


def hud_table(
    metrics: CurationMetrics,
    state: PipelineState,
    *,
    paused: bool = False,
    interrupted: bool = False,
) -> Table:
    table = Table.grid(expand=True, padding=(0, 2))
    labels = ("Raw", "Valid", "Deduplicated", "Retention", "State")
    for _ in labels:
        table.add_column(justify="center", ratio=1)
    table.add_row(*(Text(label, style="dim") for label in labels))
    table.add_row(
        Text(f"{metrics.raw_count:,}", style="bold"),
        Text(f"{metrics.schema_valid_count:,}", style="bold"),
        Text(f"{metrics.deduped_count:,}", style="bold"),
        Text(f"{metrics.retention_rate:.1%}", style="bold"),
        state_badge(state, paused=paused, interrupted=interrupted),
    )
    return table


class CurationMetricsHUD(Widget):
    """Live raw / valid / deduplicated counts, retention and state badge."""

    DEFAULT_CSS = """
    CurationMetricsHUD {
        height: auto;
        border: round $primary;
        border-title-color: $primary;
        padding: 0 1;
    }
    """

    metrics: reactive[CurationMetrics] = reactive(EMPTY_METRICS)
    state: reactive[PipelineState] = reactive(PipelineState.IDLE)
    paused: reactive[bool] = reactive(False)
    interrupted: reactive[bool] = reactive(False)

    def on_mount(self) -> None:
        self.border_title = "Curation"

    def show_snapshot(self, snapshot: PipelineSnapshot) -> None:
        self.metrics = snapshot.metrics
        self.state = snapshot.state

    def render(self) -> RenderableType:
        return hud_table(self.metrics, self.state, paused=self.paused, interrupted=self.interrupted)
