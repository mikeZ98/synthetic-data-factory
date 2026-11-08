"""Rejected-record breakdown by :class:`RejectReason`."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from synthetic_data_factory.transport.reader import RejectReason
from synthetic_data_factory.tui.widgets._bars import horizontal_bar, share

REASON_LABELS: Final[Mapping[RejectReason, str]] = MappingProxyType(
    {
        RejectReason.INVALID_JSON: "Malformed JSON",
        RejectReason.OVERSIZED: "Oversized line",
        RejectReason.DUPLICATE_METADATA_KEY: "Duplicate metadata key",
        RejectReason.NULL_FIELD: "Null required field",
        RejectReason.SCHEMA: "Schema violation",
        RejectReason.UNREPRESENTABLE: "Unrepresentable value",
    }
)
_NO_REJECTIONS: Final[Mapping[RejectReason, int]] = MappingProxyType({})


def rejection_table(counts: Mapping[RejectReason, int], *, bar_width: int = 20) -> RenderableType:
    total = sum(counts.values())
    if total == 0:
        return Text("No rejected records", style="green")
    top = max(counts.values())
    table = Table.grid(padding=(0, 1))
    table.add_column()
    table.add_column(no_wrap=True)
    table.add_column(justify="right")
    table.add_column(justify="right", style="dim")
    for reason, label in REASON_LABELS.items():
        count = counts.get(reason, 0)
        table.add_row(
            label,
            Text(horizontal_bar(share(count, top), bar_width), style="red3"),
            f"{count:,}",
            f"{share(count, total):.0%}",
        )
    table.add_row(Text("Total", style="bold"), "", Text(f"{total:,}", style="bold"), "")
    return table


class RejectionBreakdownWidget(Widget):
    """Per-reason rejection counts with proportional bars, in fixed reason order."""

    DEFAULT_CSS = """
    RejectionBreakdownWidget {
        height: auto;
        border: round $error;
        border-title-color: $error;
        padding: 0 1;
    }
    """

    counts: reactive[Mapping[RejectReason, int]] = reactive(_NO_REJECTIONS)

    def on_mount(self) -> None:
        self.border_title = "Rejections"

    def validate_counts(self, value: Mapping[RejectReason, int]) -> Mapping[RejectReason, int]:
        # Snapshot-copy: later mutation of the caller's dict must not desync the view.
        if any(count < 0 for count in value.values()):
            raise ValueError("rejection counts must be non-negative")
        return MappingProxyType(dict(value))

    def render(self) -> RenderableType:
        return rejection_table(self.counts)
