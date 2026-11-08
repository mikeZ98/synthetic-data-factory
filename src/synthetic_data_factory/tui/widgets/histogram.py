"""Distribution of exact Jaccard similarity over verified LSH candidates."""

from typing import Final

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from synthetic_data_factory.core.deduplicator import SIMILARITY_BUCKET_EDGES, similarity_bucket
from synthetic_data_factory.tui.widgets._bars import horizontal_bar, share

BUCKET_COUNT: Final = len(SIMILARITY_BUCKET_EDGES) + 1
_BOUNDS: Final = (0.0, *SIMILARITY_BUCKET_EDGES, 1.0)
BUCKET_LABELS: Final = tuple(
    f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(_BOUNDS, _BOUNDS[1:], strict=False)
)
_BUCKET_STYLES: Final = ("grey50", "steel_blue1", "gold1", "orange_red1")
EMPTY_BUCKETS: Final = (0,) * BUCKET_COUNT


def histogram_renderable(
    buckets: tuple[int, ...], *, threshold: float | None = None, bar_width: int = 24
) -> RenderableType:
    total = sum(buckets)
    marked = similarity_bucket(threshold) if threshold is not None else None
    title = f"{total:,} verified candidate pairs"
    if threshold is not None:
        title += f"  ·  dedup threshold {threshold:.2f}"
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right")
    table.add_column(justify="right", style="dim")
    table.add_column(style="bold yellow")
    peak = max(buckets)
    for index, (label, count) in enumerate(zip(BUCKET_LABELS, buckets, strict=True)):
        table.add_row(
            label,
            Text(horizontal_bar(share(count, peak), bar_width), style=_BUCKET_STYLES[index]),
            f"{count:,}",
            f"{share(count, total):.0%}",
            "◂ threshold" if index == marked else "",
        )
    return Group(Text(title, style="dim italic"), table)


class JaccardHistogramWidget(Widget):
    """Four-bucket similarity histogram: [0,.5) [.5,.7) [.7,.85) [.85,1]."""

    DEFAULT_CSS = """
    JaccardHistogramWidget {
        height: auto;
        border: round $accent;
        border-title-color: $accent;
        padding: 0 1;
    }
    """

    buckets: reactive[tuple[int, ...]] = reactive(EMPTY_BUCKETS)
    threshold: reactive[float | None] = reactive(None)

    def on_mount(self) -> None:
        self.border_title = "Jaccard similarity"

    def validate_buckets(self, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != BUCKET_COUNT or any(count < 0 for count in value):
            raise ValueError(f"expected {BUCKET_COUNT} non-negative bucket counts, got {value!r}")
        return tuple(value)

    def render(self) -> RenderableType:
        return histogram_renderable(self.buckets, threshold=self.threshold)
