"""Rolling-window throughput and ingestion backlog saturation."""

import math
import time
from collections import deque
from collections.abc import Callable
from typing import Final

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from synthetic_data_factory.tui.widgets._bars import horizontal_bar

DEFAULT_WINDOW_SECONDS: Final = 5.0
MAX_SAMPLES: Final = 4_096  # hard memory bound, whatever the observation rate


class RollingRate:
    """Events per second over a sliding time window, from cumulative totals.

    Keeps one sample at or before the window start so that the measured span
    covers the whole window. A total that goes backwards is treated as a
    counter reset.
    """

    __slots__ = ("_clock", "_samples", "window")

    def __init__(
        self, window: float = DEFAULT_WINDOW_SECONDS, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if not window > 0:
            raise ValueError(f"window must be positive, got {window}")
        self.window: Final = window
        self._clock: Final = clock
        self._samples: deque[tuple[float, int]] = deque(maxlen=MAX_SAMPLES)

    def observe(self, total: int) -> float:
        now = self._clock()
        if self._samples and total < self._samples[-1][1]:
            self._samples.clear()
        self._samples.append((now, total))
        cutoff = now - self.window
        while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        return self.rate

    @property
    def rate(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        (t0, c0), (t1, c1) = self._samples[0], self._samples[-1]
        return (c1 - c0) / (t1 - t0) if t1 > t0 else 0.0


def saturation(backlog: int, capacity: int) -> float:
    return min(1.0, max(0.0, backlog / capacity))


def saturation_style(fraction: float) -> str:
    if fraction >= 0.9:
        return "red3"
    if fraction >= 0.5:
        return "yellow3"
    return "green3"


def throughput_table(rate: float, backlog: int, capacity: int, *, bar_width: int = 30) -> Table:
    fraction = saturation(backlog, capacity)
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_row(
        Text.assemble((f"{rate:,.1f}", "bold"), " samples/s"),
        Text.assemble(
            ("backlog ", "dim"),
            (horizontal_bar(fraction, bar_width), saturation_style(fraction)),
        ),
        Text(f"{backlog:,}/{capacity:,} ({fraction:.0%})", style="dim"),
    )
    return table


class ThroughputGauge(Widget):
    """Samples per second plus backlog saturation (records read but not yet processed)."""

    DEFAULT_CSS = """
    ThroughputGauge {
        height: auto;
        border: round $success;
        border-title-color: $success;
        padding: 0 1;
    }
    """

    rate: reactive[float] = reactive(0.0)
    backlog: reactive[int] = reactive(0)
    capacity: reactive[int] = reactive(1)

    def on_mount(self) -> None:
        self.border_title = "Throughput"

    def validate_rate(self, value: float) -> float:
        return value if math.isfinite(value) and value > 0.0 else 0.0

    def validate_backlog(self, value: int) -> int:
        return max(0, value)

    def validate_capacity(self, value: int) -> int:
        if value < 1:
            raise ValueError(f"capacity must be positive, got {value}")
        return value

    def render(self) -> RenderableType:
        return throughput_table(self.rate, self.backlog, self.capacity)
