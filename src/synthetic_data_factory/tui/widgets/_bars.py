"""Unicode bar primitives shared by the telemetry widgets."""

import math
from typing import Final

_EIGHTHS: Final = " ▏▎▍▌▋▊▉"
FULL_BLOCK: Final = "█"


def share(part: int, total: int) -> float:
    """``part / total``, defined as 0.0 for an empty total."""
    return part / total if total else 0.0


def horizontal_bar(fraction: float, width: int) -> str:
    """A left-aligned bar ``width`` cells wide with 1/8-cell resolution.

    Fractions are clamped to [0, 1]; NaN counts as 0. Any strictly positive
    fraction draws at least one eighth, so a non-zero count is never invisible.
    """
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    if math.isnan(fraction) or fraction <= 0.0:
        return " " * width
    eighths = max(1, round(min(fraction, 1.0) * width * 8))
    full, remainder = divmod(eighths, 8)
    return (FULL_BLOCK * full + (_EIGHTHS[remainder] if remainder else "")).ljust(width)
