"""Contract primitives shared by every schema model.

All models are strict (no type coercion: ``"1"`` is not an int, ``True`` is not
a number), frozen, closed to unknown fields, and reject NaN/Inf. Text fields
reject blank content and NUL characters. Lone surrogates (not encodable as
UTF-8, so fatal at Arrow/Parquet export) are rejected by pydantic-core itself:
``StringConstraints`` forces conversion to a Rust ``String``, which fails with
``string_unicode``. The test suite pins that behaviour.
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated, Final

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

INT64_MAX: Final = 2**63 - 1
"""Upper bound for counters so they round-trip through Arrow/Parquet ``int64``."""

MAX_LONG_TEXT_CHARS: Final = 131_072
MAX_SHORT_TEXT_CHARS: Final = 4_096


class ContractModel(BaseModel):
    """Base class for all data contracts in this package."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        hide_input_in_errors=True,
    )


def _check_text(value: str) -> str:
    if not value.strip():
        raise ValueError("text must contain at least one non-whitespace character")
    if "\x00" in value:
        raise ValueError("text must not contain NUL characters")
    return value


def _require_utc(value: datetime) -> datetime:
    offset = value.utcoffset()
    if offset is None:
        raise ValueError("datetime must be timezone-aware with an explicit UTC timezone")
    if offset != timedelta(0) or value.tzname() != "UTC":
        raise ValueError("datetime must be expressed in UTC")
    return value.replace(tzinfo=UTC)


LongText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_LONG_TEXT_CHARS),
    AfterValidator(_check_text),
]
"""Non-blank, UTF-8-safe text up to :data:`MAX_LONG_TEXT_CHARS` code points."""

ShortText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_SHORT_TEXT_CHARS),
    AfterValidator(_check_text),
]
"""Non-blank, UTF-8-safe text up to :data:`MAX_SHORT_TEXT_CHARS` code points."""

UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
"""Timezone-aware datetime in UTC, normalized to :data:`datetime.UTC`."""
