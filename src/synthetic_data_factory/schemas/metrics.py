"""Funnel accounting for a curation run."""

import math
from typing import Annotated, Final, Self

from pydantic import Field, model_validator

from synthetic_data_factory.schemas._common import INT64_MAX, ContractModel

RETENTION_REL_TOL: Final = 1e-9
"""Relative tolerance when checking ``retention_rate`` against the counts."""

Count = Annotated[int, Field(ge=0, le=INT64_MAX)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


def _retention(deduped_count: int, raw_count: int) -> float:
    # An empty run retained nothing; define 0/0 as 0.0 rather than NaN.
    return deduped_count / raw_count if raw_count else 0.0


class CurationMetrics(ContractModel):
    """Counts at each curation stage: raw >= schema-valid >= deduplicated.

    ``retention_rate`` must equal ``deduped_count / raw_count`` (``0.0`` for an
    empty run) within :data:`RETENTION_REL_TOL`. Prefer :meth:`from_counts`.
    """

    raw_count: Count
    schema_valid_count: Count
    deduped_count: Count
    retention_rate: UnitInterval

    @model_validator(mode="after")
    def _check_funnel(self) -> Self:
        if self.schema_valid_count > self.raw_count:
            raise ValueError("schema_valid_count must not exceed raw_count")
        if self.deduped_count > self.schema_valid_count:
            raise ValueError("deduped_count must not exceed schema_valid_count")
        expected = _retention(self.deduped_count, self.raw_count)
        if not math.isclose(self.retention_rate, expected, rel_tol=RETENTION_REL_TOL):
            raise ValueError(
                f"retention_rate {self.retention_rate!r} is inconsistent with "
                f"deduped_count / raw_count = {expected!r}"
            )
        return self

    @classmethod
    def from_counts(cls, *, raw_count: int, schema_valid_count: int, deduped_count: int) -> Self:
        """Build metrics with ``retention_rate`` derived from the counts."""
        return cls(
            raw_count=raw_count,
            schema_valid_count=schema_valid_count,
            deduped_count=deduped_count,
            retention_rate=_retention(deduped_count, raw_count),
        )
