"""Near-duplicate detection parameters (MinHash + LSH)."""

from typing import Annotated, Final

from pydantic import AfterValidator, Field

from synthetic_data_factory.schemas._common import ContractModel

MIN_NUM_PERM: Final = 16
MAX_NUM_PERM: Final = 4096
MIN_SHINGLE_K: Final = 2
MAX_SHINGLE_K: Final = 64


def _require_power_of_two(value: int) -> int:
    if value & (value - 1):
        raise ValueError("num_perm must be a power of two")
    return value


class MinHashConfig(ContractModel):
    """Parameters for MinHash signatures and the LSH similarity threshold."""

    threshold: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
    num_perm: Annotated[
        int,
        Field(ge=MIN_NUM_PERM, le=MAX_NUM_PERM),
        AfterValidator(_require_power_of_two),
    ]
    shingle_k: Annotated[int, Field(ge=MIN_SHINGLE_K, le=MAX_SHINGLE_K)]
