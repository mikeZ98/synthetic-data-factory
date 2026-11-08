"""Parquet export target."""

import re
from pathlib import PurePosixPath
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, Field, StringConstraints

from synthetic_data_factory.schemas._common import ContractModel

MIN_ROW_GROUP_SIZE: Final = 1_000
MAX_ROW_GROUP_SIZE: Final = 16_777_216
"""Caps per-row-group writer memory."""

MAX_PATH_CHARS: Final = 1_024

Compression = Literal["zstd", "snappy", "gzip"]

_FORBIDDEN_PATH_CHARS_RE: Final = re.compile(r"[\x00-\x1f\x7f\\]")
_WINDOWS_DRIVE_RE: Final = re.compile(r"[A-Za-z]:")


def _validate_output_path(value: str) -> str:
    if _FORBIDDEN_PATH_CHARS_RE.search(value) or value != value.strip():
        raise ValueError(
            "output_path must be a POSIX path without control characters, "
            "backslashes, or surrounding whitespace"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("~") or _WINDOWS_DRIVE_RE.match(value):
        raise ValueError("output_path must be a relative path")
    if ".." in path.parts:
        raise ValueError("output_path must not traverse upwards with '..'")
    if value.endswith("/") or path.suffix != ".parquet":
        raise ValueError("output_path must name a file ending in '.parquet'")
    return str(path)


RelativeParquetPath = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_PATH_CHARS),
    AfterValidator(_validate_output_path),
]
"""Relative POSIX path to a ``.parquet`` file, normalized (``./a//b`` -> ``a/b``)."""


class ExportSpecification(ContractModel):
    """Where and how a curated dataset is written."""

    output_path: RelativeParquetPath
    compression: Compression
    row_group_size: Annotated[int, Field(ge=MIN_ROW_GROUP_SIZE, le=MAX_ROW_GROUP_SIZE)]
