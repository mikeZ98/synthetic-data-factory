"""The unit of curation: one instruction/response pair."""

import re
from typing import Annotated, Final

from pydantic import AfterValidator, Field, StringConstraints

from synthetic_data_factory.schemas._common import (
    ContractModel,
    LongText,
    ShortText,
    UtcDatetime,
)

MAX_DOMAIN_CHARS: Final = 64
MAX_METADATA_ENTRIES: Final = 64
MAX_METADATA_KEY_CHARS: Final = 128

_UUID4_RE: Final = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


def _require_uuid4(value: str) -> str:
    if _UUID4_RE.fullmatch(value) is None:
        raise ValueError("sample_id must be a canonical lowercase hyphenated UUIDv4")
    return value


SampleId = Annotated[str, AfterValidator(_require_uuid4)]

DomainLabel = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=MAX_DOMAIN_CHARS,
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$",
    ),
]
"""Lowercase slug such as ``code``, ``math`` or ``medical.cardiology``."""

MetadataKey = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=MAX_METADATA_KEY_CHARS,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]


class InstructionSample(ContractModel):
    """A single synthetic instruction-tuning record."""

    sample_id: SampleId
    system_prompt: LongText
    instruction: LongText
    response: LongText
    domain: DomainLabel
    metadata: Annotated[
        dict[MetadataKey, ShortText],
        Field(default_factory=dict, max_length=MAX_METADATA_ENTRIES),
    ]
    created_at_utc: UtcDatetime
