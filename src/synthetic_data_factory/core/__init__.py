"""Deterministic curation engine: shingling, MinHash LSH deduplication, async pipeline."""

from synthetic_data_factory.core.deduplicator import (
    IndexStats,
    MinHashLSHIndex,
    canonical_order,
    default_text_of,
    jaccard,
    optimal_bands,
)
from synthetic_data_factory.core.pipeline import (
    TERMINAL_STATES,
    TRANSITIONS,
    CurationPipeline,
    InvalidTransitionError,
    PipelineSnapshot,
    PipelineState,
    PipelineStateMachine,
)
from synthetic_data_factory.core.shingler import ShingleError, TextShingler, normalize_text

__all__ = [
    "TERMINAL_STATES",
    "TRANSITIONS",
    "CurationPipeline",
    "IndexStats",
    "InvalidTransitionError",
    "MinHashLSHIndex",
    "PipelineSnapshot",
    "PipelineState",
    "PipelineStateMachine",
    "ShingleError",
    "TextShingler",
    "canonical_order",
    "default_text_of",
    "jaccard",
    "normalize_text",
    "optimal_bands",
]
