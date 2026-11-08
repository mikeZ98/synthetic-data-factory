"""One consistent view assembled from reader, pipeline and index statistics.

Ingestion validation happens in the readers, which classify every rejection
by :class:`RejectReason`. The pipeline deduplicates validated samples. The
composed funnel is therefore ``raw = records read``, ``valid = records
accepted`` and ``deduplicated = kept``. It always satisfies the
:class:`CurationMetrics` invariants, because kept <= processed <= yielded <=
accepted <= read, even when individual readings come from different moments.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from synthetic_data_factory.core.deduplicator import IndexStats
from synthetic_data_factory.core.pipeline import PipelineSnapshot, PipelineState
from synthetic_data_factory.schemas import CurationMetrics
from synthetic_data_factory.transport.reader import ReaderStats, RejectReason


@dataclass(frozen=True, slots=True)
class CurationTelemetry:
    state: PipelineState
    metrics: CurationMetrics
    processed: int  # samples consumed by the pipeline
    backlog: int  # handed to the pipeline but not yet processed
    rejections: Mapping[RejectReason, int]
    similarity: tuple[int, ...]


def compose_telemetry(
    snapshot: PipelineSnapshot | None,
    readers: Iterable[ReaderStats],
    index: IndexStats,
    *,
    yielded: int,
) -> CurationTelemetry:
    read = accepted = 0
    rejections: Counter[RejectReason] = Counter()
    for stats in readers:
        read += stats.records
        accepted += stats.accepted
        rejections.update(stats.rejected)
    processed = snapshot.metrics.raw_count if snapshot is not None else 0
    kept = snapshot.metrics.deduped_count if snapshot is not None else 0
    return CurationTelemetry(
        state=snapshot.state if snapshot is not None else PipelineState.IDLE,
        metrics=CurationMetrics.from_counts(
            raw_count=read, schema_valid_count=accepted, deduped_count=kept
        ),
        processed=processed,
        backlog=yielded - processed,
        rejections=MappingProxyType({r: n for r, n in rejections.items() if n}),
        similarity=index.similarity_histogram,
    )
