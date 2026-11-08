"""Reproducible benchmarks behind the README tables.

    uv run --locked python -m tools.benchmark            # full run (several minutes)
    uv run --locked python -m tools.benchmark --quick    # smoke-sized run

Results go to ``.cache/benchmark/results.json`` and are printed as Markdown.

* complexity: exhaustive greedy pairwise deduplication (exact Jaccard against
  every kept document) against the MinHash LSH index on the same seeded corpus.
  It reports wall-clock time and exact-Jaccard comparison counts, plus how
  many of the exhaustive method's drops LSH also made (LSH can miss pairs;
  it can never invent one, because every drop is exactly verified).
* memory: peak RSS per dataset size, each measured in a fresh subprocess, for
  streaming only (reader -> sink, no index) and for full curation
  (reader -> pipeline -> index -> sink). Streaming buffers are bounded; the
  index keeps every *kept* document's shingle set for exact verification, so
  curation memory grows with the number of kept documents by design.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import resource
import subprocess  # noqa: S404 - memory is measured in isolated child processes
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

from synthetic_data_factory.core.deduplicator import MinHashLSHIndex, jaccard
from synthetic_data_factory.core.pipeline import CurationPipeline
from synthetic_data_factory.schemas import ExportSpecification, MinHashConfig
from synthetic_data_factory.transport.parquet_sink import AtomicParquetSink
from synthetic_data_factory.transport.reader import StreamingDatasetReader

# threshold 0.7 / 3-word shingles: one replaced token gives J ~ 0.85, two give
# ~0.72 (near the threshold, where LSH can miss), three ~0.61 (not duplicates).
CONFIG: Final = MinHashConfig(threshold=0.7, num_perm=128, shingle_k=3)
VOCABULARY: Final = 5_000
WORDS_PER_DOC: Final = 40
DUPLICATE_RATE: Final = 0.3
RSS_UNIT: Final = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
T0: Final = datetime(2026, 1, 1, tzinfo=UTC)

Mode = Literal["stream", "curate"]


def corpus(n: int, *, seed: int = 7) -> list[str]:
    """Seeded documents. About 30% are copies of an earlier one with 0-3 tokens replaced."""
    rng = random.Random(seed)  # noqa: S311 - benchmark data, not security
    docs: list[str] = []
    for _ in range(n):
        if docs and rng.random() < DUPLICATE_RATE:
            tokens = rng.choice(docs).split()
            for _ in range(rng.randint(0, 3)):
                tokens[rng.randrange(len(tokens))] = f"w{rng.randrange(VOCABULARY)}"
            docs.append(" ".join(tokens))
        else:
            docs.append(" ".join(f"w{rng.randrange(VOCABULARY)}" for _ in range(WORDS_PER_DOC)))
    return docs


# --------------------------------------------------------------------------- #
# Complexity                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ComplexityRow:
    n: int
    lsh_seconds: float
    lsh_comparisons: int
    lsh_dropped: int
    naive_seconds: float | None = None
    naive_comparisons: int | None = None
    naive_dropped: int | None = None
    agreement: float | None = None  # share of exhaustive drops that LSH also made


def naive_dedup(texts: Sequence[str], config: MinHashConfig = CONFIG) -> tuple[set[int], int]:
    """Exhaustive greedy keep-first: O(kept) exact comparisons per document."""
    index = MinHashLSHIndex(config)  # reused only for its feature extraction
    kept: list[frozenset[int]] = []
    dropped: set[int] = set()
    comparisons = 0
    for i, text in enumerate(texts):
        features = index.features(text)
        for other in kept:
            comparisons += 1
            if jaccard(features, other) >= config.threshold:
                dropped.add(i)
                break
        else:
            kept.append(features)
    return dropped, comparisons


def lsh_dedup(texts: Sequence[str], config: MinHashConfig = CONFIG) -> tuple[set[int], int]:
    index = MinHashLSHIndex(config)
    dropped = {i for i, text in enumerate(texts) if index.add_if_unique(f"d{i}", text) is not None}
    return dropped, index.stats.candidate_checks


def complexity(sizes: Sequence[int], *, naive_limit: int) -> list[ComplexityRow]:
    rows = []
    for n in sizes:
        texts = corpus(n)
        started = time.perf_counter()
        lsh_dropped, lsh_comparisons = lsh_dedup(texts)
        lsh_seconds = time.perf_counter() - started
        row = ComplexityRow(n, lsh_seconds, lsh_comparisons, len(lsh_dropped))
        if n <= naive_limit:
            started = time.perf_counter()
            naive_dropped, naive_comparisons = naive_dedup(texts)
            naive_seconds = time.perf_counter() - started
            agreement = (
                len(lsh_dropped & naive_dropped) / len(naive_dropped) if naive_dropped else 1.0
            )
            row = ComplexityRow(
                n,
                lsh_seconds,
                lsh_comparisons,
                len(lsh_dropped),
                naive_seconds,
                naive_comparisons,
                len(naive_dropped),
                agreement,
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Memory                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MemoryRow:
    n: int
    mode: Mode
    baseline_mib: float
    peak_mib: float
    kept: int

    @property
    def growth_mib(self) -> float:
        return self.peak_mib - self.baseline_mib


def write_jsonl(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i, text in enumerate(corpus(n)):
            record = {
                "sample_id": str(uuid.UUID(int=i + 1, version=4)),
                "system_prompt": "You are a helpful assistant.",
                "instruction": text,
                "response": "ok",
                "domain": "bench",
                "created_at_utc": (T0 + timedelta(seconds=i)).isoformat(),
            }
            handle.write(json.dumps(record) + "\n")
    return path


def _peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_UNIT / 2**20


def measure(mode: Mode, source: Path, workdir: Path) -> dict[str, float | int]:
    """Run one workload in this process; return baseline and peak RSS (MiB)."""
    baseline = _peak_mib()
    spec = ExportSpecification(
        output_path=f"{mode}.parquet", compression="zstd", row_group_size=10_000
    )
    reader = StreamingDatasetReader.jsonl(source, batch_size=512)

    async def stream() -> int:
        with AtomicParquetSink(spec, root=workdir) as sink:
            async for batch in reader.batches():
                sink.write(batch)
        return reader.stats.accepted

    async def curate() -> int:
        with AtomicParquetSink(spec, root=workdir) as sink:
            pipeline = CurationPipeline(MinHashLSHIndex(CONFIG), sink=sink, batch_size=512)
            metrics = await pipeline.run(reader.records())
        return metrics.deduped_count

    kept = asyncio.run(stream() if mode == "stream" else curate())
    return {"baseline_mib": baseline, "peak_mib": _peak_mib(), "kept": kept}


def memory(sizes: Sequence[int], workdir: Path) -> list[MemoryRow]:
    rows = []
    for n in sizes:
        source = write_jsonl(workdir / f"corpus-{n}.jsonl", n)
        for mode in ("stream", "curate"):
            completed = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
                [
                    sys.executable,
                    "-m",
                    "tools.benchmark",
                    "--measure",
                    mode,
                    str(source),
                    str(workdir),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            result = json.loads(completed.stdout)
            rows.append(
                MemoryRow(n, mode, result["baseline_mib"], result["peak_mib"], result["kept"])
            )
    return rows


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "processor": platform.processor() or "unknown",
    }


def _fmt(value: float | int | None, spec: str, suffix: str = "") -> str:
    return "—" if value is None else format(value, spec) + suffix


def complexity_table(rows: Sequence[ComplexityRow]) -> str:
    lines = [
        "| N | Exhaustive time | Exhaustive comparisons | LSH time | LSH comparisons "
        "| Comparison reduction | Drops (exhaustive / LSH) | LSH agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        reduction = (
            f"{r.naive_comparisons / r.lsh_comparisons:,.0f}x"
            if r.naive_comparisons is not None and r.lsh_comparisons
            else "—"
        )
        lines.append(
            f"| {r.n:,} | {_fmt(r.naive_seconds, '.2f', ' s')} | {_fmt(r.naive_comparisons, ',')} "
            f"| {r.lsh_seconds:.2f} s | {r.lsh_comparisons:,} | {reduction} "
            f"| {_fmt(r.naive_dropped, ',')} / {r.lsh_dropped:,} "
            f"| {_fmt(None if r.agreement is None else 100 * r.agreement, '.2f', '%')} |"
        )
    return "\n".join(lines)


def memory_table(rows: Sequence[MemoryRow]) -> str:
    lines = [
        "| Records | Mode | Baseline RSS | Peak RSS | Growth | Kept |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.n:,} | {r.mode} | {r.baseline_mib:.0f} MiB | {r.peak_mib:.0f} MiB "
            f"| {r.growth_mib:+.0f} MiB | {r.kept:,} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.benchmark")
    parser.add_argument("--quick", action="store_true", help="tiny sizes (seconds)")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/benchmark"))
    parser.add_argument(
        "--measure", nargs=3, metavar=("MODE", "SOURCE", "WORKDIR"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.measure:
        mode, source, workdir = args.measure
        print(json.dumps(measure(mode, Path(source), Path(workdir))))
        return 0
    sizes, naive_limit, memory_sizes = (
        ((200, 400), 400, (500,))
        if args.quick
        else (
            (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000),
            10_000,
            (10_000, 50_000, 100_000),
        )
    )
    complexity_rows = complexity(sizes, naive_limit=naive_limit)
    memory_rows = memory(memory_sizes, args.work_dir)
    payload = {
        "environment": environment(),
        "config": CONFIG.model_dump(),
        "complexity": [asdict(r) for r in complexity_rows],
        "memory": [asdict(r) | {"growth_mib": r.growth_mib} for r in memory_rows],
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["environment"]))
    print(complexity_table(complexity_rows))
    print()
    print(memory_table(memory_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
