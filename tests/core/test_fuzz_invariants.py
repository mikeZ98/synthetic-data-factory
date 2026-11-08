"""Invariant fuzzing for the deduplication engine and pipeline.

Each property below is either proved by fuzzing, or (where the naive
statement is false for greedy keep-first deduplication) replaced by its true
form, with the falsifying counterexample pinned as a test:

* idempotence: dedup(dedup(D)) == dedup(D).
* monotonicity: appending records that come later in canonical order never
  changes an earlier decision; exact-duplicate drops survive any addition.
  Inserting *earlier* near-duplicates can resurrect a drop (pinned).
* threshold: drops(t) >= drops(1.0) for every t. The drop count is *not*
  monotone in t (pinned: 1 drop at 0.7, 2 at 0.8).
* scale invariance: appending exact copies of D, or re-casing/re-spacing
  every text, leaves the kept set unchanged.
* schedule independence: for canonically ordered input the pipeline output is
  identical under any queue size, batch size, producer jitter and sink delay.
  Threaded and multi-task ingestion of exact duplicates keeps exactly one
  representative per content.
* model equivalence: with all candidates forced to collide, the index equals a
  brute-force greedy reference, including its exact accounting.
"""

import asyncio
import os
import random
import threading
import uuid
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    invariant,
    rule,
)

from synthetic_data_factory.core.deduplicator import (
    MinHashLSHIndex,
    Signature,
    canonical_order,
    default_text_of,
    jaccard,
)
from synthetic_data_factory.core.pipeline import CurationPipeline
from synthetic_data_factory.core.shingler import normalize_text
from synthetic_data_factory.schemas import InstructionSample, MinHashConfig

T0 = datetime(2026, 1, 1, tzinfo=UTC)
CONFIG = MinHashConfig(threshold=0.7, num_perm=64, shingle_k=3)
VOCAB = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
]
words = st.sampled_from(VOCAB)
configs = st.builds(
    MinHashConfig,
    threshold=st.sampled_from([0.5, 0.7, 0.85, 1.0]),
    num_perm=st.sampled_from([16, 64, 128]),
    shingle_k=st.sampled_from([2, 3]),
)


def make(text: str, n: int, *, at: datetime | None = None) -> InstructionSample:
    return InstructionSample(
        sample_id=str(uuid.UUID(int=n, version=4)),
        system_prompt="sys",
        instruction=text,
        response="ok",
        domain="fuzz",
        created_at_utc=at if at is not None else T0 + timedelta(seconds=n),
    )


@st.composite
def datasets(draw: st.DrawFn, *, max_size: int = 24) -> list[InstructionSample]:
    """Corpora rich in exact duplicates, one-token edits, and timestamp ties."""
    bases = draw(st.lists(st.lists(words, min_size=6, max_size=14), min_size=1, max_size=6))
    size = draw(st.integers(0, max_size))
    ids = draw(st.lists(st.integers(1, 2**60), min_size=size, max_size=size, unique=True))
    out = []
    for n in ids:
        tokens = list(draw(st.sampled_from(bases)))
        for _ in range(draw(st.integers(0, 2))):
            tokens[draw(st.integers(0, len(tokens) - 1))] = draw(words)
        at = T0 + timedelta(seconds=draw(st.integers(0, 5)))
        out.append(make(" ".join(tokens), n, at=at))
    return out


def dedup(
    samples: Sequence[InstructionSample], config: MinHashConfig = CONFIG
) -> list[InstructionSample]:
    return MinHashLSHIndex(config, max_candidates=10**6).deduplicate_batch(samples)[0]


def dropped(samples: Sequence[InstructionSample], config: MinHashConfig = CONFIG) -> set[str]:
    kept = {s.sample_id for s in dedup(samples, config)}
    return {s.sample_id for s in samples} - kept


class SetIndex(MinHashLSHIndex):
    """Features are the integer tokens of the text; every candidate collides."""

    def features(self, text: str) -> frozenset[int]:
        return frozenset(int(token) for token in text.split())

    def _band_keys(self, signature: Signature) -> list[int]:
        return [0] * self.bands


def as_text(values: set[int] | range) -> str:
    return " ".join(str(v) for v in sorted(values))


# --------------------------------------------------------------------------- #
# Idempotence                                                                 #
# --------------------------------------------------------------------------- #


@settings(max_examples=150, deadline=None)
@given(datasets(max_size=40), configs)
def test_strict_idempotence_across_configurations(
    samples: list[InstructionSample], config: MinHashConfig
) -> None:
    once = dedup(samples, config)
    assert dedup(once, config) == once
    assert dedup(dedup(once, config), config) == once


# --------------------------------------------------------------------------- #
# Monotonicity                                                                #
# --------------------------------------------------------------------------- #


@settings(max_examples=150, deadline=None)
@given(datasets(), datasets())
def test_appending_later_records_never_changes_earlier_decisions(
    head: list[InstructionSample], tail: list[InstructionSample]
) -> None:
    latest = max((s.created_at_utc for s in head), default=T0)
    head_ids = {s.sample_id for s in head}
    later = [
        make(s.instruction, 2**61 + i, at=latest + timedelta(hours=1, seconds=i))
        for i, s in enumerate(tail)
        if s.sample_id not in head_ids
    ]
    combined = {s.sample_id for s in dedup([*head, *later])}
    assert combined & head_ids == {s.sample_id for s in dedup(head)}  # no resurrection, no new drop


@settings(max_examples=150, deadline=None)
@given(datasets(), datasets(), st.data())
def test_exact_duplicate_drops_survive_any_addition(
    base: list[InstructionSample], extra: list[InstructionSample], data: st.DataObject
) -> None:
    if not base:
        return
    original = data.draw(st.sampled_from(base))
    twin = make(
        original.instruction.upper() + "  ", 2**62, at=original.created_at_utc + timedelta(days=1)
    )
    dataset = [*base, twin]
    assert twin.sample_id in dropped(dataset)
    base_ids = {s.sample_id for s in dataset}
    additions = [s for s in extra if s.sample_id not in base_ids]  # arbitrary timestamps
    assert twin.sample_id in dropped([*dataset, *additions])


def test_counterexample_earlier_insertion_can_resurrect_a_near_duplicate() -> None:
    # J(Z, A) = 0.8, J(A, B) = 0.8, J(Z, B) = 0.6: similarity is not transitive.
    z, a, b = set(range(0, 10)), set(range(1, 11)), set(range(2, 12))
    assert (jaccard(z, a), jaccard(a, b)) == (9 / 11, 9 / 11) and jaccard(z, b) < 0.7
    without_z = SetIndex(CONFIG)
    assert without_z.add_if_unique("A", as_text(a)) is None
    assert without_z.add_if_unique("B", as_text(b)) == "A"  # B dropped
    with_z = SetIndex(CONFIG)
    assert with_z.add_if_unique("Z", as_text(z)) is None  # Z is earlier in canonical order
    assert with_z.add_if_unique("A", as_text(a)) == "Z"
    assert with_z.add_if_unique("B", as_text(b)) is None  # B resurrected


# --------------------------------------------------------------------------- #
# Threshold behaviour and scale invariance                                    #
# --------------------------------------------------------------------------- #


@settings(max_examples=150, deadline=None)
@given(datasets(max_size=30), st.sampled_from([0.5, 0.7, 0.85, 0.95]))
def test_any_threshold_drops_at_least_the_exact_duplicates(
    samples: list[InstructionSample], threshold: float
) -> None:
    exact = MinHashConfig(threshold=1.0, num_perm=64, shingle_k=3)
    fuzzy = MinHashConfig(threshold=threshold, num_perm=64, shingle_k=3)
    assert dropped(samples, exact) <= dropped(samples, fuzzy)
    distinct = {normalize_text(default_text_of(s)) for s in samples}
    assert len(dedup(samples, exact)) == len(distinct)


def test_counterexample_drop_count_is_not_monotone_in_threshold() -> None:
    # Greedy keep-first: at 0.7, B falls to A and frees C and D; at 0.8, B
    # survives and absorbs both. Raising the threshold drops *more*.
    a, b = {0, 1, *range(4, 10), 12}, set(range(10))
    c, d = set(range(1, 11)), {0, *range(2, 10), 11}
    assert jaccard(a, b) == 8 / 11 and jaccard(b, c) == jaccard(b, d) == 9 / 11
    assert max(jaccard(a, c), jaccard(a, d), jaccard(c, d)) < 0.7

    def drops(threshold: float) -> int:
        index = SetIndex(MinHashConfig(threshold=threshold, num_perm=16, shingle_k=2))
        return sum(
            index.add_if_unique(k, as_text(v)) is not None
            for k, v in zip("ABCD", (a, b, c, d), strict=True)
        )

    assert (drops(0.7), drops(0.8)) == (1, 2)


@settings(max_examples=100, deadline=None)
@given(datasets())
def test_scale_invariance_under_appended_copies(samples: list[InstructionSample]) -> None:
    latest = max((s.created_at_utc for s in samples), default=T0)
    copies = [
        make(s.instruction, 2**61 + i, at=latest + timedelta(hours=1, seconds=i))
        for i, s in enumerate(sorted(samples, key=canonical_order))
    ]
    assert dedup([*samples, *copies]) == dedup(samples)


@settings(max_examples=100, deadline=None)
@given(datasets())
def test_normalization_invariance(samples: list[InstructionSample]) -> None:
    variants = [
        s.model_copy(update={"instruction": "  " + s.instruction.upper().replace(" ", " \t ")})
        for s in samples
    ]
    assert [s.sample_id for s in dedup(variants)] == [s.sample_id for s in dedup(samples)]


# --------------------------------------------------------------------------- #
# Model-based stateful fuzzing                                                #
# --------------------------------------------------------------------------- #


class BruteForceModel(RuleBasedStateMachine):
    """Forced collisions turn LSH into exhaustive search: index must equal the reference."""

    keys = Bundle("keys")

    def __init__(self) -> None:
        super().__init__()
        self.index = SetIndex(
            MinHashConfig(threshold=0.6, num_perm=16, shingle_k=2), max_candidates=10**6
        )
        self.kept: list[tuple[str, frozenset[int]]] = []
        self.expected_checks = 0
        self.counter = 0

    def reference(self, features: frozenset[int]) -> str | None:
        self.expected_checks += next(
            (i + 1 for i, (_, f) in enumerate(self.kept) if jaccard(features, f) >= 0.6),
            len(self.kept),
        )
        return next((k for k, f in self.kept if jaccard(features, f) >= 0.6), None)

    @rule(target=keys, values=st.frozensets(st.integers(0, 12), min_size=1, max_size=8))
    def add(self, values: frozenset[int]) -> str:
        self.counter += 1
        key = f"k{self.counter}"
        expected = self.reference(values)
        assert self.index.add_if_unique(key, as_text(set(values))) == expected
        if expected is None:
            self.kept.append((key, values))
        return key

    @rule(key=keys, values=st.frozensets(st.integers(0, 12), min_size=1, max_size=8))
    def re_add_existing_key(self, key: str, values: frozenset[int]) -> None:
        if key in self.index:
            before = self.index.stats.candidate_checks
            assert self.index.add_if_unique(key, as_text(set(values))) == key
            assert self.index.stats.candidate_checks == before  # identity check short-circuits

    @rule(values=st.frozensets(st.integers(0, 12), min_size=1, max_size=8))
    def query(self, values: frozenset[int]) -> None:
        assert self.index.query(as_text(set(values))) == self.reference(values)

    @invariant()
    def accounting_is_exact(self) -> None:
        stats = self.index.stats
        assert stats.documents == len(self.index) == len(self.kept)
        assert stats.candidate_checks == self.expected_checks
        assert sum(stats.similarity_histogram) == stats.candidate_checks
        assert stats.truncated_queries == 0


TestBruteForceModel = BruteForceModel.TestCase
TestBruteForceModel.settings = settings(max_examples=60, stateful_step_count=30, deadline=None)


class LSHSoundness(RuleBasedStateMachine):
    """Real banding: every drop has an exact witness; exact copies are always caught."""

    def __init__(self) -> None:
        super().__init__()
        self.index = MinHashLSHIndex(CONFIG)
        self.texts: dict[str, str] = {}
        self.counter = 0

    @rule(tokens=st.lists(words, min_size=3, max_size=14))
    def add(self, tokens: list[str]) -> None:
        self.counter += 1
        key, text = f"k{self.counter}", " ".join(tokens)
        partner = self.index.add_if_unique(key, text)
        if partner is None:
            self.texts[key] = text
        else:
            witness = self.index.features(self.texts[partner])
            assert jaccard(self.index.features(text), witness) >= CONFIG.threshold

    @rule(data=st.data())
    def exact_copy_is_detected(self, data: st.DataObject) -> None:
        if self.texts:
            key = data.draw(st.sampled_from(sorted(self.texts)))
            self.counter += 1
            assert self.index.add_if_unique(f"k{self.counter}", self.texts[key].upper()) is not None

    @invariant()
    def kept_documents_are_indexed(self) -> None:
        assert len(self.index) == len(self.texts)
        assert all(key in self.index for key in self.texts)


TestLSHSoundness = LSHSoundness.TestCase
TestLSHSoundness.settings = settings(max_examples=60, stateful_step_count=30, deadline=None)


# --------------------------------------------------------------------------- #
# Concurrency and schedule independence                                       #
# --------------------------------------------------------------------------- #


@settings(max_examples=60, deadline=None)
@given(
    samples=datasets(max_size=30),
    queue_maxsize=st.integers(1, 6),
    batch_size=st.integers(1, 9),
    jitter=st.lists(st.integers(0, 3), min_size=1, max_size=8),
    sink_yields=st.integers(0, 3),
)
def test_pipeline_output_is_schedule_independent(
    samples: list[InstructionSample],
    queue_maxsize: int,
    batch_size: int,
    jitter: list[int],
    sink_yields: int,
) -> None:
    ordered = sorted(samples, key=canonical_order)
    expected = dedup(ordered)

    async def scenario() -> list[InstructionSample]:
        received: list[InstructionSample] = []

        async def source() -> AsyncIterator[InstructionSample]:
            for i, sample in enumerate(ordered):
                for _ in range(jitter[i % len(jitter)]):
                    await asyncio.sleep(0)
                yield sample

        async def sink(batch: Sequence[InstructionSample]) -> None:
            for _ in range(sink_yields):
                await asyncio.sleep(0)
            received.extend(batch)

        index = MinHashLSHIndex(CONFIG, max_candidates=10**6)
        pipeline = CurationPipeline(
            index, sink=sink, queue_maxsize=queue_maxsize, batch_size=batch_size
        )
        metrics = await asyncio.wait_for(pipeline.run(source()), 10)
        assert metrics.deduped_count == len(received)
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return received

    assert asyncio.run(scenario()) == expected


@st.composite
def duplicate_contents(draw: st.DrawFn) -> list[str]:
    distinct = draw(
        st.lists(
            st.lists(words, min_size=4, max_size=10).map(" ".join),
            min_size=1,
            max_size=12,
            unique=True,
        )
    )
    return draw(st.lists(st.sampled_from(distinct), min_size=1, max_size=60))


def _exact_index() -> MinHashLSHIndex:
    return MinHashLSHIndex(MinHashConfig(threshold=1.0, num_perm=64, shingle_k=2))


@settings(max_examples=40, deadline=None)
@given(duplicate_contents(), st.integers(2, 8), st.randoms(use_true_random=False))
def test_threaded_ingestion_keeps_one_per_content(
    contents: list[str], threads: int, rng: random.Random
) -> None:
    index = _exact_index()
    jobs = list(enumerate(contents))
    rng.shuffle(jobs)
    barrier = threading.Barrier(threads)

    def ingest(chunk: list[tuple[int, str]]) -> list[tuple[int, str | None]]:
        barrier.wait()  # maximise overlap
        return [(n, index.add_if_unique(f"k{n}", text)) for n, text in chunk]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = [
            r for part in pool.map(ingest, [jobs[i::threads] for i in range(threads)]) for r in part
        ]
    kept = [contents[n] for n, partner in results if partner is None]
    assert sorted(kept) == sorted(set(contents))  # exactly one representative per content
    assert len(index) == len(set(contents))


@settings(max_examples=40, deadline=None)
@given(duplicate_contents(), st.integers(2, 8), st.lists(st.integers(0, 2), min_size=1, max_size=6))
def test_multi_task_ingestion_keeps_one_per_content(
    contents: list[str], tasks: int, yields: list[int]
) -> None:
    async def scenario() -> list[tuple[int, str | None]]:
        index = _exact_index()

        async def ingest(worker: int) -> list[tuple[int, str | None]]:
            out = []
            for n in range(worker, len(contents), tasks):
                for _ in range(yields[n % len(yields)]):
                    await asyncio.sleep(0)
                out.append((n, index.add_if_unique(f"k{n}", contents[n])))
            return out

        parts = await asyncio.gather(*(ingest(w) for w in range(tasks)))
        assert len(index) == len(set(contents))
        return [r for part in parts for r in part]

    results = asyncio.run(scenario())
    assert sorted(contents[n] for n, partner in results if partner is None) == sorted(set(contents))


# --------------------------------------------------------------------------- #
# Zero-leak proof under repetition                                            #
# --------------------------------------------------------------------------- #


def test_repeated_pipeline_runs_leak_no_tasks_threads_or_descriptors() -> None:
    samples = [make(f"text number {n % 7} with shared words", n) for n in range(1, 200)]

    async def once() -> int:
        async def source() -> AsyncIterator[InstructionSample]:
            for sample in samples:
                yield sample

        kept: list[InstructionSample] = []
        await CurationPipeline(MinHashLSHIndex(CONFIG), sink=kept.extend, batch_size=16).run(
            source()
        )
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return len(kept)

    asyncio.run(once())  # warm-up: lazy imports and interpreter caches
    fds, threads = len(os.listdir("/dev/fd")), threading.active_count()
    assert {asyncio.run(once()) for _ in range(25)} == {7}
    assert (len(os.listdir("/dev/fd")), threading.active_count()) == (fds, threads)


@pytest.mark.parametrize("threshold", [0.5, 1.0])
def test_empty_and_singleton_datasets(threshold: float) -> None:
    config = MinHashConfig(threshold=threshold, num_perm=16, shingle_k=2)
    assert dedup([], config) == []
    only = make("alpha beta gamma", 1)
    assert dedup([only], config) == [only]
