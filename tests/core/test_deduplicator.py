"""MinHashLSHIndex: exact Jaccard, banding, collisions, determinism, idempotence, thread safety."""

import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthetic_data_factory.core.deduplicator import (
    MERSENNE_PRIME_31,
    MinHashLSHIndex,
    Signature,
    canonical_order,
    default_text_of,
    jaccard,
    optimal_bands,
    similarity_bucket,
)
from synthetic_data_factory.core.shingler import TextShingler, normalize_text
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
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]

words = st.sampled_from(VOCAB)
word_texts = st.lists(words, min_size=3, max_size=24).map(" ".join)


def make_sample(text: str, n: int, *, at: datetime = T0, response: str = "ok") -> InstructionSample:
    return InstructionSample(
        sample_id=str(uuid.UUID(int=n, version=4)),
        system_prompt="You are helpful.",
        instruction=text,
        response=response,
        domain="test",
        created_at_utc=at,
    )


def long_text(seed: int, n_words: int = 60) -> str:
    return " ".join(f"tok{seed}x{i}" for i in range(n_words))


@st.composite
def datasets(draw: st.DrawFn) -> list[InstructionSample]:
    """Small corpora rich in exact and near duplicates and timestamp ties."""
    bases = draw(st.lists(st.lists(words, min_size=8, max_size=16), min_size=1, max_size=5))
    size = draw(st.integers(min_value=0, max_value=12))
    ids = draw(st.lists(st.integers(1, 2**64), min_size=size, max_size=size, unique=True))
    samples = []
    for n in ids:
        tokens = list(draw(st.sampled_from(bases)))
        if draw(st.booleans()):
            tokens[draw(st.integers(0, len(tokens) - 1))] = draw(words)
        at = T0 + timedelta(seconds=draw(st.integers(0, 2)))
        samples.append(make_sample(" ".join(tokens), n, at=at))
    return samples


class CollidingIndex(MinHashLSHIndex):
    """Every document lands in the same bucket of every band: worst-case LSH collisions."""

    def _band_keys(self, signature: Signature) -> list[int]:
        return [0] * self.bands


def reference_signature(index: MinHashLSHIndex, features: frozenset[int]) -> list[int]:
    """Pure-Python big-int reference for the vectorized signature."""
    p = MERSENNE_PRIME_31
    a = [int(v) for v in index._a]
    b = [int(v) for v in index._b]
    return [min((ai * (x % p) + bi) % p for x in features) for ai, bi in zip(a, b, strict=True)]


# --------------------------------------------------------------------------- #
# Exact Jaccard                                                               #
# --------------------------------------------------------------------------- #


@given(word_texts)
def test_jaccard_of_identical_strings_is_one(text: str) -> None:
    shingler = TextShingler(3)
    assert jaccard(shingler.hash_set(text), shingler.hash_set(text)) == 1.0
    assert jaccard(shingler.hash_set(text), shingler.hash_set(f"  {text.upper()}  ")) == 1.0


@given(
    st.lists(st.from_regex(r"[a-m]{2,6}", fullmatch=True), min_size=3, max_size=20),
    st.lists(st.from_regex(r"[n-z]{2,6}", fullmatch=True), min_size=3, max_size=20),
)
def test_jaccard_of_disjoint_strings_is_zero(left: list[str], right: list[str]) -> None:
    shingler = TextShingler(3)
    assert jaccard(shingler.hash_set(" ".join(left)), shingler.hash_set(" ".join(right))) == 0.0


@given(st.frozensets(st.integers(0, 50)), st.frozensets(st.integers(0, 50)))
def test_jaccard_is_symmetric_bounded_and_exact(a: frozenset[int], b: frozenset[int]) -> None:
    value = jaccard(a, b)
    assert value == jaccard(b, a)
    assert 0.0 <= value <= 1.0
    if a or b:
        assert value == len(a & b) / len(a | b)


def test_jaccard_edge_cases() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, set()) == 0.0
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5


# --------------------------------------------------------------------------- #
# Band partitioning                                                           #
# --------------------------------------------------------------------------- #


@given(
    st.sampled_from([1, 2, 12, 16, 64, 100, 128, 256]),
    st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
)
def test_optimal_bands_is_brute_force_argmin(num_perm: int, threshold: float) -> None:
    b, r = optimal_bands(num_perm, threshold)
    assert b * r == num_perm
    error = abs((1 / b) ** (1 / r) - threshold)
    for cand_b in range(1, num_perm + 1):
        if num_perm % cand_b == 0:
            cand_r = num_perm // cand_b
            cand_error = abs((1 / cand_b) ** (1 / cand_r) - threshold)
            assert (error, b) <= (cand_error, cand_b)


@pytest.mark.parametrize(
    ("num_perm", "threshold", "expected"),
    [(128, 1.0, (1, 128)), (128, 0.85, (8, 16)), (128, 0.5, (32, 4)), (64, 0.7, (8, 8))],
)
def test_optimal_bands_known_values(
    num_perm: int, threshold: float, expected: tuple[int, int]
) -> None:
    assert optimal_bands(num_perm, threshold) == expected


@pytest.mark.parametrize(("num_perm", "threshold"), [(0, 0.5), (-4, 0.5), (16, 0.0), (16, 1.01)])
def test_optimal_bands_rejects_invalid_arguments(num_perm: int, threshold: float) -> None:
    with pytest.raises(ValueError):
        optimal_bands(num_perm, threshold)


# --------------------------------------------------------------------------- #
# Signatures                                                                  #
# --------------------------------------------------------------------------- #


def test_signature_matches_golden_prefix() -> None:
    index = MinHashLSHIndex(MinHashConfig(threshold=0.8, num_perm=128, shingle_k=3))
    signature = index.signature(index.features("abc def ghi"))
    assert signature.dtype == np.uint64
    assert signature[:4].tolist() == [1871656463, 557777652, 477647881, 1166795949]


@settings(max_examples=50)
@given(st.frozensets(st.integers(0, 2**64 - 1), min_size=1, max_size=40))
def test_signature_matches_big_int_reference(features: frozenset[int]) -> None:
    index = MinHashLSHIndex(CONFIG)
    signature = index.signature(features)
    assert signature.tolist() == reference_signature(index, features)
    assert int(signature.max()) < MERSENNE_PRIME_31


def test_signature_is_exact_across_multiple_blocks() -> None:
    # num_perm=4096 -> 256-column blocks; 1000 features span 4 blocks.
    index = MinHashLSHIndex(MinHashConfig(threshold=0.9, num_perm=4096, shingle_k=2))
    assert index._block == 256
    features = frozenset(range(10**12, 10**12 + 1000))
    assert index.signature(features).tolist() == reference_signature(index, features)


@given(st.lists(st.integers(0, 2**64 - 1), min_size=1, max_size=30))
def test_signature_is_independent_of_feature_order(values: list[int]) -> None:
    index = MinHashLSHIndex(CONFIG)
    forward = index.signature(frozenset(values))
    assert np.array_equal(forward, index.signature(set(reversed(values))))


def test_signature_rejects_empty_feature_set() -> None:
    with pytest.raises(ValueError, match="empty feature set"):
        MinHashLSHIndex(CONFIG).signature(frozenset())


def test_signature_agreement_estimates_jaccard() -> None:
    index = MinHashLSHIndex(MinHashConfig(threshold=0.5, num_perm=1024, shingle_k=2))
    left = index.features(long_text(1, 200))
    right = index.features(" ".join(long_text(1, 200).split()[:150] + long_text(2, 50).split()))
    estimate = float(np.mean(index.signature(left) == index.signature(right)))
    assert abs(estimate - jaccard(left, right)) < 0.05


def test_seed_changes_signatures_deterministically() -> None:
    features = frozenset({1, 2, 3})
    s0 = MinHashLSHIndex(CONFIG, seed=0).signature(features)
    assert np.array_equal(s0, MinHashLSHIndex(CONFIG, seed=0).signature(features))
    assert not np.array_equal(s0, MinHashLSHIndex(CONFIG, seed=1).signature(features))


# --------------------------------------------------------------------------- #
# Insert / query semantics                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("max_candidates", [0, -1, True])
def test_rejects_invalid_max_candidates(max_candidates: int) -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        MinHashLSHIndex(CONFIG, max_candidates=max_candidates)


def test_exact_and_near_duplicates_are_detected_dissimilar_kept() -> None:
    index = MinHashLSHIndex(CONFIG)
    base = long_text(1)
    assert index.add_if_unique("a", base) is None
    assert index.add_if_unique("b", base.upper()) == "a"  # identical after normalization
    assert index.add_if_unique("c", base + " one more") == "a"  # J = 58/59
    assert index.add_if_unique("d", long_text(2)) is None
    assert index.add_if_unique("a", long_text(3)) == "a"  # key already indexed
    assert len(index) == 2
    assert "a" in index and "d" in index and "b" not in index


def test_query_returns_earliest_match_and_does_not_mutate() -> None:
    index = MinHashLSHIndex(CONFIG)
    index.add_if_unique("first", long_text(1))
    assert index.query(long_text(1)) == "first"
    assert index.query(long_text(9)) is None
    assert len(index) == 1


def test_texts_shorter_than_k_fall_back_to_exact_fingerprint() -> None:
    index = MinHashLSHIndex(CONFIG)  # k = 3 words
    assert index.features("Hi there") == frozenset({index.shingler.fingerprint("hi there")})
    assert index.add_if_unique("a", "Hi there") is None
    assert index.add_if_unique("b", "  HI   there ") == "a"
    assert index.add_if_unique("c", "Hi you") is None


def test_threshold_one_keeps_near_duplicates_and_drops_exact_ones() -> None:
    index = MinHashLSHIndex(MinHashConfig(threshold=1.0, num_perm=128, shingle_k=3))
    assert (index.bands, index.rows) == (1, 128)
    assert index.add_if_unique("a", long_text(1)) is None
    assert index.add_if_unique("b", long_text(1) + " tail") is None
    assert index.add_if_unique("c", long_text(1)) == "a"


# --------------------------------------------------------------------------- #
# Synthetic LSH collisions vs exact Jaccard filtering                         #
# --------------------------------------------------------------------------- #


def test_forced_collisions_never_cause_false_positives() -> None:
    index = CollidingIndex(CONFIG)
    for seed in range(10):
        assert index.add_if_unique(f"d{seed}", long_text(seed)) is None
    assert len(index) == 10
    assert index.stats.candidate_checks == sum(range(10))  # every pair was verified


@given(word_texts, word_texts)
def test_colliding_pair_is_duplicate_iff_exact_jaccard_meets_threshold(a: str, b: str) -> None:
    index = CollidingIndex(CONFIG)
    index.add_if_unique("a", a)
    similarity = jaccard(index.features(a), index.features(b))
    assert (index.query(b) == "a") is (similarity >= CONFIG.threshold)


def test_query_latency_is_bounded_by_max_candidates() -> None:
    index = CollidingIndex(CONFIG, max_candidates=5)
    for seed in range(20):
        index.add_if_unique(f"d{seed}", long_text(seed))
    before = index.stats
    assert index.query(long_text(99)) is None
    after = index.stats
    assert after.candidate_checks - before.candidate_checks == 5
    assert after.truncated_queries - before.truncated_queries == 1


def test_exactly_max_candidates_without_match_is_not_truncated() -> None:
    index = CollidingIndex(CONFIG, max_candidates=5)
    for seed in range(5):
        index.add_if_unique(f"d{seed}", long_text(seed))
    assert index.query(long_text(99)) is None
    assert index.stats.truncated_queries == 0
    assert index.stats.candidate_checks == sum(range(5)) + 5


def test_stats_report_banding() -> None:
    stats = MinHashLSHIndex(CONFIG).stats
    assert (stats.documents, stats.bands, stats.rows) == (0, 8, 8)


# --------------------------------------------------------------------------- #
# Batch deduplication: tie-breaking, determinism, idempotence                 #
# --------------------------------------------------------------------------- #


def test_tie_break_keeps_earliest_timestamp_then_lowest_sample_id() -> None:
    text = long_text(1)
    late = make_sample(text, 1, at=T0 + timedelta(hours=1))
    early_high_id = make_sample(text, 3)
    early_low_id = make_sample(text, 2)
    kept, metrics = MinHashLSHIndex(CONFIG).deduplicate_batch([late, early_high_id, early_low_id])
    assert kept == [early_low_id]
    assert (metrics.raw_count, metrics.schema_valid_count, metrics.deduped_count) == (3, 3, 1)
    assert metrics.retention_rate == pytest.approx(1 / 3)


def test_dedup_key_uses_instruction_and_response_not_system_prompt() -> None:
    sample = make_sample("question text here", 1, response="answer")
    assert default_text_of(sample) == "question text here\nanswer"
    assert canonical_order(sample) == (T0, sample.sample_id)
    other = make_sample("question text here", 2, response="a different answer entirely")
    kept, _ = MinHashLSHIndex(CONFIG).deduplicate_batch([sample, other])
    assert kept == [sample, other]


def test_empty_batch() -> None:
    kept, metrics = MinHashLSHIndex(CONFIG).deduplicate_batch([])
    assert kept == []
    assert metrics.retention_rate == 0.0


def test_batches_deduplicate_as_one_stream_and_resubmission_is_dropped() -> None:
    index = MinHashLSHIndex(CONFIG)
    first = [make_sample(long_text(1), 1), make_sample(long_text(2), 2)]
    kept, _ = index.deduplicate_batch(first)
    assert kept == first
    kept, metrics = index.deduplicate_batch([make_sample(long_text(1), 3), *first])
    assert kept == []
    assert metrics.deduped_count == 0


@settings(max_examples=150)
@given(datasets())
def test_deduplicate_is_idempotent(samples: list[InstructionSample]) -> None:
    once, _ = MinHashLSHIndex(CONFIG).deduplicate_batch(samples)
    twice, metrics = MinHashLSHIndex(CONFIG).deduplicate_batch(once)
    assert twice == once
    assert metrics.retention_rate == (1.0 if once else 0.0)


@settings(max_examples=150)
@given(datasets(), st.randoms(use_true_random=False))
def test_deduplicate_is_permutation_invariant(
    samples: list[InstructionSample], rng: random.Random
) -> None:
    shuffled = list(samples)
    rng.shuffle(shuffled)
    expected, _ = MinHashLSHIndex(CONFIG).deduplicate_batch(samples)
    actual, _ = MinHashLSHIndex(CONFIG).deduplicate_batch(shuffled)
    assert actual == expected
    assert actual == sorted(actual, key=canonical_order)


@settings(max_examples=150)
@given(datasets())
def test_deduplicate_is_sound_and_complete_for_exact_copies(
    samples: list[InstructionSample],
) -> None:
    index = MinHashLSHIndex(CONFIG)
    kept_keys: set[str] = set()
    for sample in sorted(samples, key=canonical_order):
        partner = index.add_if_unique(sample.sample_id, default_text_of(sample))
        if partner is None:
            kept_keys.add(sample.sample_id)
        else:  # soundness: every drop is backed by an exact-Jaccard witness
            witness = next(s for s in samples if s.sample_id == partner)
            similarity = jaccard(
                index.features(default_text_of(sample)), index.features(default_text_of(witness))
            )
            assert similarity >= CONFIG.threshold
    kept_texts = [normalize_text(default_text_of(s)) for s in samples if s.sample_id in kept_keys]
    assert len(kept_texts) == len(set(kept_texts))  # completeness for exact copies


# --------------------------------------------------------------------------- #
# Thread safety                                                               #
# --------------------------------------------------------------------------- #


def test_concurrent_inserts_of_one_document_keep_exactly_one() -> None:
    index = MinHashLSHIndex(CONFIG)
    text = long_text(1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: index.add_if_unique(f"k{n}", text), range(200)))
    winners = [n for n, result in enumerate(results) if result is None]
    assert len(winners) == 1
    assert len(index) == 1
    assert all(result == f"k{winners[0]}" for result in results if result is not None)


def test_concurrent_inserts_of_distinct_documents_lose_nothing() -> None:
    index = MinHashLSHIndex(CONFIG)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: index.add_if_unique(f"k{n}", long_text(n)), range(100)))
    assert results == [None] * 100
    assert len(index) == index.stats.documents == 100
    assert sorted(index._ordinal_of.values()) == list(range(100))


# --------------------------------------------------------------------------- #
# Similarity histogram telemetry                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("similarity", "bucket"),
    [(0.0, 0), (0.4999, 0), (0.5, 1), (0.6999, 1), (0.7, 2), (0.8499, 2), (0.85, 3), (1.0, 3)],
)
def test_similarity_bucket_edges_are_lower_inclusive(similarity: float, bucket: int) -> None:
    assert similarity_bucket(similarity) == bucket


def test_histogram_records_every_verified_candidate() -> None:
    index = CollidingIndex(CONFIG)  # every insert verifies against all previous documents
    base = long_text(1).split()
    variants = [
        base,
        base[:57] + ["a", "b", "c"],
        base[:30] + long_text(2).split()[:30],
        long_text(3).split(),
    ]
    for n, tokens in enumerate(variants):
        index.add_if_unique(f"d{n}", " ".join(tokens))
    stats = index.stats
    assert sum(stats.similarity_histogram) == stats.candidate_checks
    assert len(stats.similarity_histogram) == 4
    assert stats.similarity_histogram[3] >= 1  # the near-duplicate variant
    assert stats.similarity_histogram[0] >= 1  # the unrelated document


def test_fresh_index_has_empty_histogram() -> None:
    assert MinHashLSHIndex(CONFIG).stats.similarity_histogram == (0, 0, 0, 0)
