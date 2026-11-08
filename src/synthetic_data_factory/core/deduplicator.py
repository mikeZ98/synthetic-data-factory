"""MinHash LSH near-duplicate index with exact Jaccard verification.

Pipeline per document
---------------------
1. Features: the set of 64-bit shingle hashes (:class:`TextShingler`). A text
   shorter than ``k`` falls back to the singleton ``{fingerprint(text)}``, so
   exact copies of short texts are still deduplicated rather than rejected.
2. Signature: ``sig[i] = min over x in features of (a_i * (x mod P) + b_i) mod P``
   with ``P = 2**31 - 1`` and ``a_i in [1, P)``, ``b_i in [0, P)`` derived from
   the seed via BLAKE2b. Every operand is below ``2**31``, so ``a*x + b`` is
   below ``2**63`` and ``uint64`` arithmetic is exact (no wraparound).
3. Banding: the signature is split into ``b`` bands of ``r`` rows,
   ``b * r == num_perm``, choosing the factorization whose S-curve inflection
   ``(1/b) ** (1/r)`` is closest to the threshold. Each band is hashed to a
   bucket key from its little-endian bytes, so keys are platform-independent.
4. Verification: an LSH candidate counts as a duplicate only if its exact
   Jaccard similarity on feature sets is ``>= threshold``. LSH collisions can
   therefore never cause false positives. LSH misses (false negatives) remain
   possible, which is inherent to the method.

Concurrency
-----------
Signatures are computed outside the lock (pure functions). Check-then-insert
(:meth:`MinHashLSHIndex.add_if_unique`) runs under a single ``threading.Lock``,
so concurrent inserts of mutual duplicates keep exactly one. The lock is never
held across an ``await``, which makes it safe to call from an event loop.

Bounded query latency
---------------------
Each band's bucket lists document ordinals in insertion order. A query merges
its buckets lazily (``heapq.merge``) in ascending ordinal order and verifies at
most ``max_candidates`` distinct candidates. Per-query cost is therefore
O(b + max_candidates * (log b + |A| + |B|)), whatever the bucket sizes. A query
that reaches the cap without a match is counted in
``IndexStats.truncated_queries``, so truncation is observable, never silent.
"""

import bisect
import hashlib
import heapq
import threading
from collections.abc import Callable, Iterator, Sequence, Set
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
from numpy.typing import NDArray

from synthetic_data_factory.core.shingler import ShingleError, ShingleMode, TextShingler
from synthetic_data_factory.schemas import CurationMetrics, InstructionSample, MinHashConfig

MERSENNE_PRIME_31: Final = (1 << 31) - 1
DEFAULT_MAX_CANDIDATES: Final = 1_024
SIMILARITY_BUCKET_EDGES: Final = (0.5, 0.7, 0.85)
"""Histogram buckets of verified candidate similarity: [0, .5) [.5, .7) [.7, .85) [.85, 1]."""
_MAX_BLOCK_ELEMENTS: Final = 1 << 20  # caps the (num_perm x block) temporary at 8 MiB
_PERMUTATION_PERSON: Final = b"sdf/minhash/perm"
_BAND_PERSON: Final = b"sdf/minhash/band"

Signature = NDArray[np.uint64]
TextOf = Callable[[InstructionSample], str]


def jaccard(a: Set[int], b: Set[int]) -> float:
    """Exact Jaccard similarity ``|A & B| / |A | B|``. Two empty sets count as identical (1.0)."""
    if not a and not b:
        return 1.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    intersection = sum(1 for item in small if item in large)
    return intersection / (len(a) + len(b) - intersection)


def similarity_bucket(similarity: float) -> int:
    """Index into :data:`SIMILARITY_BUCKET_EDGES` buckets; each lower edge is inclusive."""
    return bisect.bisect_right(SIMILARITY_BUCKET_EDGES, similarity)


def optimal_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Return ``(b, r)`` with ``b * r == num_perm`` minimizing ``|(1/b)**(1/r) - threshold|``.

    Ties go to the smaller ``b`` (fewer, stricter bands, so fewer candidates).
    ``threshold == 1.0`` yields ``(1, num_perm)``: one band, which matches
    only identical signatures.
    """
    if num_perm < 1:
        raise ValueError(f"num_perm must be positive, got {num_perm}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    factorizations = ((b, num_perm // b) for b in range(1, num_perm + 1) if num_perm % b == 0)
    return min(
        factorizations, key=lambda br: (abs((1.0 / br[0]) ** (1.0 / br[1]) - threshold), br[0])
    )


def default_text_of(sample: InstructionSample) -> str:
    """Dedup key: instruction plus response.

    The system prompt is excluded on purpose. It is usually shared boilerplate
    and would inflate similarity between unrelated samples.
    """
    return f"{sample.instruction}\n{sample.response}"


def canonical_order(sample: InstructionSample) -> tuple[datetime, str]:
    """Total order used for tie-breaking: earliest ``created_at_utc``, then ``sample_id``."""
    return (sample.created_at_utc, sample.sample_id)


@dataclass(frozen=True, slots=True)
class IndexStats:
    documents: int
    bands: int
    rows: int
    candidate_checks: int
    truncated_queries: int
    similarity_histogram: tuple[int, ...]
    """Exact Jaccard of every verified candidate, bucketed by :data:`SIMILARITY_BUCKET_EDGES`."""


class MinHashLSHIndex:
    """Stateful near-duplicate index keyed by sample id.

    Only documents that were kept are inserted. Deduplicating a stream therefore
    means comparing each document against the documents kept before it.
    """

    def __init__(
        self,
        config: MinHashConfig,
        *,
        mode: ShingleMode = "word",
        seed: int = 0,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        text_of: TextOf = default_text_of,
    ) -> None:
        if isinstance(max_candidates, bool) or max_candidates < 1:
            raise ValueError(f"max_candidates must be a positive integer, got {max_candidates!r}")
        self.config: Final = config
        self.shingler: Final = TextShingler(config.shingle_k, mode=mode, seed=seed)
        self.max_candidates: Final = max_candidates
        self.bands, self.rows = optimal_bands(config.num_perm, config.threshold)
        self._text_of: Final = text_of
        self._a, self._b = _permutation_params(config.num_perm, seed)
        # num_perm <= MAX_NUM_PERM (4096), so the block is at least 256 columns.
        self._block: Final = _MAX_BLOCK_ELEMENTS // config.num_perm
        self._buckets: Final[list[dict[int, list[int]]]] = [{} for _ in range(self.bands)]
        self._features: Final[list[frozenset[int]]] = []
        self._keys: Final[list[str]] = []
        self._ordinal_of: Final[dict[str, int]] = {}
        self._lock: Final = threading.Lock()
        self._candidate_checks = 0
        self._truncated_queries = 0
        self._similarity_counts = [0] * (len(SIMILARITY_BUCKET_EDGES) + 1)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._ordinal_of

    @property
    def stats(self) -> IndexStats:
        with self._lock:
            return IndexStats(
                documents=len(self._keys),
                bands=self.bands,
                rows=self.rows,
                candidate_checks=self._candidate_checks,
                truncated_queries=self._truncated_queries,
                similarity_histogram=tuple(self._similarity_counts),
            )

    # ------------------------------------------------------------------ #
    # Pure feature / signature computation (no shared state, no lock).    #
    # ------------------------------------------------------------------ #

    def features(self, text: str) -> frozenset[int]:
        """Shingle-hash set of ``text``, or ``{fingerprint}`` if it is shorter than ``k``."""
        try:
            return self.shingler.hash_set(text)
        except ShingleError:
            return frozenset({self.shingler.fingerprint(text)})

    def signature(self, features: Set[int]) -> Signature:
        """MinHash signature of a non-empty feature set (order-independent)."""
        if not features:
            raise ValueError("cannot compute a MinHash signature of an empty feature set")
        values = np.fromiter(features, dtype=np.uint64, count=len(features))
        values %= np.uint64(MERSENNE_PRIME_31)
        signature = np.full(self.config.num_perm, MERSENNE_PRIME_31, dtype=np.uint64)
        a, b = self._a[:, None], self._b[:, None]
        for start in range(0, values.size, self._block):
            block = values[None, start : start + self._block]
            hashed = (a * block + b) % np.uint64(MERSENNE_PRIME_31)
            np.minimum(signature, hashed.min(axis=1), out=signature)
        return signature

    def _band_keys(self, signature: Signature) -> list[int]:
        rows = signature.astype("<u8", copy=False).reshape(self.bands, self.rows)
        return [
            int.from_bytes(
                hashlib.blake2b(row.tobytes(), digest_size=8, person=_BAND_PERSON).digest(),
                "little",
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # Queries and mutations.                                              #
    # ------------------------------------------------------------------ #

    def query(self, text: str) -> str | None:
        """Key of the earliest-inserted indexed near-duplicate of ``text``, if any."""
        features = self.features(text)
        band_keys = self._band_keys(self.signature(features))
        with self._lock:
            return self._find_duplicate_locked(features, band_keys)

    def add_if_unique(self, key: str, text: str) -> str | None:
        """Atomically insert ``text`` unless it duplicates an indexed document.

        Returns ``None`` if inserted. Otherwise returns the key it duplicates:
        ``key`` itself if that key is already indexed, or the earliest-inserted
        near-duplicate with exact Jaccard ``>= threshold``.
        """
        features = self.features(text)
        band_keys = self._band_keys(self.signature(features))
        with self._lock:
            if key in self._ordinal_of:
                return key
            duplicate = self._find_duplicate_locked(features, band_keys)
            if duplicate is None:
                self._insert_locked(key, features, band_keys)
            return duplicate

    def deduplicate_batch(
        self, samples: Sequence[InstructionSample]
    ) -> tuple[list[InstructionSample], CurationMetrics]:
        """Keep the first of every near-duplicate group, in :func:`canonical_order`.

        Processing in canonical order makes the result independent of input
        order. Samples are also checked against everything already in the index,
        so successive batches deduplicate as one stream.
        """
        kept = [
            sample
            for sample in sorted(samples, key=canonical_order)
            if self.add_if_unique(sample.sample_id, self._text_of(sample)) is None
        ]
        metrics = CurationMetrics.from_counts(
            raw_count=len(samples),
            schema_valid_count=len(samples),
            deduped_count=len(kept),
        )
        return kept, metrics

    def _candidates_locked(self, band_keys: list[int]) -> Iterator[int]:
        buckets = [
            bucket
            for table, band_key in zip(self._buckets, band_keys, strict=True)
            if (bucket := table.get(band_key)) is not None
        ]
        previous: int | None = None
        for ordinal in heapq.merge(*buckets):
            if ordinal != previous:
                previous = ordinal
                yield ordinal

    def _find_duplicate_locked(self, features: frozenset[int], band_keys: list[int]) -> str | None:
        threshold = self.config.threshold
        checks = 0
        for ordinal in self._candidates_locked(band_keys):
            if checks == self.max_candidates:
                self._truncated_queries += 1
                break
            checks += 1
            similarity = jaccard(features, self._features[ordinal])
            self._similarity_counts[similarity_bucket(similarity)] += 1
            if similarity >= threshold:
                self._candidate_checks += checks
                return self._keys[ordinal]
        self._candidate_checks += checks
        return None

    def _insert_locked(self, key: str, features: frozenset[int], band_keys: list[int]) -> None:
        ordinal = len(self._keys)
        self._keys.append(key)
        self._features.append(features)
        self._ordinal_of[key] = ordinal
        for table, band_key in zip(self._buckets, band_keys, strict=True):
            table.setdefault(band_key, []).append(ordinal)


def _permutation_params(num_perm: int, seed: int) -> tuple[Signature, Signature]:
    key = seed.to_bytes(8, "little")
    a = np.empty(num_perm, dtype=np.uint64)
    b = np.empty(num_perm, dtype=np.uint64)
    for i in range(num_perm):
        digest = hashlib.blake2b(
            i.to_bytes(4, "little"), digest_size=16, key=key, person=_PERMUTATION_PERSON
        ).digest()
        a[i] = 1 + int.from_bytes(digest[:8], "little") % (MERSENNE_PRIME_31 - 1)
        b[i] = int.from_bytes(digest[8:], "little") % MERSENNE_PRIME_31
    return a, b
