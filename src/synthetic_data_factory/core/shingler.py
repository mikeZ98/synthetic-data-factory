"""Deterministic text normalization, k-shingling, and 64-bit shingle hashing.

Determinism contract
--------------------
For a fixed ``(k, mode, seed)`` the hash of every shingle is a pure function
of the input text. It does not depend on ``PYTHONHASHSEED``, process, platform
or byte order. Hashes are keyed BLAKE2b with an 8-byte digest. The seed is the
key, and the shingling mode is the BLAKE2b personalization, so char and word
shingles that happen to spell the same string still hash differently.

Streaming
---------
Normalization has to materialize the normalized string (NFC needs the whole
text). Shingling and hashing after that are lazy. No list of shingles is ever
built. Auxiliary memory is O(k): the current char window, or a ``deque`` of
the last ``k`` word tokens.
"""

import hashlib
import re
import unicodedata
from collections import deque
from collections.abc import Iterator
from typing import Final, Literal

ShingleMode = Literal["char", "word"]

UINT64_MAX: Final = 2**64 - 1

_WORD_RE: Final = re.compile(r"[^ ]+")
_PERSON: Final[dict[str, bytes]] = {"char": b"sdf/shingle/char", "word": b"sdf/shingle/word"}
_FINGERPRINT_PERSON: Final = b"sdf/fingerprint"


class ShingleError(ValueError):
    """Raised when a text has fewer than ``k`` units (chars or words) after normalization."""


def normalize_text(text: str) -> str:
    """Canonical form: NFC, lowercased, whitespace runs collapsed to one space, trimmed.

    NFC is applied again after lowercasing because case mapping can emit
    sequences that are not NFC. The result is idempotent:
    ``normalize_text(normalize_text(t)) == normalize_text(t)``.
    """
    lowered = unicodedata.normalize("NFC", unicodedata.normalize("NFC", text).lower())
    return " ".join(lowered.split())


class TextShingler:
    """Produces ordered k-shingles of normalized text and their 64-bit hashes."""

    __slots__ = ("_base_hasher", "_fingerprint_hasher", "k", "mode", "seed")

    def __init__(self, k: int, *, mode: ShingleMode = "word", seed: int = 0) -> None:
        if isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if mode not in _PERSON:
            raise ValueError(f"mode must be 'char' or 'word', got {mode!r}")
        if isinstance(seed, bool) or not 0 <= seed <= UINT64_MAX:
            raise ValueError(f"seed must be an unsigned 64-bit integer, got {seed!r}")
        self.k: Final = k
        self.mode: Final = mode
        self.seed: Final = seed
        key = seed.to_bytes(8, "little")
        self._base_hasher: Final = hashlib.blake2b(digest_size=8, key=key, person=_PERSON[mode])
        self._fingerprint_hasher: Final = hashlib.blake2b(
            digest_size=8, key=key, person=_FINGERPRINT_PERSON
        )

    def __repr__(self) -> str:
        return f"TextShingler(k={self.k}, mode={self.mode!r}, seed={self.seed})"

    def shingles(self, text: str) -> Iterator[str]:
        """Yield the k-shingles of ``text`` in document order.

        Validation is eager: :class:`ShingleError` is raised by this call, not
        on first iteration. Word shingles are tokens joined by a single space,
        which is unambiguous because normalized tokens contain no spaces.
        """
        normalized = normalize_text(text)
        if self.mode == "char":
            units = len(normalized)
            self._require_units(units)
            return self._char_shingles(normalized)
        units = normalized.count(" ") + 1 if normalized else 0
        self._require_units(units)
        return self._word_shingles(normalized)

    def hashes(self, text: str) -> Iterator[int]:
        """Yield one unsigned 64-bit hash per shingle, in document order (eagerly validated)."""
        return self._hash_stream(self.shingles(text))

    def hash_set(self, text: str) -> frozenset[int]:
        """The set of shingle hashes: the feature set used for MinHash and exact Jaccard."""
        return frozenset(self.hashes(text))

    def fingerprint(self, text: str) -> int:
        """64-bit hash of the whole normalized text, in its own hash domain.

        Unlike :meth:`hashes`, this is defined for texts shorter than ``k``.
        """
        hasher = self._fingerprint_hasher.copy()
        hasher.update(normalize_text(text).encode("utf-8"))
        return int.from_bytes(hasher.digest(), "little")

    def _require_units(self, units: int) -> None:
        if units < self.k:
            raise ShingleError(
                f"text has {units} {self.mode} unit(s) after normalization; "
                f"at least k={self.k} required"
            )

    def _char_shingles(self, normalized: str) -> Iterator[str]:
        k = self.k
        for start in range(len(normalized) - k + 1):
            yield normalized[start : start + k]

    def _word_shingles(self, normalized: str) -> Iterator[str]:
        window: deque[str] = deque(maxlen=self.k)
        for match in _WORD_RE.finditer(normalized):
            window.append(match.group())
            if len(window) == self.k:
                yield " ".join(window)

    def _hash_stream(self, shingles: Iterator[str]) -> Iterator[int]:
        base = self._base_hasher
        for shingle in shingles:
            hasher = base.copy()
            hasher.update(shingle.encode("utf-8"))
            yield int.from_bytes(hasher.digest(), "little")
