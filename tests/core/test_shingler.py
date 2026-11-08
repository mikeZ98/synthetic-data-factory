"""TextShingler: normalization, boundary conditions, unicode, and hash invariance."""

import itertools
import json
import os
import subprocess
import sys
import unicodedata
from collections.abc import Iterator

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthetic_data_factory.core.shingler import (
    UINT64_MAX,
    ShingleError,
    TextShingler,
    normalize_text,
)

# Golden values pin the hash function itself. If one of these changes, every
# persisted signature, index, and dedup decision is invalidated. That needs a
# deliberate, versioned migration, never a silent update of this table.
GOLDEN_WORD_TEXT = "The quick brown fox jumps"
GOLDEN_WORD_K3_SEED0 = [15035376748487088215, 597933297599081844, 16114185889933408975]
GOLDEN_CHAR_K3 = {
    "seed0": [8503609430358164427, 15934431769175937824],
    "seed42": [6373193242571544939, 14494346325385532128],
}
GOLDEN_FINGERPRINT_HELLO = 10377934416881128412

words = st.from_regex(r"[a-z]{1,8}", fullmatch=True)


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [0, -1, True, False])
def test_rejects_non_positive_or_bool_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        TextShingler(k)


def test_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        TextShingler(3, mode="byte")  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, UINT64_MAX + 1, True])
def test_rejects_seed_outside_uint64(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be"):
        TextShingler(3, seed=seed)


def test_accepts_seed_bounds_and_has_stable_repr() -> None:
    assert TextShingler(3, seed=0).seed == 0
    shingler = TextShingler(2, mode="char", seed=UINT64_MAX)
    assert repr(shingler) == f"TextShingler(k=2, mode='char', seed={UINT64_MAX})"


# --------------------------------------------------------------------------- #
# Normalization                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Café", "café"),  # precomposed
        ("Café", "café"),  # decomposed -> NFC
        ("  Hello \t\n World  ", "hello world"),
        ("a b c　d e", "a b c d e"),  # NBSP, em space, ideographic, LS
        ("STRASSE Straße", "strasse straße"),  # lower(), not casefold()
        ("İstanbul", "i̇stanbul"),  # dotted capital I lowercases to i + U+0307
        ("ﬁle", "ﬁle"),  # ligature survives: NFC, not NFKC
        ("a‍b", "a‍b"),  # ZWJ is not whitespace
        ("שלום  עולם", "שלום עולם"),
        ("", ""),
        (" \t\n ", ""),
    ],
)
def test_normalize_examples(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


@given(st.text())
def test_normalize_is_idempotent_and_canonical(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once
    assert unicodedata.is_normalized("NFC", once)
    assert once == once.strip()
    assert "  " not in once
    assert all(not ch.isspace() or ch == " " for ch in once)


@given(st.text())
def test_canonically_equivalent_texts_normalize_identically(text: str) -> None:
    assert normalize_text(unicodedata.normalize("NFD", text)) == normalize_text(
        unicodedata.normalize("NFC", text)
    )


# --------------------------------------------------------------------------- #
# Shingling                                                                   #
# --------------------------------------------------------------------------- #


@given(st.text(min_size=1), st.integers(min_value=1, max_value=6))
def test_char_shingles_are_ordered_overlapping_windows(text: str, k: int) -> None:
    normalized = normalize_text(text)
    shingler = TextShingler(k, mode="char")
    if len(normalized) < k:
        with pytest.raises(ShingleError):
            shingler.shingles(text)
        return
    shingles = list(shingler.shingles(text))
    assert len(shingles) == len(normalized) - k + 1
    assert all(len(s) == k for s in shingles)
    assert shingles[0] + "".join(s[-1] for s in shingles[1:]) == normalized


@given(st.lists(words, min_size=1, max_size=20), st.integers(min_value=1, max_value=6))
def test_word_shingles_are_ordered_overlapping_windows(tokens: list[str], k: int) -> None:
    text = "  ".join(t.upper() for t in tokens)
    shingler = TextShingler(k, mode="word")
    if len(tokens) < k:
        with pytest.raises(ShingleError):
            shingler.shingles(text)
        return
    shingles = [s.split(" ") for s in shingler.shingles(text)]
    assert len(shingles) == len(tokens) - k + 1
    assert shingles == [tokens[i : i + k] for i in range(len(tokens) - k + 1)]


@pytest.mark.parametrize(
    ("mode", "text", "k"),
    [
        ("char", "abcd", 5),
        ("char", "  ab  ", 3),  # whitespace stripped before counting
        ("char", "", 1),
        ("word", "one two", 3),
        ("word", "   ", 1),
        ("word", "", 1),
    ],
)
def test_text_shorter_than_k_is_rejected_eagerly(mode: str, text: str, k: int) -> None:
    shingler = TextShingler(k, mode=mode)  # type: ignore[arg-type]
    with pytest.raises(ShingleError, match=f"at least k={k}"):
        shingler.shingles(text)  # raises on call, before any iteration
    with pytest.raises(ShingleError):
        shingler.hashes(text)


@pytest.mark.parametrize(("mode", "text"), [("char", "abc"), ("word", "one two three")])
def test_text_of_exactly_k_units_yields_one_shingle(mode: str, text: str) -> None:
    assert len(list(TextShingler(3, mode=mode).shingles(text))) == 1  # type: ignore[arg-type]


def test_hashing_is_lazy_and_streaming() -> None:
    text = " ".join(f"w{i}" for i in range(200_000))
    stream = TextShingler(3).hashes(text)
    assert isinstance(stream, Iterator)
    first = list(itertools.islice(stream, 3))
    assert len(first) == 3
    assert next(stream) != first[-1]


# --------------------------------------------------------------------------- #
# Hashing                                                                     #
# --------------------------------------------------------------------------- #


def test_word_hashes_match_golden_values() -> None:
    hashes = list(TextShingler(3, mode="word", seed=0).hashes(GOLDEN_WORD_TEXT))
    assert hashes == GOLDEN_WORD_K3_SEED0


def test_char_hashes_match_golden_values_and_depend_on_seed() -> None:
    assert list(TextShingler(3, mode="char", seed=0).hashes("abcd")) == GOLDEN_CHAR_K3["seed0"]
    assert list(TextShingler(3, mode="char", seed=42).hashes("abcd")) == GOLDEN_CHAR_K3["seed42"]


def test_fingerprint_matches_golden_value_and_ignores_k() -> None:
    assert TextShingler(3).fingerprint("Hello") == GOLDEN_FINGERPRINT_HELLO
    assert TextShingler(50, mode="char").fingerprint("  HELLO ") == GOLDEN_FINGERPRINT_HELLO


@pytest.mark.parametrize("hash_seed", ["0", "1", "4242", "random"])
def test_hashes_are_invariant_across_processes_and_pythonhashseed(hash_seed: str) -> None:
    program = (
        "import json; from synthetic_data_factory.core.shingler import TextShingler; "
        "print(json.dumps(list(TextShingler(3).hashes('The quick brown fox jumps'))))"
    )
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    completed = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", program], env=env, capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == GOLDEN_WORD_K3_SEED0


@given(st.lists(words, min_size=3, max_size=30))
def test_hashes_are_uint64_and_deterministic_across_instances(tokens: list[str]) -> None:
    text = " ".join(tokens)
    first = list(TextShingler(3, seed=7).hashes(text))
    assert first == list(TextShingler(3, seed=7).hashes(text))
    assert all(0 <= h <= UINT64_MAX for h in first)
    assert TextShingler(3, seed=7).hash_set(text) == frozenset(first)


def test_modes_are_domain_separated() -> None:
    # Both shingle to the exact string "a b c", yet must hash differently.
    assert list(TextShingler(5, mode="char").shingles("a b c")) == ["a b c"]
    assert list(TextShingler(3, mode="word").shingles("a b c")) == ["a b c"]
    char_hash = next(TextShingler(5, mode="char").hashes("a b c"))
    word_hash = next(TextShingler(3, mode="word").hashes("a b c"))
    assert char_hash != word_hash
    assert TextShingler(3).fingerprint("a b c") not in {char_hash, word_hash}


def test_normalization_equivalent_texts_share_hashes() -> None:
    shingler = TextShingler(2)
    assert shingler.hash_set("Café au LAIT") == shingler.hash_set("café   au\tlait")


def test_seed_bounds_are_the_literal_uint64_range() -> None:
    # Literal values, not the module constant: a drifted constant must not pass.
    assert TextShingler(2, seed=18_446_744_073_709_551_615).seed == 2**64 - 1
    with pytest.raises(ValueError, match="seed must be"):
        TextShingler(2, seed=18_446_744_073_709_551_616)
