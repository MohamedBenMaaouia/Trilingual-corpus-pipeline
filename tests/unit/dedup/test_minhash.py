"""dedup.minhash: words, shingles, coefficients, signatures (Story 4.2, DECISIONS S4-03).

Pure Python: no Spark. The Spark side (the pandas_udf) is checked in test_banding.py.
"""

import math

import numpy as np
import pytest
import xxhash

from corpus.dedup.minhash import (
    PRIME,
    MinHashParams,
    coefficients,
    shingle_hashes,
    signature,
    similarity,
    words,
)

PARAMS = MinHashParams()


def numbered(start: int, stop: int) -> str:
    """Distinct words w<start> ... w<stop - 1>: every shingle is different."""
    return " ".join(f"w{i}" for i in range(start, stop))


def test_words_ignore_case_punctuation_and_spacing() -> None:
    assert words("The  CAT, sat;\non the mat!") == ["the", "cat", "sat", "on", "the", "mat"]
    assert words("L'été à 10h30.") == ["l", "été", "à", "10h30"]


def test_words_fold_arabic_spellings() -> None:
    # "school" with a vowel mark and teh marbuta, and plain: one word, two spellings (D6).
    stem = "\N{ARABIC LETTER DAL}\N{ARABIC LETTER REH}\N{ARABIC LETTER SEEN}"
    vowelled = "\N{ARABIC LETTER MEEM}\N{ARABIC FATHA}" + stem + "\N{ARABIC LETTER TEH MARBUTA}"
    plain = "\N{ARABIC LETTER MEEM}" + stem + "\N{ARABIC LETTER HEH}"
    assert words(vowelled) == words(plain) == [plain]
    hamza = "\N{ARABIC LETTER ALEF WITH HAMZA ABOVE}\N{ARABIC LETTER BEH}"
    assert words(hamza) == ["\N{ARABIC LETTER ALEF}\N{ARABIC LETTER BEH}"]


def test_shingles_are_the_distinct_runs_of_five_words() -> None:
    assert shingle_hashes(numbered(0, 12), 5, 42).size == 8  # 12 - 5 + 1
    assert shingle_hashes(numbered(0, 5) + " " + numbered(0, 5), 5, 42).size == 5  # repeats once
    expected = xxhash.xxh32_intdigest(b"w0 w1 w2 w3 w4", 42)
    assert expected in shingle_hashes(numbered(0, 5), 5, 42)


def test_a_page_with_fewer_words_than_a_shingle_has_no_signature() -> None:
    assert shingle_hashes("four words only here", 5, 42).size == 0
    assert signature("four words only here", PARAMS) is None
    assert signature("", PARAMS) is None


def test_coefficients_are_fixed_and_in_range() -> None:
    a, b = coefficients(128, 42)
    assert a.dtype == b.dtype == np.uint64 and a.size == b.size == 128
    assert all(1 <= int(x) < PRIME for x in a) and all(0 <= int(x) < PRIME for x in b)
    # Pinned: a change here changes every signature, i.e. a new pipeline version (D8).
    assert [int(x) for x in a[:3]] == [1510141879, 245310547, 216702612]
    assert [int(x) for x in b[:3]] == [596278939, 3901458685, 836766127]
    other_a, _ = coefficients(128, 7)
    assert not np.array_equal(a, other_a)


def test_signature_matches_exact_integer_arithmetic() -> None:
    """numpy's uint64 must give the same minimums as Python's unbounded integers (no
    overflow, no silent conversion to float: trap S4)."""
    text = numbered(0, 300)
    sig = signature(text, PARAMS)
    assert sig is not None and sig.dtype == np.uint64 and sig.size == 128
    a, b = coefficients(128, 42)
    hashes = [int(h) for h in shingle_hashes(text, 5, 42)]
    expected = [
        min((int(ai) * h + int(bi)) % PRIME for h in hashes) for ai, bi in zip(a, b, strict=True)
    ]
    assert [int(v) for v in sig] == expected


def test_long_pages_are_processed_in_chunks_with_the_same_result() -> None:
    text = numbered(0, 9_000)  # more shingles than one numpy step (4,096)
    sig = signature(text, PARAMS)
    a, b = coefficients(128, 42)
    hashes = shingle_hashes(text, 5, 42)
    whole = ((np.outer(a, hashes) + b[:, np.newaxis]) % np.uint64(PRIME)).min(axis=1)
    assert sig is not None and np.array_equal(sig, whole)


def test_same_words_same_signature() -> None:
    first = signature("A cat, a dog. " + numbered(0, 50), PARAMS)
    second = signature("a  CAT a DOG " + numbered(0, 50), PARAMS)
    assert first is not None and second is not None and np.array_equal(first, second)


@pytest.mark.parametrize("shift", [44, 133, 267])
def test_estimate_is_close_to_the_true_jaccard(shift: int) -> None:
    """Plan 4.2.3: two pages with a known overlap. Page B is page A's 404 words moved by
    `shift`, so they share 400 - shift of their 400 shingles: Jaccard about 0.8, 0.5,
    0.2. The estimate's standard error is sqrt(J(1 - J) / 128); allow 3 of them."""
    page_a, page_b = numbered(0, 404), numbered(shift, 404 + shift)
    set_a = set(shingle_hashes(page_a, 5, 42).tolist())
    set_b = set(shingle_hashes(page_b, 5, 42).tolist())
    jaccard = len(set_a & set_b) / len(set_a | set_b)
    sig_a, sig_b = signature(page_a, PARAMS), signature(page_b, PARAMS)
    assert sig_a is not None and sig_b is not None
    tolerance = 3 * math.sqrt(jaccard * (1 - jaccard) / 128)
    assert abs(similarity(sig_a, sig_b) - jaccard) <= tolerance
