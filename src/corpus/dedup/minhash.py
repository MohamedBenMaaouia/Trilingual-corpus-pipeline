"""Stage 3b-3c: word shingles and MinHash signatures (Story 4.2, DECISIONS S4-03).

Written by hand rather than with Spark ML's MinHashLSH (plan 4.2.2): Sprint 5 needs
control over the bands and the join, and every step must be explainable.

- Words: the text lowercased and folded by the Arabic matching key (D6), then cut into
  runs of letters and digits, so case, punctuation and spacing never count.
- Shingles: every run of `shingle_size` consecutive words, as a set (spec 3b). A page
  with fewer words has none: it skips near-dup detection and passes through.
- Each shingle becomes a 32-bit number (xxh32, seeded).
- num_perm hash functions h_i(x) = (a_i * x + b_i) mod p, with p = 2^32 - 5, the largest
  prime below 2^32. Value i of the signature is the smallest h_i over the page's
  shingles. Two pages have the same value i with probability equal to the Jaccard
  similarity of their shingle sets, so the share of equal values estimates it.
- No overflow: x, a_i and b_i are below 2^32, so a_i * x + b_i < 2^64 fits numpy's uint64
  exactly (trap S4).
- The coefficients come from xxh64 of their index and a fixed seed: the same numbers on
  every machine and library version (numpy's random generators may change across
  versions), so a signature computed today matches one computed on Databricks.
"""

import re
from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt
import pandas as pd
import xxhash
from pyspark.sql import DataFrame
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import ArrayType, LongType

from corpus.silver.normalize import arabic_match_key

PRIME = 4_294_967_291  # 2^32 - 5 (trap S4)

# Runs of letters, digits or "_" in any script. arabic_match_key runs first: it deletes
# the vowel marks, which are not word characters and would otherwise cut Arabic words.
_WORD = re.compile(r"\w+")

# Shingles handled per numpy step: num_perm x 4,096 x 8 bytes = 4 MiB at 128
# permutations, whatever the page's length (pages go up to 100,000 words, S3-08).
_CHUNK = 4096

Signature = npt.NDArray[np.uint64]


@dataclass(frozen=True)
class MinHashParams:
    """Changing any of these changes every signature: it is a pipeline change (D8)."""

    shingle_size: int = 5  # word 5-grams (spec 3b, plan 4.2.1)
    num_perm: int = 128  # signature length (spec 3c, plan 4.2.2)
    seed: int = 42  # shingle hashes and coefficients; fixed, never random (trap S4)


def words(text: str) -> list[str]:
    """The page's words as dedup compares them."""
    return _WORD.findall(arabic_match_key(text.lower()))


def shingle_hashes(text: str, size: int, seed: int) -> npt.NDArray[np.uint64]:
    """The page's distinct shingles as 32-bit numbers (sorted); empty if it has fewer
    than `size` words. A set, as the Jaccard similarity is defined on sets."""
    tokens = words(text)
    count = len(tokens) - size + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    hashes = np.fromiter(
        (
            xxhash.xxh32_intdigest(" ".join(tokens[i : i + size]).encode("utf-8"), seed)
            for i in range(count)
        ),
        dtype=np.uint64,
        count=count,
    )
    return np.unique(hashes)


@cache
def coefficients(num_perm: int, seed: int) -> tuple[Signature, Signature]:
    """(a, b) of the num_perm hash functions: a_i in 1..p-1, b_i in 0..p-1."""
    a = [1 + xxhash.xxh64_intdigest(f"a{i}".encode(), seed) % (PRIME - 1) for i in range(num_perm)]
    b = [xxhash.xxh64_intdigest(f"b{i}".encode(), seed) % PRIME for i in range(num_perm)]
    return np.array(a, dtype=np.uint64), np.array(b, dtype=np.uint64)


def signature(text: str, params: MinHashParams) -> Signature | None:
    """The page's num_perm minimums; None when the page has no shingle."""
    hashes = shingle_hashes(text, params.shingle_size, params.seed)
    if hashes.size == 0:
        return None
    a, b = coefficients(params.num_perm, params.seed)
    result = np.full(params.num_perm, PRIME, dtype=np.uint64)  # above any h_i(x), all < p
    for start in range(0, hashes.size, _CHUNK):
        chunk = hashes[start : start + _CHUNK]
        values = (np.outer(a, chunk) + b[:, np.newaxis]) % np.uint64(PRIME)
        np.minimum(result, values.min(axis=1), out=result)
    return result


def similarity(left: Signature, right: Signature) -> float:
    """The share of equal values: the MinHash estimate of the Jaccard similarity."""
    return float(np.mean(left == right))


def add_signatures(docs: DataFrame, params: MinHashParams) -> DataFrame:
    """Add `signature`: array<bigint> of num_perm values, null for a page with no shingle.

    A pandas_udf on `text`: Spark sends the text in Arrow batches, one Python call per
    batch; the other columns never leave the JVM (as language ID, S3-04). Values are
    below 2^32, so Spark's signed 64-bit longs hold them unchanged.
    """

    @pandas_udf(ArrayType(LongType()))  # type: ignore[call-overload, untyped-decorator]
    def batch(texts: pd.Series) -> pd.Series:
        values = []
        for text in texts:
            sig = signature(text or "", params)
            values.append(None if sig is None else sig.astype(np.int64))
        return pd.Series(values, dtype=object)

    return docs.withColumn("signature", batch("text"))
