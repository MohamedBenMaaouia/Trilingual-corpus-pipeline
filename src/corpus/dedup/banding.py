"""Stage 3d: LSH banding, the band self-join, candidate pairs (Story 4.3, DECISIONS S4-04).

The signature's num_perm values are cut into `bands` bands of `rows` values. Two pages
whose values agree on a whole band land in the same bucket (band_id, band_hash) and
become a candidate pair. For two pages of Jaccard similarity s (spec 3d):

    P(candidate) = 1 - (1 - s^rows)^bands

an S-curve, steepest around t = (1/bands)^(1/rows): 16 x 8 gives t ~ 0.71, 8 x 16 ~ 0.88.

Candidates are then checked on their whole signatures (verified_pairs): a pair is kept
only if the signatures agree on at least t of their values. LSH alone also lets through
pairs below t (the S-curve's lower tail), and one such pair can chain two unrelated
groups of pages into one cluster (trap S4).
"""

from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class LshParams:
    """bands x rows must equal the signature length (MinHashParams.num_perm)."""

    bands: int = 16  # chosen in S4-07 from the 16 x 8 vs 8 x 16 comparison
    rows: int = 8

    @property
    def threshold(self) -> float:
        """Where the S-curve is steepest; also the similarity a candidate must reach."""
        return float((1 / self.bands) ** (1 / self.rows))


def candidate_probability(similarity: float, bands: int, rows: int) -> float:
    """P(two pages of this Jaccard similarity share at least one bucket)."""
    return float(1 - (1 - similarity**rows) ** bands)


def band_rows(signatures: DataFrame, lsh: LshParams) -> DataFrame:
    """(doc_id, band_id, band_hash): one row per page and band. band_hash is Spark's
    xxhash64 of the band's `rows` values, a signed 64-bit long. Pages without a
    signature (no shingle) get no rows."""
    hashes = F.array(
        *[
            F.xxhash64(F.slice("signature", band * lsh.rows + 1, lsh.rows))
            for band in range(lsh.bands)
        ]
    )
    return signatures.where(F.col("signature").isNotNull()).select(
        "doc_id", F.posexplode(hashes).alias("band_id", "band_hash")
    )


def candidate_pairs(bands: DataFrame) -> DataFrame:
    """(doc_a, doc_b) with doc_a < doc_b, each pair once: every two pages that share a
    bucket. The band self-join (plan 4.3.2), the pipeline's dominant shuffle. A bucket
    of n pages yields n(n-1)/2 pairs: this is where skew lives (Sprint 5)."""
    left = bands.select("band_id", "band_hash", F.col("doc_id").alias("doc_a"))
    right = bands.select("band_id", "band_hash", F.col("doc_id").alias("doc_b"))
    return (
        left.join(right, ["band_id", "band_hash"])
        .where(F.col("doc_a") < F.col("doc_b"))  # drops self-pairs and mirror pairs
        .select("doc_a", "doc_b")
        .distinct()  # a pair sharing several buckets is one pair (plan 4.3.3)
    )


def matching_share(left: Column, right: Column, num_perm: int) -> Column:
    """The share of equal values of two signatures (the estimated Jaccard similarity),
    with Spark built-ins: no Python."""
    equal = F.zip_with(left, right, lambda x, y: F.when(x == y, 1).otherwise(0))
    return F.aggregate(equal, F.lit(0), lambda total, value: total + value) / F.lit(num_perm)


def verified_pairs(
    pairs: DataFrame, signatures: DataFrame, num_perm: int, min_similarity: float
) -> DataFrame:
    """(doc_a, doc_b, similarity) for the candidates whose signatures agree on at least
    `min_similarity` of their values."""
    a = signatures.select(F.col("doc_id").alias("doc_a"), F.col("signature").alias("sig_a"))
    b = signatures.select(F.col("doc_id").alias("doc_b"), F.col("signature").alias("sig_b"))
    scored = (
        pairs.join(a, "doc_a")
        .join(b, "doc_b")
        .select(
            "doc_a",
            "doc_b",
            matching_share(F.col("sig_a"), F.col("sig_b"), num_perm).alias("similarity"),
        )
    )
    return scored.where(F.col("similarity") >= min_similarity)
