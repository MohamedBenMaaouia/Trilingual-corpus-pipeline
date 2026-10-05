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


def by_bucket(bands: DataFrame) -> DataFrame:
    """The band rows, shuffled once so that every bucket sits whole in one partition
    (Sprint 5, DECISIONS S5-04). Persist the result: then the bucket-size count and the
    self-join both find their rows already grouped by (band_id, band_hash) and shuffle
    nothing more. Without it, the self-join shuffled the band rows once per side (AQE
    on, Spark 3.5 does not reuse that exchange: 2 x 264.6 MB on the full crawl)."""
    return bands.repartition("band_id", "band_hash")


def candidate_pairs(bands: DataFrame) -> DataFrame:
    """(doc_a, doc_b) with doc_a < doc_b, each pair once: every two pages that share a
    bucket. The band self-join (plan 4.3.2), the pipeline's dominant shuffle. A bucket
    of n pages yields n(n-1)/2 pairs: this is where skew lives (Sprint 5). Fed with
    by_bucket's persisted rows, it shuffles only its pairs (for the distinct)."""
    left = bands.select("band_id", "band_hash", F.col("doc_id").alias("doc_a"))
    right = bands.select("band_id", "band_hash", F.col("doc_id").alias("doc_b"))
    return (
        left.join(right, ["band_id", "band_hash"])
        .where(F.col("doc_a") < F.col("doc_b"))  # drops self-pairs and mirror pairs
        .select("doc_a", "doc_b")
        .distinct()  # a pair sharing several buckets is one pair (plan 4.3.3)
    )


@dataclass(frozen=True)
class SaltParams:
    """Salting the band self-join's hot buckets (plan 5.4.2, DECISIONS S5-02).

    Chosen by the user from the bucket-size count of the full crawl (S5-01): buckets of
    50+ pages are 89 of 5,343,227 but make 76.3% of the pairs; splitting them at 50
    pages per piece copies 19,421 extra rows (+0.4%).
    """

    # Off by default: measured, salting bought no time on this crawl and would undo
    # by_bucket's grouping (the salt is part of the join key) (S5-03, S5-05).
    enabled: bool = False
    hot_rows: int = 50  # a bucket with at least this many pages is salted
    rows_per_salt: int = 50  # pages per piece: k = ceil(pages / rows_per_salt)
    max_salts: int = 16  # cap on k (never reached on CC-MAIN-2026-39: 290 pages -> 6)


def hot_buckets(sizes: DataFrame, salt: SaltParams) -> DataFrame:
    """(band_id, band_hash, salts): the buckets to split, and into how many pieces.
    `sizes` holds (band_id, band_hash, rows), the pages per bucket. A few rows."""
    pieces = F.least(F.lit(salt.max_salts), F.ceil(F.col("rows") / salt.rows_per_salt))
    return sizes.where(F.col("rows") >= salt.hot_rows).select(
        "band_id", "band_hash", pieces.cast("int").alias("salts")
    )


def salted_candidate_pairs(bands: DataFrame, hot: DataFrame) -> DataFrame:
    """candidate_pairs, with each hot bucket's work split over k tasks (plan 5.4.2).

    One task joins one bucket: a bucket of n pages costs it n x n comparisons. Here
    the join key gets a third part, the salt. Left side: each page's salt is
    pmod(xxhash64(doc_id), k), one of 0..k-1. Right side: each page is copied k times,
    once per salt. A hot bucket becomes k sub-buckets of about n/k x n rows, hashed to
    different tasks, and every pair still meets exactly once (under the salt of its
    left page). Buckets that are not hot get k = 1: salt 0, no copy.

    The salt is a hash of doc_id, never rand(): a retried task must draw the same salts,
    or left and right stop lining up and pairs are lost or doubled (trap S5).
    `hot` comes from hot_buckets (a few rows: broadcast, no shuffle).
    """
    tagged = bands.join(F.broadcast(hot), ["band_id", "band_hash"], "left").withColumn(
        "salts", F.coalesce(F.col("salts"), F.lit(1))
    )
    left = tagged.select(
        "band_id",
        "band_hash",
        F.col("doc_id").alias("doc_a"),
        F.pmod(F.xxhash64("doc_id"), F.col("salts")).cast("int").alias("salt"),
    )
    right = tagged.select(
        "band_id",
        "band_hash",
        F.col("doc_id").alias("doc_b"),
        F.explode(F.sequence(F.lit(0), F.col("salts") - 1)).alias("salt"),
    )
    return (
        left.join(right, ["band_id", "band_hash", "salt"])
        .where(F.col("doc_a") < F.col("doc_b"))
        .select("doc_a", "doc_b")
        .distinct()
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
