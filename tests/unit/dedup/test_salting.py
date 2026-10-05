"""The band self-join, tuned (Story 5.4, DECISIONS S5-02, S5-04): hot buckets split into
pieces with exactly the same pairs as the plain join; band rows grouped by bucket once."""

from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.dedup.banding import (
    SaltParams,
    by_bucket,
    candidate_pairs,
    hot_buckets,
    salted_candidate_pairs,
)


def skewed_bands(spark: SparkSession) -> DataFrame:
    """A hot bucket of 40 pages (780 pairs), small ones, and a pair found in two buckets."""
    rows = [(0, 7, f"d{i:02d}") for i in range(40)]
    rows += [(1, 8, "d00"), (1, 8, "x1"), (2, 9, "x2"), (3, 7, "d05"), (3, 7, "d06")]
    return spark.createDataFrame(rows, "band_id int, band_hash long, doc_id string")


def sizes_of(bands: DataFrame) -> DataFrame:
    return bands.groupBy("band_id", "band_hash").agg(F.count("*").alias("rows"))


def test_hot_buckets_and_their_pieces(spark: SparkSession) -> None:
    sizes = spark.createDataFrame(
        [(0, 1, 290), (0, 2, 108), (0, 3, 50), (0, 4, 49), (0, 5, 2000)],
        "band_id int, band_hash long, rows long",
    )
    hot = {r.band_hash: r.salts for r in hot_buckets(sizes, SaltParams()).collect()}
    # At least 50 pages; ceil(pages / 50) pieces, at most 16.
    assert hot == {1: 6, 2: 3, 3: 1, 5: 16}


def test_salting_gives_exactly_the_same_pairs(spark: SparkSession) -> None:
    bands = skewed_bands(spark)
    plain = {(r.doc_a, r.doc_b) for r in candidate_pairs(bands).collect()}
    assert len(plain) == 780 + 1  # the hot bucket's pairs + (d00, x1); (d05, d06) once
    salt = SaltParams(hot_rows=10, rows_per_salt=5, max_salts=16)  # 40 pages: 8 pieces
    salted = salted_candidate_pairs(bands, hot_buckets(sizes_of(bands), salt)).collect()
    assert len(salted) == len(plain)  # no pair doubled
    assert {(r.doc_a, r.doc_b) for r in salted} == plain  # none lost


def test_salts_are_spread_and_repeatable(spark: SparkSession) -> None:
    """pmod(xxhash64(doc_id), k): a hash of the id, the same on every run and retry."""
    docs = spark.createDataFrame([(f"d{i:02d}",) for i in range(40)], "doc_id string")
    salts = docs.select("doc_id", F.pmod(F.xxhash64("doc_id"), F.lit(8)).alias("salt"))
    first = {r.doc_id: r.salt for r in salts.collect()}
    assert first == {r.doc_id: r.salt for r in salts.collect()}
    assert len(set(first.values())) >= 6  # 40 pages over 8 pieces: no single piece


def shuffles_run(df: DataFrame) -> int:
    """Run the query, then count the shuffles that really ran: the shuffle stages of
    AQE's final plan, reused ones left out. (The plan's text is no evidence: AQE prints
    its initial plan too, and a persisted input prints the plan that built it.)"""
    df.collect()

    def count(node: Any) -> int:
        found = 0
        if node.nodeName() == "ShuffleQueryStage":
            node = node.plan()  # the exchange this stage ran
            found = 0 if node.nodeName() == "ReusedExchange" else 1
        children = node.children()
        return found + sum(count(children.apply(i)) for i in range(children.size()))

    return count(df._jdf.queryExecution().executedPlan().executedPlan())


def test_grouping_by_bucket_removes_the_self_joins_shuffles(spark: SparkSession) -> None:
    """S5-04: the band rows grouped by bucket once (by_bucket, persisted) feed the
    self-join and the bucket count without any further shuffle. Broadcast off: on tiny
    test data Spark would otherwise skip the shuffles altogether."""
    key = "spark.sql.autoBroadcastJoinThreshold"
    previous = str(spark.conf.get(key))
    spark.conf.set(key, "-1")
    try:
        plain = skewed_bands(spark).persist(StorageLevel.DISK_ONLY)
        grouped = by_bucket(skewed_bands(spark)).persist(StorageLevel.DISK_ONLY)
        # Plain: each side of the join shuffles the band rows, then the distinct.
        assert shuffles_run(candidate_pairs(plain)) == 3
        # Grouped: only the distinct shuffles (its pairs), and the count nothing.
        assert shuffles_run(candidate_pairs(grouped)) == 1
        assert shuffles_run(grouped.groupBy("band_id", "band_hash").count()) == 0
        same = {tuple(r) for r in candidate_pairs(grouped).collect()}
        assert same == {tuple(r) for r in candidate_pairs(plain).collect()}
    finally:
        spark.conf.set(key, previous)
        plain.unpersist()
        grouped.unpersist()
