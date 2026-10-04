"""dedup.banding (Story 4.3, DECISIONS S4-04), and the signature pandas_udf (Story 4.2)."""

import numpy as np
import pytest
from pyspark.sql import DataFrame, SparkSession

from corpus.dedup.banding import (
    LshParams,
    band_rows,
    candidate_pairs,
    candidate_probability,
    verified_pairs,
)
from corpus.dedup.minhash import MinHashParams, add_signatures, signature, similarity


def test_threshold_and_s_curve() -> None:
    """Spec 3d: 16 x 8 gives 0.7071, 8 x 16 gives 0.8781; P(s) = 1 - (1 - s^r)^b."""
    assert LshParams(16, 8).threshold == pytest.approx(0.7071, abs=1e-4)
    assert LshParams(8, 16).threshold == pytest.approx(0.8781, abs=1e-4)
    assert candidate_probability(0.8, 16, 8) == pytest.approx(0.9470, abs=1e-4)
    assert candidate_probability(0.8, 8, 16) == pytest.approx(0.2042, abs=1e-4)
    assert candidate_probability(1.0, 16, 8) == 1.0 and candidate_probability(0.0, 16, 8) == 0.0


def signatures(spark: SparkSession, rows: dict[str, list[int]]) -> DataFrame:
    return spark.createDataFrame(list(rows.items()), "doc_id string, signature array<bigint>")


def test_one_row_per_band_and_equal_bands_hash_equal(spark: SparkSession) -> None:
    base = list(range(128))
    one_band_changed = base[:8] + [999] * 8 + base[16:]  # band 1 differs
    sigs = signatures(spark, {"a": base, "b": list(base), "c": one_band_changed})
    rows = band_rows(sigs, LshParams(16, 8)).collect()
    assert len(rows) == 3 * 16
    by_doc = {d: {r.band_id: r.band_hash for r in rows if r.doc_id == d} for d in "abc"}
    assert sorted(by_doc["a"]) == list(range(16))
    assert by_doc["a"] == by_doc["b"]
    differing = [band for band in range(16) if by_doc["a"][band] != by_doc["c"][band]]
    assert differing == [1]


def test_pages_without_a_signature_get_no_band(spark: SparkSession) -> None:
    sigs = spark.createDataFrame(
        [("short", None), ("long", list(range(128)))], "doc_id string, signature array<bigint>"
    )
    assert {r.doc_id for r in band_rows(sigs, LshParams()).collect()} == {"long"}


def test_candidate_pairs_are_ordered_unique_and_never_self(spark: SparkSession) -> None:
    bands = spark.createDataFrame(
        [
            (0, 10, "a"),
            (0, 10, "b"),
            (0, 10, "c"),  # a bucket of 3: 3 pairs
            (1, 20, "a"),
            (1, 20, "b"),  # a and b again in another band: still one pair
            (2, 30, "d"),  # alone in its bucket: nothing
            (3, 10, "d"),
            (3, 10, "e"),  # same hash, different band: d-e only
        ],
        "band_id int, band_hash long, doc_id string",
    )
    pairs = {(r.doc_a, r.doc_b) for r in candidate_pairs(bands).collect()}
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c"), ("d", "e")}


def test_candidates_are_checked_on_the_whole_signature(spark: SparkSession) -> None:
    base = list(range(128))
    sigs = signatures(
        spark,
        {
            "a": base,
            "b": base[:100] + [-1] * 28,  # 100 of 128 equal: 0.78
            "c": base[:64] + [-1] * 64,  # 64 of 128: 0.5
        },
    )
    pairs = spark.createDataFrame([("a", "b"), ("a", "c")], "doc_a string, doc_b string")
    kept = verified_pairs(pairs, sigs, 128, LshParams(16, 8).threshold).collect()
    assert [(r.doc_a, r.doc_b, r.similarity) for r in kept] == [("a", "b", 100 / 128)]


def test_spark_signatures_equal_python_ones(spark: SparkSession) -> None:
    """The pandas_udf gives exactly signature()'s values, and null for short pages."""
    texts = {
        "long": " ".join(f"w{i}" for i in range(60)),
        "near": " ".join(f"w{i}" for i in range(2, 62)),
        "short": "too short",
    }
    docs = spark.createDataFrame(list(texts.items()), "doc_id string, text string")
    params = MinHashParams()
    rows = {r.doc_id: r.signature for r in add_signatures(docs, params).collect()}
    assert rows["short"] is None
    for doc_id in ("long", "near"):
        expected = signature(texts[doc_id], params)
        assert expected is not None and rows[doc_id] == [int(v) for v in expected]
    spark_share = similarity(np.array(rows["long"]), np.array(rows["near"]))
    python_share = similarity(signature(texts["long"], params), signature(texts["near"], params))  # type: ignore[arg-type]
    assert spark_share == python_share
