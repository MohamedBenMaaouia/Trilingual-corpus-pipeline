"""corpus.jobs.run_dedup: the whole Sprint 4 chain on a handful of pages (S4-02 to S4-06)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession

from corpus.jobs.run_dedup import DedupParams, run_dedup
from corpus.schemas.contract import assert_schema
from corpus.schemas.dedup_v1 import DEDUP_V1
from corpus.version import PIPELINE_VERSION

DAY = [datetime(2026, 9, d, tzinfo=UTC) for d in range(1, 5)]
BASE = " ".join(f"w{i}" for i in range(200))
# Two words changed, far apart: 10 of 196 shingles differ, Jaccard about 0.90.
NEAR = BASE.replace("w50 ", "x50 ").replace("w150 ", "x150 ")
OTHER = " ".join(f"v{i}" for i in range(200))
ARABIC_STEM = "\N{ARABIC LETTER KAF}\N{ARABIC LETTER TEH}\N{ARABIC LETTER BEH}"
ARABIC = " ".join(f"{ARABIC_STEM} {i}" for i in range(10))
ARABIC_VOWELLED = ARABIC.replace(
    ARABIC_STEM, "\N{ARABIC LETTER KAF}\N{ARABIC FATHA}" + ARABIC_STEM[1:]
)


def kept_pages(spark: SparkSession) -> DataFrame:
    """The silver_v1 columns the job reads, for seven kept pages."""
    rows = [
        ("a1", "en", 0.9, DAY[1], BASE),
        ("a2", "en", 0.9, DAY[0], BASE),  # same text, fetched earlier: a1 is its exact copy
        ("a3", "en", 0.95, DAY[2], NEAR),  # near duplicate of a2, better score: kept
        ("b1", "en", 0.5, DAY[0], OTHER),  # unrelated
        ("c1", "ar", 0.7, DAY[1], ARABIC_VOWELLED),  # same text once vowel marks fold
        ("c2", "ar", 0.7, DAY[0], ARABIC),
        ("d1", "fr", 0.3, DAY[0], "trop court"),  # no shingle: passes through
    ]
    return spark.createDataFrame(
        [
            (d, lang, score, day, text, f"http://site-{d}.example/page", f"site-{d}.example")
            for d, lang, score, day, text in rows
        ],
        "doc_id string, language string, quality_score double, fetch_date timestamp, "
        "text string, url string, domain string",
    )


def test_the_chain_finds_exact_and_near_duplicates(spark: SparkSession, tmp_path: Path) -> None:
    output = str(tmp_path / "dedup" / "crawl_id=CC-MAIN-2026-39")
    metrics = run_dedup(spark, kept_pages(spark), output, DedupParams())

    written = spark.read.schema(DEDUP_V1).parquet(output)
    assert_schema(written, DEDUP_V1, check_nullability=False)
    rows = {r.doc_id: (r.cluster_id, r.cluster_size, r.duplicate_type) for r in written.collect()}
    assert rows == {
        "a1": ("a3", 3, "exact"),
        "a2": ("a3", 3, "near"),
        "a3": ("a3", 3, None),
        "b1": ("b1", 1, None),
        "c1": ("c2", 2, "exact"),
        "c2": ("c2", 2, None),
        "d1": ("d1", 1, None),
    }
    assert {r.pipeline_version for r in written.collect()} == {PIPELINE_VERSION}

    assert metrics["dedup_documents_in"] == 7
    assert metrics["dedup_documents_kept"] == 4
    assert (metrics["dedup_exact_duplicates"], metrics["dedup_near_duplicates"]) == (2, 1)
    assert metrics["dedup_exact_duplicates_ar"] == 1 and metrics["dedup_near_duplicates_en"] == 1
    assert metrics["dedup_dup_rate"] == round(3 / 7, 4)
    assert metrics["dedup_clusters"] == 2 and metrics["dedup_cluster_max_size"] == 3
    # 5 exact representatives (a2, a3, b1, c2, d1); d1 has no shingle: 4 signed pages.
    assert metrics["dedup_documents_signed"] == 4
    assert metrics["dedup_band_rows"] == 4 * 16
    assert metrics["dedup_candidate_pairs"] >= 1 and metrics["dedup_verified_pairs"] == 1
    assert metrics["dedup_components_converged"] == 1
    assert metrics["dedup_documents_written"] == 7
    assert metrics["dedup_step_pairs_seconds"] >= 0  # every step is timed


def test_a_rerun_replaces_the_crawl(spark: SparkSession, tmp_path: Path) -> None:
    """Crawl-scoped and retry-safe (invariants 6-7): the second run's rows replace the first's."""
    output = str(tmp_path / "dedup" / "crawl_id=CC-MAIN-2026-39")
    other_crawl = tmp_path / "dedup" / "crawl_id=CC-MAIN-2026-40"
    pages = kept_pages(spark)
    run_dedup(spark, pages, str(other_crawl), DedupParams())
    run_dedup(spark, pages, output, DedupParams())
    run_dedup(spark, pages.where("doc_id != 'b1'"), output, DedupParams())
    assert spark.read.parquet(output).count() == 6
    assert spark.read.parquet(str(other_crawl)).count() == 7


def test_bands_must_cover_the_signature(spark: SparkSession, tmp_path: Path) -> None:
    from corpus.dedup.banding import LshParams

    with pytest.raises(ValueError, match="bands"):
        run_dedup(spark, kept_pages(spark), str(tmp_path / "x"), DedupParams(lsh=LshParams(10, 8)))
