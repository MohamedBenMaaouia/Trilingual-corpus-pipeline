"""corpus.jobs.run_gold: the whole Sprint 6 chain on a handful of pages, on local files."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession

from corpus.jobs.run_gold import run_gold
from corpus.schemas.contract import assert_schema
from corpus.schemas.gold_v1 import GOLD_V1

Batch = Callable[[list[dict[str, Any]]], DataFrame]
CRAWL, OTHER_CRAWL = "CC-MAIN-2026-39", "CC-MAIN-2026-40"
EXCLUDED = "excluded-site.com"


@pytest.fixture
def silver(silver_v1_batch: Batch) -> DataFrame:
    return silver_v1_batch(
        [
            {"doc_id": "a1"},  # kept, stands for a2 and a3
            {"doc_id": "a2"},  # its exact duplicate
            {"doc_id": "a3", "quality_tier": "medium"},  # its near duplicate
            {"doc_id": "b1", "language": "fr"},
            {"doc_id": "c1", "language": "ar", "quality_tier": "medium"},
            {"doc_id": "x1", "url": f"https://www.{EXCLUDED}/p", "domain": EXCLUDED},
            {"doc_id": "r1", "quality_tier": "rejected", "reject_reasons": ["too_few_words"]},
            {"doc_id": "z1", "crawl_id": OTHER_CRAWL},  # another crawl: not this run's
        ]
    )


@pytest.fixture
def dedup(dedup_v1_batch: Batch) -> DataFrame:
    """One row per kept page of the crawl, as dedup_v1 has (S4-06)."""
    return dedup_v1_batch(
        [
            {"doc_id": "a1", "cluster_size": 3},
            {"doc_id": "a2", "cluster_id": "a1", "cluster_size": 3, "duplicate_type": "exact"},
            {"doc_id": "a3", "cluster_id": "a1", "cluster_size": 3, "duplicate_type": "near"},
            {"doc_id": "b1", "language": "fr"},
            {"doc_id": "c1", "language": "ar"},
            {"doc_id": "x1"},
        ]
    )


def written(spark: SparkSession, table: str) -> dict[str, int]:
    rows = spark.read.format("delta").load(table)
    assert_schema(rows, GOLD_V1)  # nullability included: the table keeps it
    return {r.doc_id: r.cluster_size for r in rows.collect()}


def test_the_chain_selects_writes_compacts_describes_and_reruns(
    spark: SparkSession, tmp_path: Path, silver: DataFrame, dedup: DataFrame
) -> None:
    table = str(tmp_path / "gold_documents")
    metrics, stats = run_gold(spark, silver, dedup, table, CRAWL, (EXCLUDED,))

    assert written(spark, table) == {"a1": 3, "b1": 1, "c1": 1}
    assert metrics["gold_silver_pages"] == 6 and metrics["gold_dedup_rows"] == 6
    assert metrics["gold_exact_duplicates_removed"] == 1
    assert metrics["gold_near_duplicates_removed"] == 1
    assert metrics["gold_documents_deduplicated"] == 4
    assert metrics["gold_documents_excluded"] == 1 and metrics["gold_documents_excluded_en"] == 1
    assert metrics["gold_documents_selected"] == 3 == metrics["gold_documents_written"]
    assert metrics["gold_documents_written_ar"] == 1
    assert metrics["gold_exclusion_sites"] == 1
    assert metrics["gold_files_after_optimize"] <= metrics["gold_files_before_optimize"]
    assert metrics["gold_partitions_after_optimize"] == 3
    assert metrics["gold_vacuum_retention_days"] == 7
    assert metrics["gold_step_write_seconds"] >= 0  # every step is timed
    assert [(s["language"], s["quality_tier"], s["documents"]) for s in stats] == [
        ("ar", "medium", 1),
        ("en", "high", 1),
        ("fr", "high", 1),
    ]

    # Retry-safe (invariant 7): the same input again leaves the same rows, once.
    again, _ = run_gold(spark, silver, dedup, table, CRAWL, (EXCLUDED,))
    assert written(spark, table) == {"a1": 3, "b1": 1, "c1": 1}
    assert again["gold_delta_version_written"] > metrics["gold_delta_version_optimized"]


def test_a_page_without_a_dedup_row_stops_the_write(
    spark: SparkSession, tmp_path: Path, silver: DataFrame, dedup: DataFrame
) -> None:
    """dedup ran on another version of silver: gold refuses, nothing is created."""
    table = str(tmp_path / "gold_documents")
    with pytest.raises(RuntimeError, match="1 kept pages have no dedup row"):
        run_gold(spark, silver, dedup.where("doc_id != 'x1'"), table, CRAWL, ())
    assert not (tmp_path / "gold_documents").exists()


def test_rows_of_another_pipeline_version_stop_the_write(
    spark: SparkSession, tmp_path: Path, silver_v1_batch: Batch, dedup: DataFrame
) -> None:
    """Gold holds exactly one pipeline_version (invariant 9)."""
    silver = silver_v1_batch(
        [{"doc_id": d} for d in ("a1", "a2", "a3", "b1", "c1")]
        + [{"doc_id": "x1", "pipeline_version": "0"}]
    )
    with pytest.raises(RuntimeError, match="1 rows not made by pipeline version"):
        run_gold(spark, silver, dedup, str(tmp_path / "gold_documents"), CRAWL, ())
