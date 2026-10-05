"""corpus.gold.select: which silver pages reach gold, and in what shape (Story 6.2.1)."""

from collections.abc import Callable
from typing import Any

from pyspark.sql import DataFrame, SparkSession

from corpus.gold.select import LIGHT_COLUMNS, kept_pages, to_gold_v1, with_dedup
from corpus.schemas.contract import assert_schema
from corpus.schemas.gold_v1 import GOLD_V1
from corpus.version import PIPELINE_VERSION

Batch = Callable[[list[dict[str, Any]]], DataFrame]
OTHER_CRAWL = "CC-MAIN-2026-40"


def test_only_the_crawls_kept_tiers_are_read(silver_v1_batch: Batch) -> None:
    silver = silver_v1_batch(
        [
            {"doc_id": "h"},
            {"doc_id": "m", "quality_tier": "medium"},
            {"doc_id": "r", "quality_tier": "rejected"},
            {"doc_id": "x", "crawl_id": OTHER_CRAWL},
        ]
    )
    assert {r.doc_id for r in kept_pages(silver, "CC-MAIN-2026-39").collect()} == {"h", "m"}


def test_a_page_without_its_dedup_row_shows_up(
    silver_v1_batch: Batch, dedup_v1_batch: Batch
) -> None:
    pages = silver_v1_batch([{"doc_id": "a"}, {"doc_id": "b"}]).select(*LIGHT_COLUMNS)
    joined = with_dedup(pages, dedup_v1_batch([{"doc_id": "a", "duplicate_type": "exact"}]))
    rows = {r.doc_id: (r.in_dedup, r.duplicate_type) for r in joined.collect()}
    assert rows == {"a": (True, "exact"), "b": (None, None)}  # b is not silently dropped


def test_the_chosen_pages_get_the_gold_contracts_shape(
    spark: SparkSession, silver_v1_batch: Batch
) -> None:
    silver = silver_v1_batch([{"doc_id": "a", "pii_types": ["email"]}, {"doc_id": "b"}])
    chosen = spark.createDataFrame([("a", 3)], "doc_id string, cluster_size int")
    gold = to_gold_v1(silver, chosen)
    assert_schema(gold, GOLD_V1, check_nullability=False)
    [row] = gold.collect()
    assert (row.doc_id, row.cluster_size, row.pii_types) == ("a", 3, ["email"])
    assert (row.schema_version, row.pipeline_version) == ("gold_v1", PIPELINE_VERSION)
