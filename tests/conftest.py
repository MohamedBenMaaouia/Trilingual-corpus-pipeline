"""Fixtures shared by every test. pytest finds this file automatically."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructField, StructType

from corpus.schemas.dedup_v1 import DEDUP_V1
from corpus.schemas.gold_v1 import GOLD_V1
from corpus.schemas.silver_v1 import SILVER_V1
from corpus.session import DELTA_CONF

CRAWL = "CC-MAIN-2026-39"
FETCHED = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)

# Batch = a function: a list of rows, each given as its changes from a valid default.
Batch = Callable[[list[dict[str, Any]]], DataFrame]


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """One real, in-process Spark for the whole test run (starting one costs seconds).

    Deliberately not corpus.session.get_session(): tests must not need the cluster
    or MinIO. local[2] = driver + 2 worker threads in this one process.
    """
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("corpus-tests")
        # Default 200 shuffle partitions would make every tiny test run 200 tasks.
        .config("spark.sql.shuffle.partitions", "4")
        # Delta's VACUUM lists the table with this many tasks (default 10,000): ~30 s of
        # empty tasks per VACUUM on 2 threads (S6-06). The gold job sets 16.
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "4")
        .config("spark.ui.enabled", "false")  # no web UI needed in tests
        .config("spark.sql.session.timeZone", "UTC")  # same results on every machine
    )
    for key, value in DELTA_CONF.items():
        builder = builder.config(key, value)
    session = builder.getOrCreate()
    yield session
    session.stop()


def _batch(
    spark: SparkSession, schema: StructType, defaults: Callable[[str], dict[str, Any]]
) -> Batch:
    # Every column nullable, as a Parquet read gives them, so a test can put a null
    # where the contract forbids one.
    nullable = StructType([StructField(f.name, f.dataType, True) for f in schema.fields])

    def build(rows: list[dict[str, Any]]) -> DataFrame:
        full = [defaults(r["doc_id"]) | r for r in rows]
        return spark.createDataFrame(
            [tuple(r[n] for n in schema.fieldNames()) for r in full], nullable
        )

    return build


def _silver_v1_row(doc_id: str) -> dict[str, Any]:
    """A valid silver_v1 row: a kept English page of CRAWL."""
    return {
        "doc_id": doc_id, "url": f"http://site-{doc_id}.example.com/page",
        "domain": "example.com", "crawl_id": CRAWL, "segment_id": "00000",
        "fetch_date": FETCHED, "content_length": 100, "cc_language": "eng", "language": "en",
        "language_detected": "en", "language_conf": 0.9, "text": f"A good page, {doc_id}.",
        "char_count": 12, "word_count": 3, "mean_word_length": 3.3,
        "symbol_to_word_ratio": 0.0, "stopword_ratio": 0.3, "repeated_line_ratio": 0.0,
        "ellipsis_line_ratio": 0.0, "boilerplate_lines_removed": 2, "quality_tier": "high",
        "quality_score": 0.9, "reject_reasons": None, "pii_redacted": False,
        "pii_types": None, "schema_version": "silver_v1", "pipeline_version": "1",
    }  # fmt: skip


def _dedup_v1_row(doc_id: str) -> dict[str, Any]:
    """A valid dedup_v1 row: a page with no duplicate, which gold keeps."""
    return {
        "doc_id": doc_id, "language": "en", "cluster_id": doc_id, "cluster_size": 1,
        "duplicate_type": None, "schema_version": "dedup_v1", "pipeline_version": "1",
    }  # fmt: skip


def _gold_v1_row(doc_id: str) -> dict[str, Any]:
    """A valid gold_v1 row: an English high-tier page of CRAWL."""
    silver = _silver_v1_row(doc_id)
    gold = {name: silver[name] for name in GOLD_V1.fieldNames() if name in silver}
    return gold | {"cluster_size": 1, "schema_version": "gold_v1"}


@pytest.fixture
def silver_v1_batch(spark: SparkSession) -> Batch:
    """silver_v1 rows from their changes, e.g. [{"doc_id": "a"}, {"doc_id": "b", "text": ""}]."""
    return _batch(spark, SILVER_V1, _silver_v1_row)


@pytest.fixture
def dedup_v1_batch(spark: SparkSession) -> Batch:
    """dedup_v1 rows from their changes, e.g. [{"doc_id": "a", "duplicate_type": "exact"}]."""
    return _batch(spark, DEDUP_V1, _dedup_v1_row)


@pytest.fixture
def gold_v1_batch(spark: SparkSession) -> Batch:
    """gold_v1 rows from their changes, e.g. [{"doc_id": "a", "crawl_id": "CC-MAIN-2026-40"}]."""
    return _batch(spark, GOLD_V1, _gold_v1_row)
