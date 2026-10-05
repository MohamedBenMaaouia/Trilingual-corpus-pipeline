"""Plan 6.5.3: a deliberately bad batch is blocked by Gate B and never reaches gold.

The real path: the batch's numbers go to run_metrics in the real Postgres, the real
Soda scan judges them (checks/gate_b.yml) and records its verdict, then the gold job's
entry (gold_for_crawl) reads that verdict before it reads silver (DECISIONS S6-08).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from delta.tables import DeltaTable
from psycopg2.extensions import connection as Connection
from pyspark.sql import DataFrame, SparkSession

from corpus.jobs.gate import record_result, run_checks
from corpus.jobs.run_gold import GateNotPassed, gold_for_crawl, require_gate_b
from corpus.jobs.run_silver_v1 import silver_v1_metrics
from corpus.metrics.emit import emit, stamp_versions, start_run
from corpus.schemas.silver_v1 import SCHEMA_VERSION
from corpus.silver.language import LanguageRules
from corpus.version import PIPELINE_VERSION

pytestmark = pytest.mark.integration

Batch = Callable[[list[dict[str, Any]]], DataFrame]

CLEAN: list[dict[str, Any]] = [{"doc_id": "a"}, {"doc_id": "b", "language": "fr"}]
BAD: list[dict[str, Any]] = [
    {"doc_id": "a"},
    {"doc_id": "kept-but-empty", "text": ""},  # a kept page with no text
    {"doc_id": "bad-tier", "quality_tier": "excellent"},  # outside the vocabulary
    {"doc_id": "no-url", "url": None},  # a null in a non-null column
]


def silver_run(conn: Connection, crawl_id: str, run_id: str, *, silver: DataFrame) -> bool:
    """What a silver_v1 run and Gate B leave behind for this batch: the run (stamped
    silver_v1, dev), its numbers, the real Soda verdict. Returns whether it passed."""
    start_run(conn, run_id, crawl_id, dev_mode=True)
    stamp_versions(conn, run_id, PIPELINE_VERSION, SCHEMA_VERSION)
    metrics = silver_v1_metrics(silver, LanguageRules())
    metrics |= {
        "silver_v1_schema_ok": 1,
        "silver_v1_documents_read": metrics["silver_v1_documents"],
        "silver_v1_documents_written": metrics["silver_v1_documents"],
    }
    emit(conn, run_id, crawl_id, "silver", metrics)
    passed, report = run_checks("gate_b", {"crawl_id": crawl_id, "run_id": run_id})
    record_result(run_id, crawl_id, "gate_b", passed, report)
    return passed


def test_a_bad_batch_is_blocked_by_gate_b_and_never_reaches_gold(
    conn: Connection,
    spark: SparkSession,
    crawl_id: str,
    tmp_path: Path,
    silver_v1_batch: Batch,
    dedup_v1_batch: Batch,
) -> None:
    table = str(tmp_path / "gold_documents")

    def batch(rows: list[dict[str, Any]]) -> DataFrame:
        return silver_v1_batch([r | {"crawl_id": crawl_id} for r in rows])

    bad = batch(BAD)
    assert not silver_run(conn, crawl_id, f"gold-gate-bad-{crawl_id}", silver=bad)
    with pytest.raises(GateNotPassed, match="Gate B failed"):
        gold_for_crawl(conn, spark, bad, dedup_v1_batch(BAD), table, crawl_id, dev=True, sites=())
    assert not DeltaTable.isDeltaTable(spark, table)  # gold was never even created

    # The crawl, cleaned and judged again by a later run: now it reaches gold.
    clean = batch(CLEAN)
    assert silver_run(conn, crawl_id, f"gold-gate-clean-{crawl_id}", silver=clean)
    metrics, _ = gold_for_crawl(
        conn, spark, clean, dedup_v1_batch(CLEAN), table, crawl_id, dev=True, sites=()
    )
    assert metrics["gold_documents_written"] == 2
    gold = spark.read.format("delta").load(table)
    assert {r.doc_id for r in gold.collect()} == {"a", "b"}


def test_a_crawl_with_no_silver_run_is_refused(conn: Connection, crawl_id: str) -> None:
    with pytest.raises(GateNotPassed, match="no silver_v1 run"):
        require_gate_b(conn, crawl_id, dev=True)


def test_a_silver_run_gate_b_has_not_judged_is_refused(conn: Connection, crawl_id: str) -> None:
    """e.g. a silver_v1 run that crashed after replacing the crawl: no verdict, no gold."""
    run_id = f"gold-gate-unjudged-{crawl_id}"
    start_run(conn, run_id, crawl_id, dev_mode=True)
    stamp_versions(conn, run_id, PIPELINE_VERSION, SCHEMA_VERSION)
    with pytest.raises(GateNotPassed, match="has no verdict"):
        require_gate_b(conn, crawl_id, dev=True)
