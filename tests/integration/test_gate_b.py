"""Gate B with the real Soda scan (Story 3.5, Sprint 3 DoD; DECISIONS S3-09).

A batch of silver_v1 rows -> the job's metrics (silver_v1_metrics) -> run_metrics in the
real Postgres -> Soda runs checks/gate_b.yml. A clean batch passes; a deliberately
corrupted batch fails, on exactly the checks it breaks. Soda is installed in the test
image since S3 (it used to exist only in the Airflow image, S1-06).
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from psycopg2.extensions import connection as Connection
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructField, StructType

from corpus.jobs.gate import run_checks
from corpus.jobs.run_silver_v1 import silver_v1_metrics
from corpus.metrics.emit import emit, start_run
from corpus.schemas.silver_v1 import SILVER_V1
from corpus.silver.language import LanguageRules

pytestmark = pytest.mark.integration

FETCHED = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def row(doc_id: str, **changes: Any) -> dict[str, Any]:
    """A valid silver_v1 row (a kept English page), with some fields changed."""
    good: dict[str, Any] = {
        "doc_id": doc_id, "url": f"http://e.com/{doc_id}", "domain": "e.com",
        "crawl_id": "CC-MAIN-2026-39", "segment_id": "00000", "fetch_date": FETCHED,
        "content_length": 100, "cc_language": "eng", "language": "en",
        "language_detected": "en", "language_conf": 0.9, "text": "A good page.",
        "char_count": 12, "word_count": 3, "mean_word_length": 3.3,
        "symbol_to_word_ratio": 0.0, "stopword_ratio": 0.3, "repeated_line_ratio": 0.0,
        "ellipsis_line_ratio": 0.0, "boilerplate_lines_removed": 2, "quality_tier": "high",
        "quality_score": 0.9, "reject_reasons": None, "pii_redacted": False,
        "pii_types": None, "schema_version": "silver_v1", "pipeline_version": "1",
    }  # fmt: skip
    return good | changes


def batch(spark: SparkSession, rows: list[dict[str, Any]]) -> DataFrame:
    # Every column nullable here, so a test can put a null where the contract forbids one.
    nullable = StructType([StructField(f.name, f.dataType, True) for f in SILVER_V1.fields])
    return spark.createDataFrame(
        [tuple(r[f] for f in SILVER_V1.fieldNames()) for r in rows], nullable
    )


def gate_on(
    conn: Connection, spark: SparkSession, crawl_id: str, run_id: str, rows: list[dict[str, Any]]
) -> tuple[bool, str]:
    """What run_silver_v1 records for this batch, then the real Gate B scan."""
    metrics = silver_v1_metrics(batch(spark, rows), LanguageRules())
    metrics |= {
        "silver_v1_schema_ok": 1,
        "silver_v1_documents_read": metrics["silver_v1_documents"],
        "silver_v1_documents_written": metrics["silver_v1_documents"],
    }
    start_run(conn, run_id, crawl_id, dev_mode=True)
    emit(conn, run_id, crawl_id, "silver", metrics)
    return run_checks("gate_b", {"crawl_id": crawl_id, "run_id": run_id})


def test_a_clean_batch_passes(conn: Connection, spark: SparkSession, crawl_id: str) -> None:
    rows = [
        row("a"),
        row("b", language="other", language_detected="de", quality_tier="rejected",
            reject_reasons=["language_not_targeted"], quality_score=0.0),
        row("c", text="", word_count=0, quality_tier="rejected", reject_reasons=["too_few_words"]),
    ]  # fmt: skip
    passed, report = gate_on(conn, spark, crawl_id, f"gate-b-clean-{crawl_id}", rows)
    assert passed, report


def test_a_corrupted_batch_fails_on_what_it_breaks(
    conn: Connection, spark: SparkSession, crawl_id: str
) -> None:
    rows = [
        row("a"),
        row("kept-but-empty", text=""),  # a kept page with no text
        row("bad-tier", quality_tier="excellent"),  # outside the vocabulary
        row("no-url", url=None),  # a null in a non-null column
    ]
    passed, report = gate_on(conn, spark, crawl_id, f"gate-b-corrupt-{crawl_id}", rows)
    assert not passed
    for check in (
        "Every kept (non-rejected) page has text [FAILED]",
        "quality_tier is one of high, medium, rejected [FAILED]",
        "No null in a non-null contract column [FAILED]",
    ):
        assert check in report, report
    assert "language is one of en, fr, ar, other [PASSED]" in report


def test_a_run_that_recorded_nothing_fails(conn: Connection, crawl_id: str) -> None:
    run_id = f"gate-b-empty-{crawl_id}"
    start_run(conn, run_id, crawl_id, dev_mode=True)
    passed, report = run_checks("gate_b", {"crawl_id": crawl_id, "run_id": run_id})
    assert not passed, report  # missing numbers read as -1: never a silent pass
