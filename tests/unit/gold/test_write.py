"""corpus.gold.write: the gold Delta table's behaviour (Stories 6.2, 6.3, 6.5.1; D12).

Each test checks what Delta really did: the log's commits, the files a version is
made of, what an old version still reads.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.gold.write import (
    RETENTION,
    crawl_files,
    create_table,
    last_commit,
    optimize_crawl,
    replace_crawl,
    vacuum,
)
from corpus.schemas.contract import SchemaMismatch

Batch = Callable[[list[dict[str, Any]]], DataFrame]  # the conftest's row builders

A, B = "CC-MAIN-2026-39", "CC-MAIN-2026-40"


def ids(spark: SparkSession, table: str, version: int | None = None) -> set[str]:
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", version)
    return {r.doc_id for r in reader.load(table).select("doc_id").collect()}


def files_of(spark: SparkSession, table: str, crawl_id: str) -> list[str]:
    """The data files the table's current version holds for one crawl."""
    current = spark.read.format("delta").load(table).where(F.col("crawl_id") == crawl_id)
    return sorted(current.inputFiles())


@pytest.fixture
def table(spark: SparkSession, tmp_path: Path) -> str:
    path = str(tmp_path / "gold_documents")
    create_table(spark, path)
    return path


def test_the_table_is_created_from_the_contract(spark: SparkSession, table: str) -> None:
    detail = DeltaTable.forPath(spark, table).detail().first()
    assert detail is not None
    assert detail.partitionColumns == ["language", "quality_tier", "crawl_id"]
    assert detail.properties["delta.deletedFileRetentionDuration"] == RETENTION
    create_table(spark, table)  # a second call changes nothing
    assert last_commit(spark, table)[:2] == (0, "CREATE TABLE")


def test_a_table_of_another_shape_is_refused(spark: SparkSession, tmp_path: Path) -> None:
    path = str(tmp_path / "other")
    spark.createDataFrame([("x",)], "doc_id string").write.format("delta").save(path)
    with pytest.raises(SchemaMismatch, match="missing column"):
        create_table(spark, path)


def test_a_rerun_replaces_one_crawl_and_never_touches_another(
    spark: SparkSession, table: str, gold_v1_batch: Batch
) -> None:
    """D12: crawl_id is a partition column, so a rerun removes that crawl's files whole
    and copies no row of any other crawl (S6-02)."""
    replace_crawl(gold_v1_batch([{"doc_id": "a1"}, {"doc_id": "a2", "language": "fr"}]), table, A)
    replace_crawl(gold_v1_batch([{"doc_id": "b1", "crawl_id": B}]), table, B)
    b_files = files_of(spark, table, B)

    # The rerun no longer has a French page: its partition must not keep stale rows.
    replace_crawl(gold_v1_batch([{"doc_id": "a3"}]), table, A)
    assert ids(spark, table) == {"a3", "b1"}
    assert files_of(spark, table, B) == b_files  # the very same files: never rewritten
    _, operation, commit = last_commit(spark, table)
    assert operation == "WRITE"
    assert commit["numRemovedFiles"] >= 2 and commit["numCopiedRows"] == 0


def test_time_travel_reads_the_version_before_a_rerun(
    spark: SparkSession, table: str, gold_v1_batch: Batch
) -> None:
    """Plan 6.5.1: write, rewrite, then read the previous version: it differs."""
    replace_crawl(gold_v1_batch([{"doc_id": "a1"}, {"doc_id": "a2"}]), table, A)
    first = last_commit(spark, table)[0]
    replace_crawl(gold_v1_batch([{"doc_id": "a3"}]), table, A)

    assert ids(spark, table, version=first) == {"a1", "a2"}
    assert ids(spark, table) == {"a3"}
    history = DeltaTable.forPath(spark, table).history().orderBy("version").collect()
    assert [h.operation for h in history] == ["CREATE TABLE", "WRITE", "WRITE"]


def test_a_null_in_a_non_null_column_is_refused_and_nothing_lands(
    spark: SparkSession, table: str, gold_v1_batch: Batch
) -> None:
    """The contract's NOT NULL columns are enforced by Delta itself (S6-01)."""
    replace_crawl(gold_v1_batch([{"doc_id": "a1"}]), table, A)
    version = last_commit(spark, table)[0]
    with pytest.raises(Exception, match="DELTA_NOT_NULL_CONSTRAINT_VIOLATED"):
        replace_crawl(gold_v1_batch([{"doc_id": "a2", "url": None}]), table, A)
    assert last_commit(spark, table)[0] == version  # no commit: the old rows stay
    assert ids(spark, table) == {"a1"}


def test_a_row_of_another_crawl_is_refused(
    spark: SparkSession, table: str, gold_v1_batch: Batch
) -> None:
    with pytest.raises(Exception, match="DELTA_REPLACE_WHERE_MISMATCH"):
        replace_crawl(gold_v1_batch([{"doc_id": "b1", "crawl_id": B}]), table, A)
    assert ids(spark, table) == set()


def test_a_crawl_id_that_is_not_one_never_reaches_the_predicate(
    spark: SparkSession, table: str, gold_v1_batch: Batch
) -> None:
    with pytest.raises(ValueError, match="Common Crawl id"):
        replace_crawl(gold_v1_batch([{"doc_id": "a1"}]), table, "x' OR '1'='1")


def test_optimize_packs_one_crawls_small_files_only(
    spark: SparkSession, table: str, gold_v1_batch: Batch
) -> None:
    """Plan 6.3.1: four small files of one partition become one; the other crawl's files
    are left alone, and every row is still there."""
    rows = gold_v1_batch([{"doc_id": f"a{i}"} for i in range(8)]).repartition(4)
    replace_crawl(rows, table, A)
    replace_crawl(
        gold_v1_batch([{"doc_id": f"b{i}", "crawl_id": B} for i in range(4)]).repartition(2),
        table,
        B,
    )
    b_files = files_of(spark, table, B)
    [before] = crawl_files(spark, table, A)
    assert (before.language, before.quality_tier, before.files) == ("en", "high", 4)

    result = optimize_crawl(spark, table, A)
    [after] = crawl_files(spark, table, A)
    assert result == {
        "partitions_optimized": 1,
        "files_considered": 4,
        "files_removed": 4,
        "files_added": 1,
    }
    assert after.files == 1 and after.largest == after.bytes
    assert files_of(spark, table, B) == b_files
    assert len(ids(spark, table)) == 12
    assert last_commit(spark, table)[1] == "OPTIMIZE"


def test_vacuum_inside_the_retention_keeps_time_travel(
    spark: SparkSession, table: str, gold_v1_batch: Batch
) -> None:
    """S6-06: files removed less than 7 days ago survive VACUUM, so the old version
    still reads."""
    replace_crawl(gold_v1_batch([{"doc_id": "a1"}]), table, A)
    first = last_commit(spark, table)[0]
    replace_crawl(gold_v1_batch([{"doc_id": "a2"}]), table, A)
    assert vacuum(spark, table) == 0
    assert ids(spark, table, version=first) == {"a1"}
    operations = [h.operation for h in DeltaTable.forPath(spark, table).history(2).collect()]
    assert operations == ["VACUUM END", "VACUUM START"]  # the cleanup is in the history


def test_vacuum_without_retention_ends_time_travel(
    spark: SparkSession, table: str, gold_v1_batch: Batch
) -> None:
    """The other side of S6-06, shown once: with no retention (Delta's safety check
    switched off for this test only), VACUUM deletes the old version's files and that
    version can no longer be read. The job never does this."""
    replace_crawl(gold_v1_batch([{"doc_id": "a1"}]), table, A)
    first = last_commit(spark, table)[0]
    replace_crawl(gold_v1_batch([{"doc_id": "a2"}]), table, A)
    check = "spark.databricks.delta.retentionDurationCheck.enabled"
    spark.conf.set(check, "false")
    try:
        DeltaTable.forPath(spark, table).vacuum(0)
    finally:
        spark.conf.unset(check)
    assert ids(spark, table) == {"a2"}
    with pytest.raises(Exception, match="(?i)file.*(not exist|not found)|FileNotFound"):
        ids(spark, table, version=first)
