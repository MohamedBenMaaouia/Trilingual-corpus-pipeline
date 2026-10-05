"""Writing gold_documents, a Delta table (Stories 6.2-6.3; D12, DECISIONS S6-02, S6-06).

- Created from the contract (createIfNotExists): declared, never inferred. Delta keeps
  the NOT NULL columns and refuses any write that breaks them (S6-01).
- A crawl is replaced with replaceWhere on crawl_id, a partition column (D12): one
  commit removes that crawl's files and adds the new ones, atomically. Another crawl's
  files are never read or rewritten, and readers see the old rows or the new ones,
  never a gap (silver's delete-then-append has one, S3-09).
- One writer at a time: Delta's log on S3A is safe for a single writing driver only
  (S3 has no atomic "create if absent" Delta can rely on across processes). The DAG
  runs one run at a time (max_active_runs=1); never run two gold jobs on one table.
- OPTIMIZE after each write, on the crawl's partitions only: small files packed into
  files of at most 1 GB (a maximum, not a target: C19). VACUUM deletes files removed
  from the table more than 7 days ago: time travel to a version works for at least 7
  days, after which its files may be gone (S6-06).
"""

from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F

from corpus.io import check_crawl_id
from corpus.schemas.contract import assert_schema
from corpus.schemas.gold_v1 import GOLD_V1, PARTITION_COLUMNS

# How long removed files are kept for time travel before VACUUM may delete them
# (plan 6.3.4; Delta's default, stated so it is visible in the table's properties).
RETENTION_DAYS = 7
RETENTION = f"interval {RETENTION_DAYS} days"


def create_table(spark: SparkSession, path: str) -> None:
    """Create the table from the contract if it does not exist yet, then check that the
    table there has the contract's schema, nullability included (Delta keeps it)."""
    (
        DeltaTable.createIfNotExists(spark)
        .location(path)
        .addColumns(GOLD_V1)
        .partitionedBy(*PARTITION_COLUMNS)
        .property("delta.deletedFileRetentionDuration", RETENTION)
        .execute()
    )
    assert_schema(spark.read.format("delta").load(path), GOLD_V1)


def _crawl_predicate(crawl_id: str) -> str:
    check_crawl_id(crawl_id)  # the id goes into SQL: only CC-MAIN-YYYY-WW gets through
    return f"crawl_id = '{crawl_id}'"


def replace_crawl(rows: DataFrame, path: str, crawl_id: str) -> None:
    """Replace one crawl's rows with `rows` in one commit. Delta refuses the write if a
    row belongs to another crawl (DELTA_REPLACE_WHERE_MISMATCH) or breaks NOT NULL."""
    (
        rows.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", _crawl_predicate(crawl_id))
        .save(path)
    )


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def last_commit(spark: SparkSession, path: str) -> tuple[int, str, dict[str, float]]:
    """The table's latest commit: version, operation, and its numbers (files and bytes
    added and removed, rows written), from the log's commitInfo."""
    row = DeltaTable.forPath(spark, path).history(1).first()
    assert row is not None  # a table always has a version 0
    metrics = {name: _number(value) for name, value in (row.operationMetrics or {}).items()}
    return int(row.version), str(row.operation), metrics


def crawl_files(spark: SparkSession, path: str, crawl_id: str) -> list[Row]:
    """The crawl's data files in the table's current version, per partition: how many,
    total bytes, largest. From the files' metadata column (paths and sizes; no column of
    data is read)."""
    files = (
        spark.read.format("delta")
        .load(path)
        .where(F.col("crawl_id") == crawl_id)
        .select("language", "quality_tier", "_metadata.file_path", "_metadata.file_size")
        .distinct()
    )
    return (
        files.groupBy("language", "quality_tier")
        .agg(
            F.count("*").alias("files"),
            F.sum("file_size").alias("bytes"),
            F.max("file_size").alias("largest"),
        )
        .orderBy("language", "quality_tier")
        .collect()  # one row per partition: at most languages x tiers
    )


def optimize_crawl(spark: SparkSession, path: str, crawl_id: str) -> dict[str, float]:
    """OPTIMIZE (bin-packing) the crawl's partitions. The largest file it writes is
    spark.databricks.delta.optimize.maxFileSize (the job sets 1 GB). Returns Delta's
    own counts: partitions optimized, files considered, removed, added."""
    result = (
        DeltaTable.forPath(spark, path)
        .optimize()
        .where(_crawl_predicate(crawl_id))
        .executeCompaction()
        .select("metrics.*")
        .first()
    )
    assert result is not None
    return {
        "partitions_optimized": result.partitionsOptimized,
        "files_considered": result.totalConsideredFiles,
        "files_removed": result.numFilesRemoved,
        "files_added": result.numFilesAdded,
    }


def vacuum(spark: SparkSession, path: str) -> int:
    """Delete the files that left the table more than RETENTION ago (the table's
    property). Returns how many it deleted.

    With Delta's vacuum logging on, every VACUUM leaves two commits in the table's
    history, VACUUM START and VACUUM END (numDeletedFiles): an audit trail of what was
    deleted, and the count, without a second listing of the table (a dry run)."""
    spark.conf.set("spark.databricks.delta.vacuum.logging.enabled", "true")
    DeltaTable.forPath(spark, path).vacuum()
    _, operation, metrics = last_commit(spark, path)
    if operation != "VACUUM END":
        raise RuntimeError(f"VACUUM left no VACUUM END commit (latest: {operation})")
    return int(metrics.get("numDeletedFiles", 0.0))
