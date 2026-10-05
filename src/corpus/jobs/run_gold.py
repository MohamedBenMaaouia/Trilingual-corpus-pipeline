"""Gold job, Sprint 6: one crawl's kept, deduplicated silver pages -> gold_documents (Delta).

Gate B's verdict on the crawl's silver (no pass: refuse) -> kept silver_v1 pages (high +
medium) with their dedup_v1 rows -> the pages dedup keeps -> minus the excluded sites
-> the gold_v1 contract -> replace the crawl in gold_documents (replaceWhere, D12)
-> OPTIMIZE the crawl's partitions, VACUUM -> corpus_stats from the rows written

python -m corpus.jobs.run_gold --crawl-id CC-MAIN-2026-39 [--dev] [--run-id ...]
    [--keep-run-open] [--spark-conf KEY=VALUE ...]
"""

import argparse
import time
from collections.abc import Sequence
from datetime import UTC
from typing import Any

from psycopg2.extensions import connection as Connection
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.db import connect
from corpus.dedup.clusters import EXACT, NEAR
from corpus.gold.exclusions import exclude_sites, load_exclusions
from corpus.gold.select import LIGHT_COLUMNS, kept_pages, to_gold_v1, with_dedup
from corpus.gold.stats import corpus_stats
from corpus.gold.write import (
    RETENTION_DAYS,
    crawl_files,
    create_table,
    last_commit,
    optimize_crawl,
    replace_crawl,
    vacuum,
)
from corpus.io import check_crawl_id, dedup_path, gold_documents_path, silver_v1_path
from corpus.metrics.emit import emit, finish_run, replace_corpus_stats, start_run
from corpus.metrics.spark_steps import step, step_metrics
from corpus.run_id import default_run_id
from corpus.schemas.contract import assert_schema
from corpus.schemas.dedup_v1 import DEDUP_V1
from corpus.schemas.gold_v1 import GOLD_V1, LANGUAGES
from corpus.schemas.silver_v1 import SCHEMA_VERSION as SILVER_SCHEMA_VERSION
from corpus.schemas.silver_v1 import SILVER_V1
from corpus.session import get_session
from corpus.version import PIPELINE_VERSION

PREFIX = "gold"  # run_metrics stage and metric-name prefix
MB = 1_000_000

# Spark settings for this job, applied at start; --spark-conf overrides them.
SPARK_SETTINGS = {
    # Every shuffle here is small (counts, file lists): 16 as in dedup (S5-05), not 200.
    "spark.sql.shuffle.partitions": "16",
    # OPTIMIZE writes files of at most 1 GiB (C19: a maximum, not a target). Delta's
    # default, stated here so it is visible and the same on Databricks.
    "spark.databricks.delta.optimize.maxFileSize": str(1024**3),
    # VACUUM lists the table's folders with this many tasks; Spark's default is 10,000,
    # i.e. ~10,000 empty tasks per listing for a table of a few dozen folders (S6-06).
    "spark.sql.sources.parallelPartitionDiscovery.parallelism": "16",
}


class GateNotPassed(RuntimeError):
    """The crawl's silver has no passing Gate B verdict: gold must not read it."""


def require_gate_b(conn: Connection, crawl_id: str, *, dev: bool) -> str:
    """The silver run gold is about to read, if Gate B passed it; raises otherwise.

    A crawl's silver_v1 rows are those of the latest run that wrote them (a silver_v1
    run stamps schema_version when it starts, S3-09). If that run failed, or Gate B has
    not judged it, gold refuses: bad silver never reaches gold (invariant 8), even when
    gold is started by hand, outside the DAG (S6-08)."""
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.run_id, g.passed
            FROM pipeline_runs r
            LEFT JOIN gate_results g ON g.run_id = r.run_id AND g.gate_name = 'gate_b'
            WHERE r.crawl_id = %s AND r.schema_version = %s AND r.dev_mode = %s
            ORDER BY r.started_at DESC, r.run_id DESC
            LIMIT 1
            """,
            (crawl_id, SILVER_SCHEMA_VERSION, dev),
        )
        row = cur.fetchone()
    which = f"{crawl_id}{' (dev)' if dev else ''}"
    if row is None:
        raise GateNotPassed(f"no silver_v1 run for {which}: nothing for gold to read")
    run_id, passed = row
    if not passed:
        verdict = "failed" if passed is False else "has no verdict for"
        raise GateNotPassed(f"Gate B {verdict} silver run {run_id!r} of {which}: gold refuses it")
    return str(run_id)


def _count_if(condition: Column) -> Column:
    return F.sum(F.when(condition, 1).otherwise(0))


def _per_language(rows: DataFrame, name: str) -> dict[str, float]:
    """`rows` counted per language, as <name>_<language> (every gold language, 0 if none)."""
    counts = {r.language: r["count"] for r in rows.groupBy("language").count().collect()}
    return {f"{name}_{lang}": float(counts.get(lang, 0)) for lang in LANGUAGES}


def selection_metrics(light: DataFrame) -> dict[str, float]:
    """What the selection found, from one aggregation over the light rows (each kept
    silver page with its dedup row), and the checks made before anything is written."""
    dup, in_dedup = F.col("duplicate_type"), F.col("in_dedup")
    aggregates = [
        F.count("*").alias("gold_silver_pages"),
        _count_if(in_dedup.isNull()).alias("gold_pages_without_dedup_row"),
        _count_if(
            ~F.col("pipeline_version").eqNullSafe(PIPELINE_VERSION)
            | (in_dedup.isNotNull() & ~F.col("dedup_pipeline_version").eqNullSafe(PIPELINE_VERSION))
        ).alias("gold_version_mismatches"),
        _count_if(~F.coalesce(F.col("language").isin(*LANGUAGES), F.lit(False))).alias(
            "gold_unknown_language"
        ),
        _count_if(dup == EXACT).alias("gold_exact_duplicates_removed"),
        _count_if(dup == NEAR).alias("gold_near_duplicates_removed"),
        _count_if(in_dedup.isNotNull() & dup.isNull()).alias("gold_documents_deduplicated"),
    ]
    totals = light.agg(*aggregates).first()
    assert totals is not None
    return {name: float(value or 0) for name, value in totals.asDict().items()}


def _utc_dates(row: dict[str, Any]) -> dict[str, Any]:
    """PySpark returns timestamps as naive datetimes in the driver's local time;
    astimezone reads them as such and makes them explicit UTC for Postgres."""
    return row | {name: row[name].astimezone(UTC) for name in ("first_fetch", "last_fetch")}


def _file_metrics(rows: Sequence[Any], when: str) -> dict[str, float]:
    files = sum(r.files for r in rows)
    size = sum(r.bytes for r in rows)
    for r in rows:
        print(
            f"run_gold: files {when}: {r.language}/{r.quality_tier}: {r.files} files, "
            f"{r.bytes / MB:,.1f} MB, largest {r.largest / MB:,.1f} MB"
        )
    return {
        f"gold_files_{when}": files,
        f"gold_bytes_{when}": size,
        f"gold_mean_file_mb_{when}": round(size / files / MB, 1) if files else 0.0,
        f"gold_largest_file_mb_{when}": round(max((r.largest for r in rows), default=0) / MB, 1),
        f"gold_partitions_{when}": len(rows),
    }


def run_gold(
    spark: SparkSession,
    silver: DataFrame,
    dedup: DataFrame,
    table: str,
    crawl_id: str,
    sites: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Select, write, compact and describe one crawl's gold rows. `silver` is the
    silver_v1 table, `dedup` the crawl's dedup_v1 rows, `table` the Delta table's path
    (parameters, so a test runs the whole job on local files). Returns the run's metrics
    and its corpus_stats rows."""
    check_crawl_id(crawl_id)
    metrics: dict[str, float] = {"gold_exclusion_sites": len(sites)}
    pages = kept_pages(silver, crawl_id)

    # Decide on light columns (no text): which pages, and whether the inputs agree.
    with step(spark, PREFIX, "select", metrics):
        light = with_dedup(pages.select(*LIGHT_COLUMNS), dedup).persist(StorageLevel.DISK_ONLY)
        metrics.update(selection_metrics(light))
        metrics["gold_dedup_rows"] = dedup.count()
        problems = [
            f"{metrics['gold_pages_without_dedup_row']:.0f} kept pages have no dedup row",
            f"{metrics['gold_dedup_rows']:.0f} dedup rows for "
            f"{metrics['gold_silver_pages']:.0f} kept pages",
            f"{metrics['gold_version_mismatches']:.0f} rows not made by pipeline version "
            f"{PIPELINE_VERSION}",
            f"{metrics['gold_unknown_language']:.0f} pages in a language gold does not hold",
        ]
        if (
            metrics["gold_pages_without_dedup_row"]
            or metrics["gold_dedup_rows"] != metrics["gold_silver_pages"]
            or metrics["gold_version_mismatches"]
            or metrics["gold_unknown_language"]
        ):
            # dedup ran on another version of silver, or a version changed (invariant 9).
            raise RuntimeError("gold refuses to write: " + "; ".join(problems))

        deduplicated = light.where(F.col("in_dedup") & F.col("duplicate_type").isNull())
        chosen = exclude_sites(deduplicated, sites).select(
            "doc_id", "cluster_size", "language", "quality_tier"
        )
        chosen = chosen.persist(StorageLevel.DISK_ONLY)
        metrics["gold_documents_selected"] = chosen.count()
        metrics["gold_documents_excluded"] = (
            metrics["gold_documents_deduplicated"] - metrics["gold_documents_selected"]
        )
        selected = _per_language(chosen, "gold_documents_selected")
        before = _per_language(deduplicated, "gold_documents_deduplicated")
        metrics.update(selected)
        for lang in LANGUAGES:
            metrics[f"gold_documents_excluded_{lang}"] = (
                before[f"gold_documents_deduplicated_{lang}"]
                - selected[f"gold_documents_selected_{lang}"]
            )

    # The chosen pages' full rows, read once; the contract checked; the crawl replaced.
    with step(spark, PREFIX, "write", metrics):
        rows = to_gold_v1(pages, chosen)
        assert_schema(rows, GOLD_V1, check_nullability=False)  # names, types, order
        create_table(spark, table)  # and the table's own schema, nullability included
        replace_crawl(rows, table, crawl_id)  # Delta enforces NOT NULL as it writes
        version, _, commit = last_commit(spark, table)
        metrics["gold_delta_version_written"] = version
        metrics["gold_write_rows"] = commit.get("numOutputRows", 0.0)
        metrics["gold_write_files_removed"] = commit.get("numRemovedFiles", 0.0)
        metrics["gold_write_bytes_removed"] = commit.get("numRemovedBytes", 0.0)

    # Small files packed per partition (plan 6.3.1), measured before and after.
    with step(spark, PREFIX, "optimize", metrics):
        metrics.update(_file_metrics(crawl_files(spark, table, crawl_id), "before_optimize"))
        optimized = optimize_crawl(spark, table, crawl_id)
        metrics.update({f"gold_optimize_{name}": value for name, value in optimized.items()})
        metrics.update(_file_metrics(crawl_files(spark, table, crawl_id), "after_optimize"))
        metrics["gold_delta_version_optimized"] = last_commit(spark, table)[0]
        metrics["gold_vacuum_files_deleted"] = vacuum(spark, table)
        metrics["gold_vacuum_retention_days"] = RETENTION_DAYS

    # What landed, described from the table itself (plan 6.4.2).
    with step(spark, PREFIX, "stats", metrics):
        written = spark.read.format("delta").load(table).where(F.col("crawl_id") == crawl_id)
        stats = [_utc_dates(row.asDict()) for row in corpus_stats(written).collect()]
        metrics["gold_documents_written"] = float(sum(row["documents"] for row in stats))
        for lang in LANGUAGES:
            metrics[f"gold_documents_written_{lang}"] = float(
                sum(row["documents"] for row in stats if row["language"] == lang)
            )
        if metrics["gold_documents_written"] != metrics["gold_documents_selected"]:
            raise RuntimeError(
                f"{metrics['gold_documents_written']:.0f} rows in gold for "
                f"{metrics['gold_documents_selected']:.0f} selected pages"
            )

    for df in (chosen, light):
        df.unpersist()
    return metrics, stats


def gold_for_crawl(
    conn: Connection,
    spark: SparkSession,
    silver: DataFrame,
    dedup: DataFrame,
    table: str,
    crawl_id: str,
    *,
    dev: bool,
    sites: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Gate B's verdict first, then run_gold: what main runs, and what the bad-batch
    test runs (plan 6.5.3)."""
    silver_run = require_gate_b(conn, crawl_id, dev=dev)
    print(f"run_gold: Gate B passed silver run {silver_run}")
    return run_gold(spark, silver, dedup, table, crawl_id, sites)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--dev", action="store_true", help="the dev run's 5 segments")
    parser.add_argument("--run-id", default=None, help="Airflow's run_id; generated if omitted")
    parser.add_argument(
        "--keep-run-open",
        action="store_true",
        help="not the DAG's last task: leave the run open for the task after it",
    )
    parser.add_argument(
        "--spark-conf",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a Spark setting for experiments; the DAG passes none",
    )
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)
    run_id = args.run_id or default_run_id()
    table = gold_documents_path(dev=args.dev)

    with connect() as conn:
        start_run(conn, run_id, args.crawl_id, dev_mode=args.dev)
    print(f"run_gold: {args.crawl_id}{' (dev)' if args.dev else ''} -> {table}")

    started = time.monotonic()
    spark = get_session(f"run_gold {args.crawl_id}{' dev' if args.dev else ''}")
    try:
        sites = load_exclusions()  # a malformed list stops the run before anything is read
        for key, setting in SPARK_SETTINGS.items():
            spark.conf.set(key, setting)
        for override in args.spark_conf:
            key, _, setting = override.partition("=")
            spark.conf.set(key, setting)
        silver = spark.read.schema(SILVER_V1).parquet(silver_v1_path(dev=args.dev))
        dedup = spark.read.schema(DEDUP_V1).parquet(dedup_path(args.crawl_id, dev=args.dev))
        with connect() as conn:
            metrics, stats = gold_for_crawl(
                conn, spark, silver, dedup, table, args.crawl_id, dev=args.dev, sites=sites
            )
        metrics.update(step_metrics(spark, PREFIX))
    except Exception:
        with connect() as conn:
            finish_run(conn, run_id, "failed")
        raise
    finally:
        spark.stop()
    metrics["gold_duration_seconds"] = round(time.monotonic() - started, 1)

    with connect() as conn:
        emit(conn, run_id, args.crawl_id, PREFIX, metrics)
        replace_corpus_stats(conn, run_id, args.crawl_id, PIPELINE_VERSION, stats)
        if not args.keep_run_open:
            finish_run(conn, run_id, "success")
    for name, value in sorted(metrics.items()):
        print(f"run_gold: {name} = {value:,}")
    for row in stats:
        print(f"run_gold: corpus_stats: {row}")


if __name__ == "__main__":
    main()
