"""Silver job, Sprint 2: bronze WET files -> the interim silver output (Stories 2.1-2.4).

parse + normalize + per-line rules (one Python pass) -> domain rule pass 1 (shuffle) ->
pass 2 (broadcast) -> the declared shape -> written, with its dead letters, per crawl.

    python -m corpus.jobs.run_silver --crawl-id CC-MAIN-2026-39 [--dev] [--run-id ...]
"""

import argparse
import time

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from corpus.bronze.control import SegmentControl
from corpus.db import connect
from corpus.io import check_crawl_id, dead_letter_path, silver_stage1_path
from corpus.metrics.emit import emit, finish_run, start_run
from corpus.run_id import default_run_id
from corpus.schemas.contract import assert_schema
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.session import get_session
from corpus.silver.boilerplate import (
    DomainRules,
    LineRules,
    domain_boilerplate,
    remove_domain_boilerplate,
)
from corpus.silver.parse import parse_files
from corpus.silver.stage1 import to_stage1

# --dev = 5 segments (plan 2.4.1): the 5 lowest segment ids of the crawl's random
# sample, so still a random sample, and always the same 5 (S2-04).
DEV_SEGMENTS = 5

DEAD_LETTER_COLUMNS = ["path", "record_offset", "exception", "raw"]


def build_stage1(
    spark: SparkSession,
    files: list[tuple[str, str]],
    crawl_id: str,
    stage1_out: str,
    dead_out: str,
) -> dict[str, float]:
    """Run the Sprint 2 chain on `files` ((segment_id, path) pairs) and write both outputs.

    Returns the run's silver metrics. The output locations are parameters so a test can
    run the whole chain on the fixture; the job passes the corpus.io paths.
    """
    line_rules, domain_rules = LineRules(), DomainRules()

    # binaryFile: one row per file, read through S3A as the processing user, who can read
    # bronze and never write it (S2-02). Persisted: every step below reuses the parse
    # instead of re-reading and re-parsing bronze (S2-04).
    raw = spark.read.format("binaryFile").load([path for _, path in files])
    parsed = parse_files(raw, line_rules).persist(StorageLevel.MEMORY_AND_DISK)
    documents = parsed.where(F.col("exception").isNull())

    # Domain rule, pass 1: the first shuffle (S2-07). Persisted: used twice (the metrics
    # and pass 2), and small (one row per boilerplate line).
    boilerplate = domain_boilerplate(documents.select("domain", "text"), domain_rules)
    boilerplate = boilerplate.persist(StorageLevel.MEMORY_AND_DISK)
    found = boilerplate.agg(
        F.count("*").alias("lines"), F.countDistinct("domain").alias("domains")
    ).first()
    assert found is not None

    # Pass 2 (broadcast, no shuffle), then the per-document total (D7).
    cleaned = remove_domain_boilerplate(documents, boilerplate, domain_rules).withColumn(
        "boilerplate_lines_removed",
        F.col("lines_removed_few_words")
        + F.col("lines_removed_no_sentence_end")
        + F.col("lines_removed_non_letters")
        + F.col("lines_removed_domain"),
    )

    # The declared shape (S2-10), checked before anything is written (invariant 3).
    segments = spark.createDataFrame(
        [(path, segment_id) for segment_id, path in files], "path string, segment_id string"
    )
    stage1 = to_stage1(cleaned, segments, crawl_id)
    assert_schema(stage1, STAGE1_SCHEMA)

    # One aggregate row of numbers, not corpus data: safe to bring to the driver.
    totals = cleaned.agg(
        F.count("*").alias("documents"),
        F.count(F.when(F.col("text") == "", 1)).alias("emptied"),
        F.sum("lines_removed_few_words").alias("few_words"),
        F.sum("lines_removed_no_sentence_end").alias("no_sentence_end"),
        F.sum("lines_removed_non_letters").alias("non_letters"),
        F.sum("lines_removed_domain").alias("domain"),
    ).first()
    unknown = stage1.where(F.col("segment_id").isNull()).count()
    dead_letters = parsed.where(F.col("exception").isNotNull()).select(*DEAD_LETTER_COLUMNS)
    assert totals is not None
    if unknown:
        # A file path that does not match the control table: never write rows that
        # cannot be traced back to their bronze file.
        raise RuntimeError(f"{unknown} documents have no segment_id: refusing to write")

    # Both writes replace this crawl's folder and nothing else (invariant 6), so a rerun
    # or a retried task is safe (invariant 7). Dead letters are written even when there
    # are none: a clean rerun must not leave the previous run's dead letters behind.
    stage1.write.mode("overwrite").parquet(stage1_out)
    dead_letters.write.mode("overwrite").parquet(dead_out)
    dead_count = spark.read.parquet(dead_out).count()  # what landed, not what was meant to

    boilerplate.unpersist()
    parsed.unpersist()
    return {
        "files_read": len(files),
        "documents": totals.documents,
        "dead_letters": dead_count,
        "lines_removed_few_words": totals.few_words or 0,
        "lines_removed_no_sentence_end": totals.no_sentence_end or 0,
        "lines_removed_non_letters": totals.non_letters or 0,
        "lines_removed_domain": totals.domain or 0,
        "boilerplate_entries": found.lines,
        "boilerplate_domains": found.domains,
        "boilerplate_entries_over_cap": max(found.lines - domain_rules.max_entries, 0),
        "documents_emptied": totals.emptied,
        "documents_written": totals.documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--dev", action="store_true", help=f"only {DEV_SEGMENTS} segments")
    parser.add_argument("--run-id", default=None, help="Airflow's run_id; generated if omitted")
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)
    run_id = args.run_id or default_run_id()

    # The file list comes from the control table: exactly what Gate A validated.
    with connect() as conn:
        files = SegmentControl(conn).complete_files(args.crawl_id)
        if args.dev:
            files = files[:DEV_SEGMENTS]
        if not files:
            raise SystemExit(f"no complete segments for {args.crawl_id}: run the download first")
        # Opens the run, or reopens it (upsert) when acquire already did; marks dev runs,
        # whose numbers never feed gate baselines (T11).
        start_run(conn, run_id, args.crawl_id, dev_mode=args.dev)
    stage1_out = silver_stage1_path(args.crawl_id, dev=args.dev)
    dead_out = dead_letter_path("silver", args.crawl_id, dev=args.dev)
    print(f"run_silver: {args.crawl_id}, {len(files)} files{' (dev)' if args.dev else ''}")

    started = time.monotonic()
    spark = get_session(f"run_silver {args.crawl_id}{' dev' if args.dev else ''}")
    try:
        metrics = build_stage1(spark, files, args.crawl_id, stage1_out, dead_out)
    except Exception:
        # The last task of the DAG closes the run, and a crash closes it as failed (S2-10).
        with connect() as conn:
            finish_run(conn, run_id, "failed")
        raise
    finally:
        spark.stop()
    metrics["duration_seconds"] = round(time.monotonic() - started, 1)

    with connect() as conn:
        emit(conn, run_id, args.crawl_id, "silver", metrics)
        finish_run(conn, run_id, "success")
    for name, value in metrics.items():
        print(f"run_silver: {name} = {value:,}")
    print(f"run_silver: wrote {stage1_out} and {dead_out} (run {run_id})")


if __name__ == "__main__":
    main()
