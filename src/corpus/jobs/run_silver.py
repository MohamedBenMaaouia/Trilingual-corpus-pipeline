"""Silver job: bronze WET files become silver rows (Sprint 2, Stories 2.1-2.4).

So far: parse only, and nothing is written. Boilerplate (2.2), normalization (2.3),
the write and the metrics (2.4) are added story by story.

    python -m corpus.jobs.run_silver --crawl-id CC-MAIN-2026-39 [--dev]
"""

import argparse

from pyspark import StorageLevel
from pyspark.sql import functions as F

from corpus.bronze.control import SegmentControl
from corpus.db import connect
from corpus.io import check_crawl_id
from corpus.session import get_session
from corpus.silver.boilerplate import DomainRules, domain_boilerplate, remove_domain_boilerplate
from corpus.silver.parse import parse_files

# --dev = 5 segments (plan 2.4.1): the 5 lowest segment ids of the crawl's random
# sample, so still a random sample, and always the same 5 (S2-04).
DEV_SEGMENTS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--dev", action="store_true", help=f"only {DEV_SEGMENTS} segments")
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)

    # The file list comes from the control table: exactly what Gate A validated.
    with connect() as conn:
        files = SegmentControl(conn).complete_files(args.crawl_id)
    if args.dev:
        files = files[:DEV_SEGMENTS]
    if not files:
        raise SystemExit(f"no complete segments for {args.crawl_id}: run the download first")
    print(f"run_silver: {args.crawl_id}, {len(files)} files{' (dev)' if args.dev else ''}")

    spark = get_session(f"run_silver {args.crawl_id}{' dev' if args.dev else ''}")
    try:
        # binaryFile: one row per file (path, content, ...), read through S3A as the
        # processing user, who can read bronze and never write it (S2-02).
        raw = spark.read.format("binaryFile").load([final_key for _, final_key in files])
        # Persist once: the counts below and, from Story 2.4, the two writes all reuse
        # this result instead of re-reading and re-parsing bronze (S2-04).
        parsed = parse_files(raw).persist(StorageLevel.MEMORY_AND_DISK)
        # One aggregate row of numbers, not corpus data: safe to bring to the driver.
        totals = parsed.agg(
            F.count(F.when(F.col("exception").isNull(), 1)).alias("documents"),
            F.count(F.when(F.col("exception").isNotNull(), 1)).alias("dead_letters"),
            F.count(F.when(F.col("text") == "", 1)).alias("emptied"),
            F.sum("lines_removed_few_words").alias("few_words"),
            F.sum("lines_removed_no_sentence_end").alias("no_sentence_end"),
            F.sum("lines_removed_non_letters").alias("non_letters"),
        ).first()
        assert totals is not None
        print(f"run_silver: {totals.documents} documents, {totals.dead_letters} dead letters")
        print(
            f"run_silver: line rules removed A={totals.few_words} B={totals.no_sentence_end} "
            f"C={totals.non_letters} lines; {totals.emptied} documents left empty"
        )

        # Pass 1 of the domain rule: the job's first shuffle (S2-07). Persisted because
        # it is used twice (the report below and pass 2): without it, Spark would run
        # pass 1 and its shuffle again. It is small: one row per boilerplate line.
        domain_rules = DomainRules()
        documents = parsed.where(F.col("exception").isNull())
        boilerplate = domain_boilerplate(documents.select("domain", "text"), domain_rules)
        boilerplate = boilerplate.persist(StorageLevel.MEMORY_AND_DISK)
        found = boilerplate.agg(
            F.count("*").alias("lines"), F.countDistinct("domain").alias("domains")
        ).first()
        assert found is not None
        print(
            f"run_silver: pass 1 found {found.lines} boilerplate lines in {found.domains} "
            f"domains ({max(found.lines - domain_rules.max_entries, 0)} above the "
            f"{domain_rules.max_entries} cap)"
        )

        # Pass 2: remove them (a broadcast, no second shuffle), then the total per
        # document: the silver column boilerplate_lines_removed (S2-07, Q5).
        cleaned = remove_domain_boilerplate(documents, boilerplate, domain_rules).withColumn(
            "boilerplate_lines_removed",
            F.col("lines_removed_few_words")
            + F.col("lines_removed_no_sentence_end")
            + F.col("lines_removed_non_letters")
            + F.col("lines_removed_domain"),
        )
        after = cleaned.agg(
            F.sum("lines_removed_domain").alias("domain_lines"),
            F.count(F.when(F.col("lines_removed_domain") > 0, 1)).alias("touched"),
            F.count(F.when(F.col("text") == "", 1)).alias("emptied"),
            F.sum("boilerplate_lines_removed").alias("total_lines"),
        ).first()
        assert after is not None
        print(
            f"run_silver: pass 2 removed {after.domain_lines} lines from {after.touched} "
            f"documents; {after.total_lines} boilerplate lines removed in total; "
            f"{after.emptied} documents left empty"
        )
        boilerplate.unpersist()
        parsed.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
