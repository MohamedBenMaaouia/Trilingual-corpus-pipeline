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
        counts = (
            parsed.groupBy(F.col("exception").isNull().alias("is_document")).count().collect()
        )  # two rows of numbers, not corpus data: safe to bring to the driver
        by_kind = {row.is_document: row["count"] for row in counts}
        print(f"run_silver: {by_kind.get(True, 0)} documents, {by_kind.get(False, 0)} dead letters")
        parsed.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
