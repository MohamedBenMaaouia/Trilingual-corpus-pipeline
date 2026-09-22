"""Sprint 0 smoke job (task 0.5.2): bronze WET files -> line count -> one-row Parquet in silver.

No real transform logic: it proves the whole path works (Airflow -> Spark cluster ->
MinIO read -> MinIO write -> history server).
"""

import argparse

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from corpus.io import bronze_path, smoke_output_path
from corpus.session import get_session


def summarize_lines(lines: DataFrame, crawl_id: str, segment_count: int) -> DataFrame:
    """One row describing what was read: which crawl, how many segments, how many lines.

    `lines` is what spark.read.text gives: one row per line of text, column `value`.
    """
    return lines.agg(F.count(F.lit(1)).alias("line_count")).select(
        F.lit(crawl_id).alias("crawl_id"),
        F.lit(segment_count).alias("segment_count"),
        F.col("line_count"),
    )


def main() -> None:
    """python -m corpus.jobs.smoke --crawl-id CC-MAIN-2026-39 --segments 0 1 2"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--segments", type=int, nargs="+", required=True)
    args = parser.parse_args()

    spark = get_session("smoke")
    try:
        paths = [bronze_path(args.crawl_id, s) for s in args.segments]
        lines = spark.read.text(paths)
        summary = summarize_lines(lines, args.crawl_id, len(args.segments))
        # overwrite: a rerun replaces the previous smoke result instead of adding to it.
        summary.write.mode("overwrite").parquet(smoke_output_path())
    finally:
        # Always release the executors, even if the job failed halfway.
        spark.stop()


if __name__ == "__main__":
    main()
