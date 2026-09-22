"""corpus.jobs.smoke: the summary row, checked on the committed fixture."""

import gzip
from pathlib import Path

from pyspark.sql import SparkSession

from corpus.jobs.smoke import summarize_lines

SAMPLE = Path(__file__).parent.parent / "fixtures" / "sample.wet.gz"


def test_summary_is_one_row_with_the_real_line_count(spark: SparkSession) -> None:
    # The expected count, computed without Spark: plain Python reads the same file.
    with gzip.open(SAMPLE, "rt", encoding="utf-8") as f:
        expected = sum(1 for _ in f)

    summary = summarize_lines(spark.read.text(str(SAMPLE)), "CC-MAIN-2026-39", 1)

    assert summary.columns == ["crawl_id", "segment_count", "line_count"]
    assert [tuple(r) for r in summary.collect()] == [("CC-MAIN-2026-39", 1, expected)]
