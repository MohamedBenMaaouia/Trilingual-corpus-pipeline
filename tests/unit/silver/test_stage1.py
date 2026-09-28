"""corpus.silver.stage1.to_stage1: the interim output's shape (Story 2.4, S2-10)."""

from datetime import UTC, datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.schemas.contract import assert_schema
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.silver.stage1 import to_stage1

CRAWL = "CC-MAIN-2026-39"
FETCHED = datetime(2026, 9, 4, 14, 40, 25, tzinfo=UTC)


def documents(spark: SparkSession, paths: list[str]) -> DataFrame:
    """Cleaned rows as pass 2 leaves them, including a working column to be dropped."""
    rows = [
        (p, f"id-{i}", f"http://e.com/{i}", "e.com", FETCHED, 100, "eng", "Text.", 3, 2)
        for i, p in enumerate(paths)
    ]
    return spark.createDataFrame(
        rows,
        "path string, doc_id string, url string, domain string, fetch_date timestamp, "
        "content_length long, cc_language string, text string, "
        "boilerplate_lines_removed long, lines_removed_few_words long",
    )


def files(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [("s3a://b/seg=00048/f.wet.gz", "00048"), ("s3a://b/seg=01326/g.wet.gz", "01326")],
        "path string, segment_id string",
    )


def test_output_matches_the_declared_schema(spark: SparkSession) -> None:
    docs = documents(spark, ["s3a://b/seg=00048/f.wet.gz"])
    assert_schema(to_stage1(docs, files(spark), CRAWL), STAGE1_SCHEMA)


def test_segment_and_crawl_are_attached_and_working_columns_dropped(
    spark: SparkSession,
) -> None:
    docs = documents(spark, ["s3a://b/seg=00048/f.wet.gz", "s3a://b/seg=01326/g.wet.gz"])
    result = to_stage1(docs, files(spark), CRAWL)
    assert "lines_removed_few_words" not in result.columns
    assert "path" not in result.columns
    rows = {r.doc_id: (r.segment_id, r.crawl_id) for r in result.collect()}
    assert rows == {"id-0": ("00048", CRAWL), "id-1": ("01326", CRAWL)}


def test_a_document_from_an_unknown_file_is_kept_with_no_segment(spark: SparkSession) -> None:
    """Never silently dropped: the job counts null segment_ids and refuses to write."""
    docs = documents(spark, ["s3a://b/somewhere/else.wet.gz"])
    result = to_stage1(docs, files(spark), CRAWL)
    assert result.count() == 1
    assert result.where(F.col("segment_id").isNull()).count() == 1
