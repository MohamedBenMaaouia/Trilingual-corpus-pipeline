"""Which silver pages reach gold (Story 6.2.1, DECISIONS S6-04).

The crawl's kept silver_v1 pages (high + medium) that dedup keeps (duplicate_type null,
S4-06), minus the excluded sites. Decided on a few light columns first, without the
text; the chosen pages' full rows are then read once for the write, joined to a
broadcast list of their ids, so the text is never shuffled and never persisted.

dedup_v1 has one row per kept silver page: a page without its dedup row means dedup
ran on another version of silver, and the job refuses to write (rerun dedup).
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from corpus.schemas.gold_v1 import GOLD_SCHEMA_VERSION, GOLD_V1, QUALITY_TIERS
from corpus.version import PIPELINE_VERSION

# What the decision needs from silver: no text.
LIGHT_COLUMNS = ("doc_id", "url", "domain", "language", "quality_tier", "pipeline_version")


def kept_pages(silver: DataFrame, crawl_id: str) -> DataFrame:
    """One crawl's silver_v1 rows in gold's tiers. Both filters are on partition columns:
    only those folders are read."""
    in_crawl = F.col("crawl_id") == crawl_id
    return silver.where(in_crawl & F.col("quality_tier").isin(*QUALITY_TIERS))


def with_dedup(pages: DataFrame, dedup: DataFrame) -> DataFrame:
    """Each page (light columns) with its dedup_v1 row: cluster_size, duplicate_type, the
    row's pipeline version, and in_dedup (null when the page has no dedup row).

    A left join, so a missing row shows up instead of silently dropping the page.
    dedup_v1 is small (doc ids and a few numbers, ~24 MB for the full crawl): broadcast."""
    rows = dedup.select(
        "doc_id",
        "cluster_size",
        "duplicate_type",
        F.col("pipeline_version").alias("dedup_pipeline_version"),
        F.lit(True).alias("in_dedup"),
    )
    return pages.join(F.broadcast(rows), "doc_id", "left")


def to_gold_v1(pages: DataFrame, chosen: DataFrame) -> DataFrame:
    """The chosen pages' silver_v1 rows in the gold contract's shape. `chosen` (doc_id,
    cluster_size) is a few MB: broadcast, so the pages, text included, stay where they
    were read. Versions are stamped as constants: the job has checked that silver and
    dedup were made by this pipeline version."""
    joined = pages.drop("schema_version", "pipeline_version").join(
        F.broadcast(chosen.select("doc_id", "cluster_size")), "doc_id"
    )
    return joined.withColumn("schema_version", F.lit(GOLD_SCHEMA_VERSION)).select(
        *[
            (F.lit(PIPELINE_VERSION) if f.name == "pipeline_version" else F.col(f.name))
            .cast(f.dataType)
            .alias(f.name)
            for f in GOLD_V1.fields
        ]
    )
