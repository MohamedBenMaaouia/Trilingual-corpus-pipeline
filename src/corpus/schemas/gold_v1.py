"""The gold contract, gold_v1 (Sprint 6; C24: the spec never defines gold's columns).

What a consumer of the corpus gets: one row per page that survived silver's filters,
dedup and the exclusion list. Silver's working columns stay in silver (segment_id,
content_length, cc_language, language_detected, the five signals, reject_reasons,
boilerplate_lines_removed); cluster_size comes from dedup (how many pages of the crawl
this page stands for). DECISIONS S6-03.

Unlike silver's Parquet files, a Delta table keeps nullability: the table is created
from this schema, and Delta refuses any write that puts a null in a non-null column
(DELTA_NOT_NULL_CONSTRAINT_VIOLATED, S6-01). The job also counts nulls before writing.
"""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

GOLD_SCHEMA_VERSION = "gold_v1"

GOLD_V1 = StructType(
    [
        StructField("doc_id", StringType(), False),  # WARC-Record-ID uuid (T9)
        StructField("url", StringType(), False),
        StructField("domain", StringType(), False),  # registrable domain
        StructField("crawl_id", StringType(), False),  # partition (D12)
        StructField("fetch_date", TimestampType(), False),
        StructField("language", StringType(), False),  # en | fr | ar (partition)
        StructField("language_conf", DoubleType(), False),
        StructField("text", StringType(), False),  # normalized, cleaned, PII redacted
        StructField("char_count", IntegerType(), False),
        StructField("word_count", IntegerType(), False),
        StructField("quality_tier", StringType(), False),  # high | medium (partition)
        StructField("quality_score", DoubleType(), False),
        StructField("pii_redacted", BooleanType(), False),
        StructField("pii_types", ArrayType(StringType()), True),  # null: none found
        StructField("cluster_size", IntegerType(), False),  # 1: no duplicate in the crawl
        StructField("schema_version", StringType(), False),
        StructField("pipeline_version", StringType(), False),
    ]
)

# Physical layout (plan 6.2.1 + D12, DECISIONS S6-02): crawl last, so a rerun replaces
# whole files of one crawl and OPTIMIZE never mixes two crawls in one file.
PARTITION_COLUMNS = ("language", "quality_tier", "crawl_id")

# Controlled vocabularies: gold holds the target languages and the kept tiers only.
LANGUAGES = ("en", "fr", "ar")
QUALITY_TIERS = ("high", "medium")
