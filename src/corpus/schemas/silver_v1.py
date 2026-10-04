"""The silver contract, silver_v1 (spec 9.1, plan 3.4.1; D7 = extend, S2-10; DECISIONS S3-09).

Declared, never inferred (invariant 3). The spec's 17 columns plus D7's extensions:
segment_id, content_length, cc_language and boilerplate_lines_removed from Sprint 2,
language_detected and the five signals beyond char/word counts from Sprint 3.

Nullability states the promise. Spark marks most computed columns nullable and reads
every Parquet column back as nullable, so the non-null promise is not something the
files can hold: the job counts nulls in every non-null column before writing and
refuses to write if any (S3-09), and Gate B checks that count.
"""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

SCHEMA_VERSION = "silver_v1"

SILVER_V1 = StructType(
    [
        StructField("doc_id", StringType(), False),  # WARC-Record-ID uuid (T9)
        StructField("url", StringType(), False),
        StructField("domain", StringType(), False),
        StructField("crawl_id", StringType(), False),  # partition column
        StructField("segment_id", StringType(), False),  # the bronze file (D7)
        StructField("fetch_date", TimestampType(), False),
        StructField("content_length", LongType(), False),  # raw record, UTF-8 bytes (D7)
        StructField("cc_language", StringType(), True),  # Common Crawl's tag; can be absent
        StructField("language", StringType(), False),  # en | fr | ar | other (partition)
        StructField("language_detected", StringType(), True),  # fastText's label (D7)
        StructField("language_conf", DoubleType(), False),
        StructField("text", StringType(), False),  # normalized, cleaned, PII redacted
        StructField("char_count", IntegerType(), False),
        StructField("word_count", IntegerType(), False),
        StructField("mean_word_length", DoubleType(), False),  # signals (D7)
        StructField("symbol_to_word_ratio", DoubleType(), False),
        StructField("stopword_ratio", DoubleType(), False),
        StructField("repeated_line_ratio", DoubleType(), False),
        StructField("ellipsis_line_ratio", DoubleType(), False),
        StructField("boilerplate_lines_removed", LongType(), False),  # D7
        StructField("quality_tier", StringType(), False),  # high | medium | rejected (partition)
        StructField("quality_score", DoubleType(), False),
        StructField("reject_reasons", ArrayType(StringType()), True),  # null: kept
        StructField("pii_redacted", BooleanType(), False),
        StructField("pii_types", ArrayType(StringType()), True),  # null: none found
        StructField("schema_version", StringType(), False),
        StructField("pipeline_version", StringType(), False),
    ]
)

# Physical layout (plan 3.5.1): rejects are a partition of the same table (invariant 4).
PARTITION_COLUMNS = ("language", "quality_tier", "crawl_id")

# Controlled vocabularies (pipeline.md section 10), checked by the job and Gate B.
LANGUAGES = ("en", "fr", "ar", "other")
QUALITY_TIERS = ("high", "medium", "rejected")
