"""The interim silver output of Sprint 2 (plan 2.4.1, DECISIONS S2-10).

Sprint 3 reads it, adds language, quality and PII, and writes the real contract,
silver_v1. Declared, never inferred. Nullable except crawl_id (a constant): Spark marks
computed columns nullable, so this interim output does not promise more (the job checks
the columns that matter, e.g. segment_id). silver_v1 will, and Sprint 3 rebuilds the
DataFrame to get there (traps, S3).
"""

from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

STAGE1_SCHEMA = StructType(
    [
        StructField("doc_id", StringType(), True),  # WARC-Record-ID uuid (T9)
        StructField("url", StringType(), True),
        StructField("domain", StringType(), True),  # registrable domain, lowercase
        StructField("crawl_id", StringType(), False),  # a constant per run: never null
        StructField("segment_id", StringType(), True),  # which bronze file (T6)
        StructField("fetch_date", TimestampType(), True),  # WARC-Date, UTC
        StructField("content_length", LongType(), True),  # raw record, UTF-8 bytes (D7)
        StructField("cc_language", StringType(), True),  # Common Crawl's own guess (D7)
        StructField("text", StringType(), True),  # normalized, boilerplate removed
        StructField("boilerplate_lines_removed", LongType(), True),  # all four rules (D7)
    ]
)
