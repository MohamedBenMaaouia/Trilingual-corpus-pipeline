"""corpus_stats (Story 6.4.2; C21, DECISIONS S6-07): what one crawl adds to gold, per
language and quality tier. Computed from the rows as written (read back from the
table), so the numbers describe what landed. Cut (S3-05): token totals (tiktoken),
the top-20 domains.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def corpus_stats(gold: DataFrame) -> DataFrame:
    """One row per language x tier: documents, characters and words in total, mean
    length in characters, the first and last fetch date. A few rows: safe to collect."""
    return (
        gold.groupBy("language", "quality_tier")
        .agg(
            F.count("*").alias("documents"),
            F.sum(F.col("char_count").cast("long")).alias("chars_total"),
            F.sum(F.col("word_count").cast("long")).alias("words_total"),
            F.round(F.avg("char_count"), 1).alias("mean_chars"),
            F.min("fetch_date").alias("first_fetch"),
            F.max("fetch_date").alias("last_fetch"),
        )
        .orderBy("language", "quality_tier")
    )
