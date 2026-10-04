"""Stage 3a: exact duplicates (Story 4.1, DECISIONS S4-02).

Two kept pages with the same text are one page. "Same" is decided on a match key: the
silver text (normalize_common already ran, S2-06; invariant 5) folded by the Arabic
matching key (D6), so spellings that differ only by vowel marks, hamza forms, teh
marbuta, alef maqsura or digit shapes are the same text. Spark built-ins only.
"""

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from corpus.silver.normalize import arabic_key_translation


def match_key(text: Column) -> Column:
    """arabic_match_key, through Spark's translate(): one table, two engines (S3-08's
    parity test checks they agree). Only Arabic-script characters change."""
    matching, replacement = arabic_key_translation()
    return F.translate(text, matching, replacement)


def exact_hash(text: Column) -> Column:
    """xxhash64 of the match key (spec 3a). Spark's xxhash64 is already a signed 64-bit
    long (seed 42), so there is no unsigned value to convert (trap S4)."""
    return F.xxhash64(match_key(text))


def keep_order() -> list[Column]:
    """Which page of a group is kept, best first: highest quality_score, then earliest
    fetch_date, then smallest doc_id (C13; plan 4.1.1 and 4.4.4; the doc_id makes every
    tie-break total, CLAUDE.md section 8). Pages with the same text have the same score,
    so an exact group keeps its earliest copy, which is plan 4.1.1's rule."""
    return [F.col("quality_score").desc(), F.col("fetch_date").asc(), F.col("doc_id").asc()]


def exact_representatives(docs: DataFrame) -> DataFrame:
    """Add exact_rep: the doc_id that the page's group keeps (its own id when it is kept).

    `docs` needs doc_id, exact_hash, quality_score and fetch_date. A window over the
    hash: one shuffle of these few columns, never of the text.
    """
    group = Window.partitionBy("exact_hash").orderBy(*keep_order())
    return docs.withColumn("exact_rep", F.first("doc_id").over(group))
