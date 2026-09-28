"""Shape the cleaned documents into Sprint 2's interim output (Story 2.4, S2-10)."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from corpus.schemas.silver_stage1 import STAGE1_SCHEMA


def to_stage1(documents: DataFrame, files: DataFrame, crawl_id: str) -> DataFrame:
    """documents: cleaned rows (after pass 2), with `path` and `boilerplate_lines_removed`.
    files: one row per bronze file, (path, segment_id), from the control table.

    Attaches segment_id by the file's path (files is tiny: broadcast, no shuffle) and the
    crawl id, then keeps exactly the declared columns, in the declared order and types.
    A left join: a document whose file is unknown gets a null segment_id, which the job
    counts and refuses, instead of silently dropping the document.
    """
    joined = documents.join(F.broadcast(files), "path", "left").withColumn(
        "crawl_id", F.lit(crawl_id)
    )
    return joined.select(
        *[F.col(f.name).cast(f.dataType).alias(f.name) for f in STAGE1_SCHEMA.fields]
    )
