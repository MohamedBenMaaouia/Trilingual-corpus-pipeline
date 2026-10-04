"""Stage 3's output, dedup_v1 (Sprint 4, DECISIONS S4-06): one row per kept silver page
of a crawl, with its cluster and whether gold keeps it.

Declared, never inferred, and checked before the write, like silver (invariant 3). The
crawl is the folder (dedup/crawl_id=...), not a column, as in Sprint 2's _stage1.
Small on purpose: no text (gold joins it to silver_v1 by doc_id in Sprint 6).
"""

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

DEDUP_SCHEMA_VERSION = "dedup_v1"

DEDUP_V1 = StructType(
    [
        StructField("doc_id", StringType(), False),
        StructField("language", StringType(), False),  # en | fr | ar (kept silver pages only)
        StructField("cluster_id", StringType(), False),  # doc_id of the page the cluster keeps
        StructField("cluster_size", IntegerType(), False),  # 1: no duplicate
        StructField("duplicate_type", StringType(), True),  # null: kept; exact | near: removed
        StructField("schema_version", StringType(), False),
        StructField("pipeline_version", StringType(), False),
    ]
)

DUPLICATE_TYPES = ("exact", "near")
