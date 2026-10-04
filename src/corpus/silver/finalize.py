"""Shape classified documents into the silver_v1 contract (Story 3.4, DECISIONS S3-09)."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from corpus.schemas.silver_v1 import SCHEMA_VERSION, SILVER_V1
from corpus.version import PIPELINE_VERSION


def to_silver_v1(docs: DataFrame) -> DataFrame:
    """Stamp the versions (invariant 3) and keep exactly the contract's columns, in its
    order and types. Helper columns (language_reject_reason, stopword_count, path, ...)
    are dropped here. Nullability is proven by the job's null count, not by this."""
    stamped = docs.withColumn("schema_version", F.lit(SCHEMA_VERSION)).withColumn(
        "pipeline_version", F.lit(PIPELINE_VERSION)
    )
    return stamped.select(*[F.col(f.name).cast(f.dataType).alias(f.name) for f in SILVER_V1.fields])
