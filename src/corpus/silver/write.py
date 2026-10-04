"""Writing silver_v1 crawl by crawl (Story 3.5.1; D18, DECISIONS S3-01 and S3-09).

silver_v1 is partitioned language / quality_tier / crawl_id, so one crawl spreads over
up to 12 folders. A rerun must replace exactly that crawl (invariant 6):
- overwriting the table root would erase every other crawl;
- Spark's dynamic partition overwrite replaces only the folders the new output has,
  so a folder the rerun no longer produces (a tier that lost its last page) would
  keep stale rows.
So: delete every folder of the crawl, then append. A crash in between leaves the crawl
missing or partial; the rerun deletes and writes again (invariant 7), and Gate B runs
after the write, so nothing downstream reads the gap (invariant 8).
"""

from pyspark.sql import DataFrame, SparkSession

from corpus.schemas.silver_v1 import PARTITION_COLUMNS


def delete_matching(spark: SparkSession, pattern: str) -> int:
    """Delete every folder matching a glob, through Spark's own Hadoop filesystem (the
    same S3A client and credentials as the write: the processing user, which can write
    silver and never bronze, S2-02). Returns how many folders were deleted."""
    jvm = spark.sparkContext._jvm
    assert jvm is not None  # always set on the driver
    path = jvm.org.apache.hadoop.fs.Path(pattern)
    filesystem = path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    matches = filesystem.globStatus(path) or []
    for status in matches:
        filesystem.delete(status.getPath(), True)
    return len(matches)


def replace_crawl(spark: SparkSession, df: DataFrame, root: str, crawl_glob: str) -> int:
    """Replace one crawl's rows in the table at `root` with `df` (D18).

    `crawl_glob` comes from corpus.io (silver_v1_crawl_glob). Returns the number of old
    folders deleted.
    """
    deleted = delete_matching(spark, crawl_glob)
    df.write.mode("append").partitionBy(*PARTITION_COLUMNS).parquet(root)
    return deleted
