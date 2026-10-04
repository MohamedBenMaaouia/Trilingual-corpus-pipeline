"""Dedup job, Sprint 4: one crawl's kept silver pages -> the pages gold keeps.

kept silver_v1 pages (T14) -> per page: exact hash + MinHash signature (one pass)
-> exact duplicates (3a) -> bands of the exact representatives (3d)
-> band self-join -> candidates checked on their signatures
-> connected components (3e) -> one page per cluster (3f)
-> dedup_v1, written to dedup/crawl_id=<crawl> (crawl-scoped)

python -m corpus.jobs.run_dedup --crawl-id CC-MAIN-2026-39 [--dev] [--run-id ...]
    [--bands 16 --rows 8] [--keep-run-open] [--spark-conf KEY=VALUE ...]
"""

import argparse
import time
from dataclasses import dataclass, field

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.db import connect
from corpus.dedup.banding import LshParams, band_rows, candidate_pairs, verified_pairs
from corpus.dedup.clusters import EXACT, NEAR, assign_clusters, connected_components
from corpus.dedup.exact import exact_hash, exact_representatives
from corpus.dedup.minhash import MinHashParams, add_signatures
from corpus.io import check_crawl_id, dedup_path, silver_v1_path
from corpus.metrics.emit import emit, finish_run, start_run
from corpus.metrics.spark_steps import step, step_metrics
from corpus.run_id import default_run_id
from corpus.schemas.contract import assert_schema, null_count
from corpus.schemas.dedup_v1 import DEDUP_SCHEMA_VERSION, DEDUP_V1, DUPLICATE_TYPES
from corpus.schemas.silver_v1 import SILVER_V1
from corpus.session import get_session
from corpus.silver.language import LanguageRules
from corpus.silver.quality import REJECTED
from corpus.version import PIPELINE_VERSION

PREFIX = "dedup"  # run_metrics stage and metric-name prefix
TOP_KEYS = 20  # the largest buckets, logged (plan 5.4.1)
TOP_CLUSTERS = 10  # the largest clusters, logged (plan 4.4.3)


@dataclass(frozen=True)
class DedupParams:
    minhash: MinHashParams = field(default_factory=MinHashParams)
    lsh: LshParams = field(default_factory=LshParams)
    # Label propagation stops here even if not converged, and says so (plan 4.4.1).
    max_rounds: int = 30
    # Clusters above this size are listed for investigation (plan 4.4.3).
    mega_cluster: int = 1_000


def _count_if(condition: Column) -> Column:
    return F.sum(F.when(condition, 1).otherwise(0))


def bucket_sizes(bands: DataFrame, top: int) -> tuple[dict[str, float], DataFrame]:
    """How the band rows spread over buckets: the skew evidence (plan 5.4.1) and what
    each way of linking a bucket would cost. A bucket of n pages gives n(n-1)/2 pairs
    to the self-join (all pairs), or n - 1 links to its smallest page (a star, D11)."""
    sizes = bands.groupBy("band_id", "band_hash").agg(F.count("*").alias("rows"))
    sizes = sizes.persist(StorageLevel.DISK_ONLY)
    rows = F.col("rows").cast("long")
    totals = sizes.agg(
        F.count("*").alias("dedup_band_keys"),
        F.sum(rows).alias("dedup_band_rows"),
        _count_if(rows > 1).alias("dedup_band_keys_shared"),
        F.sum(rows * (rows - 1) / 2).alias("dedup_pairs_all_pairs"),
        F.sum(F.when(rows > 1, rows - 1).otherwise(0)).alias("dedup_pairs_star"),
        F.max(rows).alias("dedup_band_key_max_rows"),
    ).first()
    assert totals is not None
    metrics = {k: float(v or 0) for k, v in totals.asDict().items()}
    largest = sizes.orderBy(F.desc("rows"), "band_id", "band_hash").limit(top)
    return metrics, largest


def describe_keys(largest: DataFrame, bands: DataFrame, kept: DataFrame) -> list[str]:
    """One line per large bucket: its size, how many sites, one page's url. A few
    rows come back to the driver, never page text."""
    members = bands.join(F.broadcast(largest), ["band_id", "band_hash"]).join(
        kept.select("doc_id", "url", "domain"), "doc_id"
    )
    rows = (
        members.groupBy("band_id", "band_hash", "rows")
        .agg(F.countDistinct("domain").alias("domains"), F.min("url").alias("url"))
        .orderBy(F.desc("rows"), "band_id", "band_hash")
        .collect()
    )
    return [
        f"band {r.band_id:>2} hash {r.band_hash:>20}: {r.rows:>6} pages, {r.domains:>5} sites, "
        f"e.g. {r.url}"
        for r in rows
    ]


def describe_clusters(result: DataFrame, kept: DataFrame, top: int) -> list[str]:
    """One line per large cluster: size, sites, the kept page's url (plan 4.4.3)."""
    largest = (
        result.where(F.col("duplicate_type").isNull() & (F.col("cluster_size") > 1))
        .orderBy(F.desc("cluster_size"), "cluster_id")
        .limit(top)
        .select("cluster_id", "cluster_size")
    )
    members = (
        result.select("doc_id", "cluster_id")
        .join(F.broadcast(largest), "cluster_id")
        .join(kept.select("doc_id", "url", "domain"), "doc_id")
    )
    rows = (
        members.groupBy("cluster_id", "cluster_size")
        .agg(
            F.countDistinct("domain").alias("domains"),
            F.max(F.when(F.col("doc_id") == F.col("cluster_id"), F.col("url"))).alias("url"),
        )
        .orderBy(F.desc("cluster_size"), "cluster_id")
        .collect()
    )
    return [f"{r.cluster_size:>6} pages, {r.domains:>5} sites, kept: {r.url}" for r in rows]


def dedup_metrics(result: DataFrame, languages: tuple[str, ...], mega: int) -> dict[str, float]:
    """The run's counts from one aggregation over the (persisted) dedup_v1 rows."""
    dup, lang, size = F.col("duplicate_type"), F.col("language"), F.col("cluster_size")
    kept = dup.isNull()
    aggregates = [
        F.count("*").alias("dedup_documents_in"),
        _count_if(kept).alias("dedup_documents_kept"),
        _count_if(dup == EXACT).alias("dedup_exact_duplicates"),
        _count_if(dup == NEAR).alias("dedup_near_duplicates"),
        *[_count_if(lang == lg).alias(f"dedup_documents_in_{lg}") for lg in languages],
        *[_count_if(kept & (lang == lg)).alias(f"dedup_documents_kept_{lg}") for lg in languages],
        *[
            _count_if((dup == EXACT) & (lang == lg)).alias(f"dedup_exact_duplicates_{lg}")
            for lg in languages
        ],
        *[
            _count_if((dup == NEAR) & (lang == lg)).alias(f"dedup_near_duplicates_{lg}")
            for lg in languages
        ],
        # One kept row per cluster: the cluster size distribution (plan 4.4.3).
        _count_if(kept & (size > 1)).alias("dedup_clusters"),
        _count_if(kept & (size == 2)).alias("dedup_clusters_size_2"),
        _count_if(kept & size.between(3, 10)).alias("dedup_clusters_size_3_10"),
        _count_if(kept & size.between(11, 100)).alias("dedup_clusters_size_11_100"),
        _count_if(kept & size.between(101, mega)).alias(f"dedup_clusters_size_101_{mega}"),
        _count_if(kept & (size > mega)).alias(f"dedup_clusters_over_{mega}"),
        F.max(size).alias("dedup_cluster_max_size"),
        # Checked before the write, like silver's contract (S3-09).
        null_count(DEDUP_V1).alias("dedup_null_violations"),
        _count_if(~kept & ~F.coalesce(dup.isin(*DUPLICATE_TYPES), F.lit(False))).alias(
            "dedup_unknown_duplicate_type"
        ),
    ]
    totals = result.agg(*aggregates).first()
    assert totals is not None
    metrics = {k: float(v or 0) for k, v in totals.asDict().items()}
    documents = metrics["dedup_documents_in"] or 1.0
    metrics["dedup_exact_dup_rate"] = round(metrics["dedup_exact_duplicates"] / documents, 4)
    metrics["dedup_near_dup_rate"] = round(metrics["dedup_near_duplicates"] / documents, 4)
    metrics["dedup_dup_rate"] = round(
        (metrics["dedup_exact_duplicates"] + metrics["dedup_near_duplicates"]) / documents, 4
    )
    return metrics


def run_dedup(
    spark: SparkSession, kept: DataFrame, output: str, params: DedupParams
) -> dict[str, float]:
    """Deduplicate one crawl's kept pages (silver_v1 rows) and write dedup_v1 to `output`
    (a crawl's folder, overwritten). Returns the run's metrics."""
    minhash, lsh = params.minhash, params.lsh
    if lsh.bands * lsh.rows != minhash.num_perm:
        raise ValueError(f"{lsh.bands} bands x {lsh.rows} rows != {minhash.num_perm} values")
    metrics: dict[str, float] = {
        "dedup_shingle_size": minhash.shingle_size,
        "dedup_num_perm": minhash.num_perm,
        "dedup_bands": lsh.bands,
        "dedup_rows": lsh.rows,
        "dedup_threshold": round(lsh.threshold, 4),
    }

    # Every page once: exact hash (JVM) and signature (one Python pass). The text is not
    # kept: everything after this reads ~1 KB per page from the executors' disks.
    with step(spark, PREFIX, "signatures", metrics):
        pages = add_signatures(
            kept.select(
                "doc_id",
                "language",
                "quality_score",
                "fetch_date",
                "text",
                exact_hash(F.col("text")).alias("exact_hash"),
            ),
            minhash,
        ).drop("text")
        pages = pages.persist(StorageLevel.DISK_ONLY)
        pages.count()

    # 3a: each text's best page; the rest are exact duplicates and never enter the join.
    with step(spark, PREFIX, "exact", metrics):
        docs = exact_representatives(
            pages.select("doc_id", "language", "quality_score", "fetch_date", "exact_hash")
        ).drop("exact_hash")
        docs = docs.persist(StorageLevel.DISK_ONLY)
        docs.count()

    # 3d: the exact representatives' band rows, and how they fill the buckets.
    with step(spark, PREFIX, "bands", metrics):
        representatives = docs.where(F.col("doc_id") == F.col("exact_rep")).select("doc_id")
        signed = pages.join(F.broadcast(representatives), "doc_id", "left_semi")
        bands = band_rows(signed.select("doc_id", "signature"), lsh).persist(StorageLevel.DISK_ONLY)
        key_metrics, largest = bucket_sizes(bands, TOP_KEYS)
        metrics.update(key_metrics)
        metrics["dedup_documents_signed"] = key_metrics["dedup_band_rows"] / lsh.bands
        for line in describe_keys(largest, bands, kept):
            print(f"run_dedup: largest bucket: {line}")

    # The band self-join: every two pages sharing a bucket (plan 4.3.2).
    with step(spark, PREFIX, "pairs", metrics):
        pairs = candidate_pairs(bands).persist(StorageLevel.DISK_ONLY)
        metrics["dedup_candidate_pairs"] = pairs.count()

    # Each candidate checked on its whole signature (trap S4: no chaining via weak links).
    with step(spark, PREFIX, "verify", metrics):
        edges = verified_pairs(
            pairs, pages.select("doc_id", "signature"), minhash.num_perm, lsh.threshold
        ).persist(StorageLevel.DISK_ONLY)
        verified = edges.agg(F.count("*").alias("n"), F.avg("similarity").alias("mean")).first()
        assert verified is not None
        metrics["dedup_verified_pairs"] = verified.n
        metrics["dedup_verified_mean_similarity"] = round(verified.mean or 0.0, 4)

    # 3e: connected components over the verified pairs.
    with step(spark, PREFIX, "components", metrics):
        components, rounds, converged = connected_components(
            edges.select("doc_a", "doc_b"), params.max_rounds
        )
        metrics["dedup_components_rounds"] = rounds
        metrics["dedup_components_converged"] = int(converged)
        if not converged:
            print(f"run_dedup: WARNING components not converged after {rounds} rounds")

    # 3f: one page per cluster; the contract's shape; checked, then written.
    with step(spark, PREFIX, "write", metrics):
        result = (
            assign_clusters(docs, components)
            .withColumn("schema_version", F.lit(DEDUP_SCHEMA_VERSION))
            .withColumn("pipeline_version", F.lit(PIPELINE_VERSION))
            .select(*[F.col(f.name).cast(f.dataType) for f in DEDUP_V1.fields])
            .persist(StorageLevel.DISK_ONLY)
        )
        assert_schema(result, DEDUP_V1, check_nullability=False)  # names, types, order
        languages = tuple(LanguageRules().min_conf)
        metrics.update(dedup_metrics(result, languages, params.mega_cluster))
        if metrics["dedup_null_violations"] or metrics["dedup_unknown_duplicate_type"]:
            raise RuntimeError(
                f"{metrics['dedup_null_violations']:.0f} nulls in non-null columns, "
                f"{metrics['dedup_unknown_duplicate_type']:.0f} unknown duplicate types"
            )
        for line in describe_clusters(result, kept, TOP_CLUSTERS):
            print(f"run_dedup: largest cluster: {line}")
        # About 40 bytes per page: one file per crawl (no small files).
        result.coalesce(1).write.mode("overwrite").parquet(output)
        metrics["dedup_documents_written"] = spark.read.schema(DEDUP_V1).parquet(output).count()

    for df in (result, edges, pairs, bands, docs, pages):
        df.unpersist()
    return metrics


def read_kept(spark: SparkSession, crawl_id: str, *, dev: bool) -> DataFrame:
    """The crawl's non-rejected silver_v1 pages (T14). The filters are on partition
    columns, so only the kept folders of this crawl are read."""
    return (
        spark.read.schema(SILVER_V1)  # declared, never inferred
        .parquet(silver_v1_path(dev=dev))
        .where((F.col("crawl_id") == crawl_id) & (F.col("quality_tier") != REJECTED))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--dev", action="store_true", help="the dev run's 5 segments")
    parser.add_argument("--run-id", default=None, help="Airflow's run_id; generated if omitted")
    parser.add_argument(
        "--keep-run-open",
        action="store_true",
        help="not the DAG's last task: leave the run open for the task after it",
    )
    parser.add_argument("--bands", type=int, default=LshParams.bands)
    parser.add_argument("--rows", type=int, default=LshParams.rows)
    parser.add_argument(
        "--spark-conf",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a Spark SQL setting for tuning experiments (Sprint 5); the DAG passes none",
    )
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)
    run_id = args.run_id or default_run_id()
    params = DedupParams(lsh=LshParams(bands=args.bands, rows=args.rows))
    output = dedup_path(args.crawl_id, dev=args.dev)

    with connect() as conn:
        start_run(conn, run_id, args.crawl_id, dev_mode=args.dev)
    print(f"run_dedup: {args.crawl_id}{' (dev)' if args.dev else ''} -> {output}, {params}")

    started = time.monotonic()
    spark = get_session(f"run_dedup {args.crawl_id}{' dev' if args.dev else ''}")
    try:
        for setting in args.spark_conf:
            key, _, value = setting.partition("=")
            spark.conf.set(key, value)
        metrics = run_dedup(spark, read_kept(spark, args.crawl_id, dev=args.dev), output, params)
        metrics["dedup_spark_shuffle_partitions"] = float(
            spark.conf.get("spark.sql.shuffle.partitions") or 0
        )
        metrics["dedup_spark_aqe"] = float(spark.conf.get("spark.sql.adaptive.enabled") == "true")
        metrics.update(step_metrics(spark, PREFIX))
    except Exception:
        with connect() as conn:
            finish_run(conn, run_id, "failed")
        raise
    finally:
        spark.stop()
    metrics["dedup_duration_seconds"] = round(time.monotonic() - started, 1)

    with connect() as conn:
        emit(conn, run_id, args.crawl_id, PREFIX, metrics)
        if not args.keep_run_open:
            finish_run(conn, run_id, "success")
    for name, value in sorted(metrics.items()):
        print(f"run_dedup: {name} = {value:,}")


if __name__ == "__main__":
    main()
