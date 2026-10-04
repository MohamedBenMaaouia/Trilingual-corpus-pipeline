"""Silver job, Sprint 3: the interim output (_stage1) -> silver_v1 (Stories 3.1-3.5).

A separate job from run_silver (user, S3-04): it starts from Sprint 2's saved output,
so changing the language, quality or PII rules never re-reads or re-cleans bronze.

    language ID -> PII redaction -> quality signals -> reasons, score, tier
    -> the contract's shape -> null count -> replace this crawl in silver_v1 (D18)

    python -m corpus.jobs.run_silver_v1 --crawl-id CC-MAIN-2026-39 [--dev] [--run-id ...]
        [--keep-run-open]
"""

import argparse
import time

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.db import connect
from corpus.io import (
    check_crawl_id,
    lid_model_path,
    silver_stage1_path,
    silver_v1_crawl_glob,
    silver_v1_path,
)
from corpus.metrics.emit import emit, finish_run, stamp_versions, start_run
from corpus.run_id import default_run_id
from corpus.schemas.contract import assert_schema, null_count
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.schemas.silver_v1 import LANGUAGES, QUALITY_TIERS, SCHEMA_VERSION, SILVER_V1
from corpus.session import get_session
from corpus.silver.finalize import to_silver_v1
from corpus.silver.language import (
    LANGUAGE_LOW_CONFIDENCE,
    LANGUAGE_NOT_TARGETED,
    LanguageRules,
    detect_languages,
)
from corpus.silver.pii import EMAIL, PHONE, redact_pii
from corpus.silver.quality import REJECTED, QualityRules, classify, quality_signals
from corpus.silver.write import replace_crawl
from corpus.version import PIPELINE_VERSION


def build_silver_v1(docs: DataFrame, languages: LanguageRules, quality: QualityRules) -> DataFrame:
    """The Sprint 3 chain on _stage1 rows, no I/O. Language ID sees the text as Sprint 2
    left it; the signals are computed on the redacted text, the one that is published."""
    labelled = detect_languages(docs, languages)
    redacted = redact_pii(labelled)
    return to_silver_v1(classify(quality_signals(redacted, quality), quality))


def _empty_if_null(column: str) -> Column:
    return F.coalesce(F.col(column), F.array().cast("array<string>"))


def silver_v1_metrics(silver: DataFrame, languages: LanguageRules) -> dict[str, float]:
    """The run's numbers, including what Gate B checks, from two aggregations over the
    (persisted) silver_v1 rows. A few rows of numbers come back, never corpus data.

    Names: lid_* (Story 3.1, unchanged), pii_*, silver_v1_*; and the rejection breakdown
    silver_v1_rejected__<reason>__<language> (spec 12.4: rejection rate by reason and
    language), plus silver_v1_rejected__<reason> over all languages.
    """
    reasons, types = _empty_if_null("reject_reasons"), _empty_if_null("pii_types")
    language, tier = F.col("language"), F.col("quality_tier")

    def count(condition: Column) -> Column:
        return F.sum(F.when(condition, 1).otherwise(0))

    aggregates = [
        F.count("*").alias("silver_v1_documents"),
        *[count(language == lang).alias(f"lid_documents_{lang}") for lang in LANGUAGES],
        *[count(tier == t).alias(f"silver_v1_tier_{t}") for t in QUALITY_TIERS],
        count(F.col("language_detected").isNull()).alias("lid_documents_empty"),
        *[
            count(F.array_contains(reasons, reason)).alias(f"lid_rejected_{reason}")
            for reason in (LANGUAGE_NOT_TARGETED, LANGUAGE_LOW_CONFIDENCE)
        ],
        *[
            F.avg(F.when(language == lang, F.col("language_conf"))).alias(f"lid_mean_conf_{lang}")
            for lang in languages.min_conf
        ],
        count(F.col("pii_redacted")).alias("pii_documents_redacted"),
        count(F.array_contains(types, EMAIL)).alias("pii_documents_email"),
        count(F.array_contains(types, PHONE)).alias("pii_documents_phone"),
        # What Gate B checks (S3-09): the contract's non-null columns hold no null; no
        # kept page is empty; every language and tier is in its vocabulary.
        null_count(SILVER_V1).alias("silver_v1_null_violations"),
        count((tier != REJECTED) & (F.coalesce(F.col("text"), F.lit("")) == "")).alias(
            "silver_v1_empty_text_kept"
        ),
        count(~F.coalesce(language.isin(*LANGUAGES), F.lit(False))).alias(
            "silver_v1_unknown_language"
        ),
        count(~F.coalesce(tier.isin(*QUALITY_TIERS), F.lit(False))).alias("silver_v1_unknown_tier"),
    ]
    totals = silver.agg(*aggregates).first()
    assert totals is not None
    metrics = {
        name: round(float(value), 4) for name, value in totals.asDict().items() if value is not None
    }

    breakdown = (
        silver.select("language", F.explode("reject_reasons").alias("reason"))
        .groupBy("language", "reason")
        .count()
        .collect()  # at most (reasons x languages) rows
    )
    for row in breakdown:
        metrics[f"silver_v1_rejected__{row.reason}__{row.language}"] = row["count"]
        total = f"silver_v1_rejected__{row.reason}"
        metrics[total] = metrics.get(total, 0) + row["count"]
    return metrics


def run_silver_v1(
    spark: SparkSession, docs: DataFrame, root: str, crawl_glob: str, crawl_id: str
) -> dict[str, float]:
    """Build, check and write one crawl's silver_v1; returns the run's metrics.

    The outputs are parameters so a test can run the whole job on local files; the job
    passes the corpus.io paths. The model must already be shipped (addFile).
    """
    languages, quality = LanguageRules(), QualityRules()
    # Persisted on the executors' disks: the checks and the write both read it, and
    # every read would otherwise run fastText and the PII step again (S2-11: DISK_ONLY).
    silver = build_silver_v1(docs, languages, quality).persist(StorageLevel.DISK_ONLY)
    assert_schema(silver, SILVER_V1, check_nullability=False)  # names, types, order
    metrics = silver_v1_metrics(silver, languages)
    if metrics["silver_v1_null_violations"]:
        # The contract's non-null promise (invariant 3): refuse to write.
        raise RuntimeError(f"{metrics['silver_v1_null_violations']:.0f} nulls in non-null columns")
    metrics["silver_v1_schema_ok"] = 1

    metrics["silver_v1_folders_replaced"] = replace_crawl(spark, silver, root, crawl_glob)
    # What landed, not what was meant to (the partition filter reads one crawl only).
    landed = spark.read.schema(SILVER_V1).parquet(root).where(F.col("crawl_id") == crawl_id)
    metrics["silver_v1_documents_read"] = metrics["silver_v1_documents"]
    metrics["silver_v1_documents_written"] = landed.count()
    silver.unpersist()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--dev", action="store_true", help="the dev run's 5 segments")
    parser.add_argument("--run-id", default=None, help="Airflow's run_id; generated if omitted")
    parser.add_argument(
        "--keep-run-open",
        action="store_true",
        help="not the DAG's last task: leave the run open for the gate after it",
    )
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)
    run_id = args.run_id or default_run_id()

    with connect() as conn:
        start_run(conn, run_id, args.crawl_id, dev_mode=args.dev)
        stamp_versions(conn, run_id, PIPELINE_VERSION, SCHEMA_VERSION)
    stage1_path = silver_stage1_path(args.crawl_id, dev=args.dev)
    root = silver_v1_path(dev=args.dev)
    print(f"run_silver_v1: {args.crawl_id}{' (dev)' if args.dev else ''}: {stage1_path} -> {root}")

    started = time.monotonic()
    spark = get_session(f"run_silver_v1 {args.crawl_id}{' dev' if args.dev else ''}")
    try:
        spark.sparkContext.addFile(lid_model_path())  # one copy per executor (S3-03)
        docs = spark.read.schema(STAGE1_SCHEMA).parquet(stage1_path)  # declared, never inferred
        crawl_glob = silver_v1_crawl_glob(args.crawl_id, dev=args.dev)
        metrics = run_silver_v1(spark, docs, root, crawl_glob, args.crawl_id)
    except Exception:
        with connect() as conn:
            finish_run(conn, run_id, "failed")
        raise
    finally:
        spark.stop()
    metrics["silver_v1_duration_seconds"] = round(time.monotonic() - started, 1)

    with connect() as conn:
        emit(conn, run_id, args.crawl_id, "silver", metrics)
        if not args.keep_run_open:
            finish_run(conn, run_id, "success")
    for name, value in sorted(metrics.items()):
        print(f"run_silver_v1: {name} = {value:,}")


if __name__ == "__main__":
    main()
