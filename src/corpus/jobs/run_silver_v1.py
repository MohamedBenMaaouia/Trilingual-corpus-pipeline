"""Silver job, Sprint 3: the interim output (_stage1) -> silver_v1 (Stories 3.1-3.5).

A separate job from run_silver (user, 2026-10-04): it starts from Sprint 2's saved
output, so changing the language, quality or PII rules never re-reads or re-cleans
bronze.

Story 3.1 state: reads _stage1, detects languages, records the counts. Writes no data.

    python -m corpus.jobs.run_silver_v1 --crawl-id CC-MAIN-2026-39 [--dev] [--run-id ...]
"""

import argparse
import time

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from corpus.db import connect
from corpus.io import check_crawl_id, lid_model_path, silver_stage1_path
from corpus.metrics.emit import emit, finish_run, start_run
from corpus.run_id import default_run_id
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.session import get_session
from corpus.silver.language import (
    LANGUAGE_LOW_CONFIDENCE,
    LANGUAGE_NOT_TARGETED,
    OTHER,
    LanguageRules,
    detect_languages,
)


def language_metrics(labelled: DataFrame, rules: LanguageRules) -> dict[str, float]:
    """The language counts of one run, from documents carrying the language columns.

    One aggregation, so one pass over the documents: every Spark action on `labelled`
    re-runs fastText on every document, so the job must not ask several questions.
    What comes back to the driver is one row per (language, reason, empty) group, a
    few rows of numbers, never corpus data. Metric names start with lid_: run_silver
    writes under the same stage and run id, and must never be overwritten.
    """
    groups = (
        labelled.groupBy(
            "language",
            "language_reject_reason",
            F.col("language_detected").isNull().alias("empty"),
        )
        .agg(F.count("*").alias("documents"), F.sum("language_conf").alias("conf_sum"))
        .collect()
    )
    languages = [*rules.min_conf, OTHER]
    metrics: dict[str, float] = {"lid_documents": sum(g.documents for g in groups)}
    for language in languages:
        metrics[f"lid_documents_{language}"] = sum(
            g.documents for g in groups if g.language == language
        )
    metrics["lid_documents_empty"] = sum(g.documents for g in groups if g.empty)
    for reason in (LANGUAGE_NOT_TARGETED, LANGUAGE_LOW_CONFIDENCE):
        metrics[f"lid_rejected_{reason}"] = sum(
            g.documents for g in groups if g.language_reject_reason == reason
        )
    # How sure fastText was, on average, about the documents it put in each language
    # (low-confidence ones included). Not written when a language has no documents.
    for language in rules.min_conf:
        count = metrics[f"lid_documents_{language}"]
        if count:
            total = sum(g.conf_sum for g in groups if g.language == language)
            metrics[f"lid_mean_conf_{language}"] = round(total / count, 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--dev", action="store_true", help="the dev run's 5 segments")
    parser.add_argument("--run-id", default=None, help="Airflow's run_id; generated if omitted")
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)
    run_id = args.run_id or default_run_id()
    stage1_path = silver_stage1_path(args.crawl_id, dev=args.dev)

    with connect() as conn:
        start_run(conn, run_id, args.crawl_id, dev_mode=args.dev)
    print(f"run_silver_v1: {args.crawl_id}{' (dev)' if args.dev else ''}, reading {stage1_path}")

    started = time.monotonic()
    spark = get_session(f"run_silver_v1 {args.crawl_id}{' dev' if args.dev else ''}")
    try:
        # Every executor fetches its own copy of the model from MinIO, once; the
        # language step then opens it with SparkFiles.get (silver.language).
        spark.sparkContext.addFile(lid_model_path())
        # The declared schema, never inferred (invariant 3).
        docs = spark.read.schema(STAGE1_SCHEMA).parquet(stage1_path)
        rules = LanguageRules()
        metrics = language_metrics(detect_languages(docs, rules), rules)
    except Exception:
        with connect() as conn:
            finish_run(conn, run_id, "failed")
        raise
    finally:
        spark.stop()
    metrics["lid_duration_seconds"] = round(time.monotonic() - started, 1)

    with connect() as conn:
        emit(conn, run_id, args.crawl_id, "silver", metrics)
        finish_run(conn, run_id, "success")
    for name, value in metrics.items():
        print(f"run_silver_v1: {name} = {value:,}")


if __name__ == "__main__":
    main()
