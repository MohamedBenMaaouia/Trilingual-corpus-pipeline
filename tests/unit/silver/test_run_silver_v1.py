"""corpus.jobs.run_silver_v1: the Sprint 3 chain, its metrics and the D18 write (S3-09).

Runs locally on the fixture-sized data with the toy language model (conftest.py).
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.jobs.run_silver_v1 import run_silver_v1, silver_v1_metrics
from corpus.schemas.contract import assert_schema
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.schemas.silver_v1 import SILVER_V1
from corpus.silver.language import LanguageRules
from corpus.version import PIPELINE_VERSION

CRAWL, OTHER_CRAWL = "CC-MAIN-2026-39", "CC-MAIN-2026-40"
FETCHED = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)

# 70+ words of the toy model's English, each line different, with stopwords: kept.
ENGLISH = "\n".join(
    f"the cat is on the table and the dog is in the garden with the friend {i}." for i in range(6)
)
FRENCH_SHORT = "le chat est sur la table et le chien est dans le jardin"  # 12 words: rejected


def stage1(spark: SparkSession, rows: list[tuple[str, str, str]]) -> DataFrame:
    """_stage1 rows: (doc_id, crawl_id, text)."""
    data = [
        (d, f"http://e.com/{d}", "e.com", crawl, "00000", FETCHED, 100, "eng", text, 3)
        for d, crawl, text in rows
    ]
    return spark.createDataFrame(data, STAGE1_SCHEMA)


def folders(root: Path) -> set[str]:
    return {
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.glob("language=*/quality_tier=*/crawl_id=*")
    }


@pytest.mark.usefixtures("toy_model")
def test_the_chain_writes_the_contract_partitioned(spark: SparkSession, tmp_path: Path) -> None:
    root = tmp_path / "silver_v1"
    docs = stage1(
        spark,
        [
            ("en-good", CRAWL, ENGLISH + " Mail jean.dupont@example.fr please."),
            ("fr-short", CRAWL, FRENCH_SHORT),
            ("de", CRAWL, "die katze ist auf dem tisch und der hund ist im garten"),
            ("empty", CRAWL, ""),
        ],
    )
    glob = f"{root}/language=*/quality_tier=*/crawl_id={CRAWL}"
    metrics = run_silver_v1(spark, docs, str(root), glob, CRAWL)

    assert metrics["silver_v1_documents_written"] == metrics["silver_v1_documents_read"] == 4
    assert metrics["silver_v1_null_violations"] == 0
    assert metrics["silver_v1_empty_text_kept"] == 0
    assert metrics["pii_documents_email"] == 1

    written = spark.read.schema(SILVER_V1).parquet(str(root))
    assert_schema(written.select(*SILVER_V1.fieldNames()), SILVER_V1, check_nullability=False)
    rows = {r.doc_id: r for r in written.collect()}  # 4 test rows
    good = rows["en-good"]
    assert (good.language, good.reject_reasons) == ("en", None)
    assert good.quality_tier in ("high", "medium")
    assert "[EMAIL]" in good.text and "jean.dupont" not in good.text  # redacted
    assert (good.pii_redacted, good.pii_types) == (True, ["email"])
    assert good.pipeline_version == PIPELINE_VERSION and good.schema_version == "silver_v1"
    assert rows["fr-short"].reject_reasons[0] == "too_few_words"
    assert rows["de"].reject_reasons[0] == "language_not_targeted"
    assert rows["empty"].reject_reasons == ["too_few_words"]
    # Rejects are a partition of the same table (invariant 4).
    assert f"language=other/quality_tier=rejected/crawl_id={CRAWL}" in folders(root)


@pytest.mark.usefixtures("toy_model")
def test_a_rerun_replaces_its_crawl_only_and_leaves_no_stale_folder(
    spark: SparkSession, tmp_path: Path
) -> None:
    root = tmp_path / "silver_v1"

    def glob(crawl: str) -> str:
        return f"{root}/language=*/quality_tier=*/crawl_id={crawl}"

    first = stage1(spark, [("en", CRAWL, ENGLISH), ("de", CRAWL, "der hund ist im garten")])
    run_silver_v1(spark, first, str(root), glob(CRAWL), CRAWL)
    run_silver_v1(
        spark,
        stage1(spark, [("x", OTHER_CRAWL, ENGLISH)]),
        str(root),
        glob(OTHER_CRAWL),
        OTHER_CRAWL,
    )
    # Rerun the first crawl without its German page: its "other" folder must disappear
    # (dynamic partition overwrite would have kept it, D18).
    metrics = run_silver_v1(
        spark, stage1(spark, [("en", CRAWL, ENGLISH)]), str(root), glob(CRAWL), CRAWL
    )

    assert metrics["silver_v1_folders_replaced"] == 2  # the old en + other folders
    assert not any(f.startswith("language=other") and CRAWL in f for f in folders(root))
    table = spark.read.schema(SILVER_V1).parquet(str(root))
    counts = {r.crawl_id: r["count"] for r in table.groupBy("crawl_id").count().collect()}
    assert counts == {CRAWL: 1, OTHER_CRAWL: 1}  # the other crawl untouched, nothing doubled


def test_metrics_count_the_breakdown_and_the_gate_numbers(spark: SparkSession) -> None:
    rows = [
        ("a", "en", "en", 0.9, "kept text", "high", None, False, None),
        (
            "b",
            "en",
            "en",
            0.5,
            "x",
            "rejected",
            ["language_low_confidence", "too_few_words"],
            False,
            None,
        ),
        ("c", "other", "de", 0.9, "y", "rejected", ["language_not_targeted"], True, ["phone"]),
        ("d", "other", None, 0.0, "", "rejected", ["too_few_words"], False, None),
        ("e", "fr", "fr", 0.9, "", "medium", None, False, None),  # corrupted: kept but empty
        ("f", "xx", "xx", 0.9, "z", "excellent", None, False, None),  # corrupted vocabularies
    ]
    schema = (
        "doc_id string, language string, language_detected string, language_conf double, "
        "text string, quality_tier string, reject_reasons array<string>, pii_redacted boolean, "
        "pii_types array<string>"
    )
    silver = spark.createDataFrame(rows, schema)
    for field in SILVER_V1.fields:  # the other contract columns, filled
        if field.name not in silver.columns:
            silver = silver.withColumn(field.name, F.lit(1).cast(field.dataType))

    # One deliberate null in a non-null contract column (url of row "a").
    silver = silver.withColumn("url", F.when(F.col("doc_id") != "a", F.lit("http://e.com")))

    metrics = silver_v1_metrics(silver, LanguageRules())

    assert metrics["silver_v1_documents"] == 6
    assert metrics["silver_v1_rejected__too_few_words__en"] == 1
    assert metrics["silver_v1_rejected__too_few_words__other"] == 1
    assert metrics["silver_v1_rejected__too_few_words"] == 2
    assert metrics["lid_rejected_language_low_confidence"] == 1
    assert metrics["lid_documents_empty"] == 1
    assert metrics["pii_documents_phone"] == 1
    assert metrics["silver_v1_empty_text_kept"] == 1
    assert metrics["silver_v1_unknown_language"] == 1
    assert metrics["silver_v1_unknown_tier"] == 1
    assert metrics["silver_v1_null_violations"] == 1
