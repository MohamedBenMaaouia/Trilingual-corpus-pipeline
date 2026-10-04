"""corpus.silver.language.detect_languages in Spark (Story 3.1, DECISIONS S3-01).

The real model (126 MB) never enters the repository, and Spark runs the language step
in separate Python worker processes, where a fake swapped in by the test would not
reach. So the tests use a toy fastText model (conftest.py: 3 languages, 60 sentences,
~2 KB) shipped with addFile under the real file name, exactly as the job ships
lid.176.bin. What is tested is the mechanism, not fastText's accuracy.
"""

import pytest
from pyspark.sql import DataFrame, SparkSession

from corpus.silver.language import (
    LANGUAGE_NOT_TARGETED,
    LANGUAGE_SCHEMA,
    detect_languages,
)


def documents(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            ("d-en", "the dog is on the table\nand the cat is in the garden"),
            ("d-fr", "le chien est sur la table"),
            ("d-de", "der hund ist im garten"),
            ("d-empty", ""),  # emptied by boilerplate removal (S2-07)
        ],
        "doc_id string, text string",
    )


@pytest.mark.usefixtures("toy_model")
def test_language_columns_per_document(spark: SparkSession) -> None:
    rows = {
        r.doc_id: r
        for r in detect_languages(documents(spark)).collect()  # 4 test rows, not corpus data
    }

    assert (rows["d-en"].language, rows["d-en"].language_reject_reason) == ("en", None)
    assert (rows["d-fr"].language, rows["d-fr"].language_reject_reason) == ("fr", None)
    assert rows["d-en"].language_conf >= 0.65 and rows["d-fr"].language_conf >= 0.65
    # German is not a target language: kept as 'other', with what the model saw.
    assert (rows["d-de"].language, rows["d-de"].language_detected) == ("other", "de")
    assert rows["d-de"].language_reject_reason == LANGUAGE_NOT_TARGETED
    # Empty text never reaches the model.
    empty = rows["d-empty"]
    assert (empty.language, empty.language_detected, empty.language_conf) == ("other", None, 0.0)
    assert empty.language_reject_reason is None


@pytest.mark.usefixtures("toy_model")
def test_input_columns_are_kept_and_four_added(spark: SparkSession) -> None:
    docs = documents(spark)
    result = detect_languages(docs)
    assert result.columns == docs.columns + LANGUAGE_SCHEMA.fieldNames()
    texts = {r.doc_id: r.text for r in result.select("doc_id", "text").collect()}
    assert texts["d-en"] == "the dog is on the table\nand the cat is in the garden"  # unchanged


@pytest.mark.usefixtures("toy_model")
def test_python_runs_once_per_document_not_once_per_column(spark: SparkSession) -> None:
    # Four columns come out of one Python call. If Spark copied the call into each of
    # the four columns, every document would go through fastText four times.
    plan = detect_languages(documents(spark))._jdf.queryExecution().executedPlan().toString()
    assert plan.count("ArrowEvalPython") == 1
