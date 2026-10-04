"""corpus.jobs.run_silver_v1.language_metrics: the counts a run records (Story 3.1).

Built from hand-made rows that already carry the language columns: no model needed.
"""

from pyspark.sql import SparkSession

from corpus.jobs.run_silver_v1 import language_metrics
from corpus.silver.language import LANGUAGE_SCHEMA, LanguageRules


def test_counts_per_language_reason_and_empty(spark: SparkSession) -> None:
    rows = [
        ("en", "en", 0.9, None),
        ("en", "en", 0.5, "language_low_confidence"),
        ("fr", "fr", 0.8, None),
        ("other", "de", 0.99, "language_not_targeted"),
        ("other", "arz", 0.7, "language_not_targeted"),
        ("other", None, 0.0, None),  # an empty document
    ]
    labelled = spark.createDataFrame(rows, LANGUAGE_SCHEMA)

    metrics = language_metrics(labelled, LanguageRules())

    assert metrics == {
        "lid_documents": 6,
        "lid_documents_en": 2,
        "lid_documents_fr": 1,
        "lid_documents_ar": 0,
        "lid_documents_other": 3,
        "lid_documents_empty": 1,
        "lid_rejected_language_not_targeted": 2,
        "lid_rejected_language_low_confidence": 1,
        "lid_mean_conf_en": 0.7,  # (0.9 + 0.5) / 2, low confidence included
        "lid_mean_conf_fr": 0.8,
        # no lid_mean_conf_ar: no Arabic document, nothing to average
    }
