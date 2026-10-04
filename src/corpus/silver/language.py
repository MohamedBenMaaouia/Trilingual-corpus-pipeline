"""Silver step 2d: language identification with fastText lid.176 (Story 3.1, DECISIONS S3-01).

The model file (~126 MB) is shipped to every executor with SparkContext.addFile and
loaded once per Python worker process, never per row (plan risk: ~50x slower).
"""

import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import fasttext
import pandas as pd
from pyspark import SparkFiles
from pyspark.sql import DataFrame
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

# The name addFile gives the model on each executor; SparkFiles.get finds it there.
MODEL_FILE = "lid.176.bin"

# The language of a document we do not keep (D7: non-target languages stay in silver
# as 'other', with what fastText saw in language_detected).
OTHER = "other"

# Reason codes (controlled vocabulary, pipeline.md section 10).
LANGUAGE_NOT_TARGETED = "language_not_targeted"  # the spec's code: not en, fr or ar
LANGUAGE_LOW_CONFIDENCE = "language_low_confidence"  # a target language, but unsure (D10)


@dataclass(frozen=True)
class LanguageRules:
    """Which languages we keep, and how sure fastText must be for each.

    The keys are the target languages. Provisional values (D10, S3-01): FineWeb's
    English cut-off, "removing any document with en language score lower than 0.65"
    (HuggingFaceFW/fineweb dataset card), used for all three languages until the
    300-document labelled sample sets a value per language.
    """

    min_conf: Mapping[str, float] = field(
        default_factory=lambda: {"en": 0.65, "fr": 0.65, "ar": 0.65}
    )


class LanguageModel(Protocol):
    """What we use of a fastText model: predict() returns the top-k labels
    ("__label__en", ...) and their probabilities, best first."""

    def predict(self, text: str, k: int = 1) -> tuple[Sequence[str], Sequence[float]]: ...


# One model per Python worker process. Spark reuses its Python workers between tasks
# (spark.python.worker.reuse, on by default), so this survives from one task to the next.
_loaded: LanguageModel | None = None


def language_model() -> LanguageModel:
    """The worker's fastText model, loaded from the executor's copy on first use.

    Each load prints one line to the executor's stderr (Spark UI, Executors tab): the
    evidence that the model loads once per worker process, not once per row.
    """
    global _loaded
    if _loaded is None:
        started = time.monotonic()
        model: LanguageModel = fasttext.load_model(SparkFiles.get(MODEL_FILE))
        _loaded = model
        seconds = time.monotonic() - started
        print(
            f"language_model: loaded {MODEL_FILE} in {seconds:.1f} s, pid {os.getpid()}",
            file=sys.stderr,
        )
    return _loaded


class Prediction(NamedTuple):
    label: str  # fastText's code without "__label__": "en", "fr", "ar", "arz", ...
    conf: float  # its probability, 0..1


def predict_language(model: LanguageModel, text: str) -> Prediction | None:
    """fastText's best guess for a document, or None when there is no text to judge
    (a document emptied by boilerplate removal)."""
    flat = text.replace("\n", " ")  # fastText predicts one line: it rejects "\n"
    if not flat.strip():
        return None
    labels, probs = model.predict(flat, k=1)
    # fastText's softmax can round a certain answer to just above 1 (e.g. 1.00001);
    # capped so language_conf always stays a probability.
    return Prediction(labels[0].removeprefix("__label__"), min(float(probs[0]), 1.0))


class LanguageResult(NamedTuple):
    language: str  # en | fr | ar | other
    language_detected: str | None  # fastText's raw label; None for an empty document
    language_conf: float  # 0.0 for an empty document
    reject_reason: str | None  # None: language ID has no objection


def assign_language(prediction: Prediction | None, rules: LanguageRules) -> LanguageResult:
    """Turn fastText's guess into silver's language columns and, if any, a reject reason.

    - Not a target language (incl. "arz", Egyptian Arabic: MSA-only scope): 'other',
      rejected as language_not_targeted.
    - A target language below its threshold: keeps its language, so the rejection
      breakdown shows low confidence per language (what D10's values are tuned on),
      rejected as language_low_confidence.
    - No text: 'other' with no language reason; the quality rules reject it (Story 3.2).
    """
    if prediction is None:
        return LanguageResult(OTHER, None, 0.0, None)
    label, conf = prediction
    if label not in rules.min_conf:
        return LanguageResult(OTHER, label, conf, LANGUAGE_NOT_TARGETED)
    if conf < rules.min_conf[label]:
        return LanguageResult(label, label, conf, LANGUAGE_LOW_CONFIDENCE)
    return LanguageResult(label, label, conf, None)


# --- In Spark ----------------------------------------------------------------------------

# The four columns language ID adds, in LanguageResult's order (D7). All nullable:
# results coming back from pandas are always marked nullable, and Spark 3.5 rejects a
# pandas_udf whose declared type says otherwise ("Invalid schema from pandas_udf").
# language and language_conf are never null in practice; silver_v1's contract states
# that, and the write rebuilds the columns to it (Story 3.4, trap S3).
LANGUAGE_SCHEMA = StructType(
    [
        StructField("language", StringType(), True),
        StructField("language_detected", StringType(), True),
        StructField("language_conf", DoubleType(), True),
        StructField("language_reject_reason", StringType(), True),
    ]
)


def detect_languages(docs: DataFrame, rules: LanguageRules | None = None) -> DataFrame:
    """Add the four language columns to `docs` (which needs a `text` column).

    A pandas_udf on `text` only: Spark sends just the text to Python, in batches of
    rows (not one call per row), and gets four small values back per document. The
    other columns never leave Spark. mapInPandas would send whole rows to Python and
    the text back again. The model is loaded once per Python worker (language_model).
    """
    rules = rules or LanguageRules()

    # PySpark 3.5's own type hints for pandas_udf list no StructType return type (a
    # struct result is valid at runtime), so mypy matches no overload and sees the
    # decorator as untyped. Checked again with pandas-stubs installed (S3): still needed.
    @pandas_udf(LANGUAGE_SCHEMA)  # type: ignore[call-overload, untyped-decorator]
    def detect(texts: pd.Series) -> pd.DataFrame:
        model = language_model()
        results = [assign_language(predict_language(model, text or ""), rules) for text in texts]
        # Column types pinned, never guessed by pandas (S2-04: an all-empty column
        # guessed as float64 breaks Spark's Arrow transfer).
        return pd.DataFrame(results, columns=LANGUAGE_SCHEMA.fieldNames()).astype(
            {
                "language": object,
                "language_detected": object,
                "language_conf": "float64",
                "language_reject_reason": object,
            }
        )

    return docs.withColumn("_language", detect("text")).select("*", "_language.*").drop("_language")
