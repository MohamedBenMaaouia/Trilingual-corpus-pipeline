"""Fixtures shared by every test. pytest finds this file automatically."""

import json
import random
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fasttext
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructField, StructType

from corpus.schemas.dedup_v1 import DEDUP_V1
from corpus.schemas.gold_v1 import GOLD_V1
from corpus.schemas.silver_v1 import SILVER_V1
from corpus.session import DELTA_CONF
from corpus.silver.language import MODEL_FILE

CRAWL = "CC-MAIN-2026-39"
FETCHED = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)

# Batch = a function: a list of rows, each given as its changes from a valid default.
Batch = Callable[[list[dict[str, Any]]], DataFrame]


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """One real, in-process Spark for the whole test run (starting one costs seconds).

    Deliberately not corpus.session.get_session(): tests must not need the cluster
    or MinIO. local[2] = driver + 2 worker threads in this one process.
    """
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("corpus-tests")
        # Default 200 shuffle partitions would make every tiny test run 200 tasks.
        .config("spark.sql.shuffle.partitions", "4")
        # Delta's VACUUM lists the table with this many tasks (default 10,000): ~30 s of
        # empty tasks per VACUUM on 2 threads (S6-06). The gold job sets 16.
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "4")
        .config("spark.ui.enabled", "false")  # no web UI needed in tests
        .config("spark.sql.session.timeZone", "UTC")  # same results on every machine
    )
    for key, value in DELTA_CONF.items():
        builder = builder.config(key, value)
    session = builder.getOrCreate()
    yield session
    session.stop()


def _batch(
    spark: SparkSession, schema: StructType, defaults: Callable[[str], dict[str, Any]]
) -> Batch:
    # Every column nullable, as a Parquet read gives them, so a test can put a null
    # where the contract forbids one.
    nullable = StructType([StructField(f.name, f.dataType, True) for f in schema.fields])

    def build(rows: list[dict[str, Any]]) -> DataFrame:
        full = [defaults(r["doc_id"]) | r for r in rows]
        return spark.createDataFrame(
            [tuple(r[n] for n in schema.fieldNames()) for r in full], nullable
        )

    return build


def _silver_v1_row(doc_id: str) -> dict[str, Any]:
    """A valid silver_v1 row: a kept English page of CRAWL."""
    return {
        "doc_id": doc_id, "url": f"http://site-{doc_id}.example.com/page",
        "domain": "example.com", "crawl_id": CRAWL, "segment_id": "00000",
        "fetch_date": FETCHED, "content_length": 100, "cc_language": "eng", "language": "en",
        "language_detected": "en", "language_conf": 0.9, "text": f"A good page, {doc_id}.",
        "char_count": 12, "word_count": 3, "mean_word_length": 3.3,
        "symbol_to_word_ratio": 0.0, "stopword_ratio": 0.3, "repeated_line_ratio": 0.0,
        "ellipsis_line_ratio": 0.0, "boilerplate_lines_removed": 2, "quality_tier": "high",
        "quality_score": 0.9, "reject_reasons": None, "pii_redacted": False,
        "pii_types": None, "schema_version": "silver_v1", "pipeline_version": "1",
    }  # fmt: skip


def _dedup_v1_row(doc_id: str) -> dict[str, Any]:
    """A valid dedup_v1 row: a page with no duplicate, which gold keeps."""
    return {
        "doc_id": doc_id, "language": "en", "cluster_id": doc_id, "cluster_size": 1,
        "duplicate_type": None, "schema_version": "dedup_v1", "pipeline_version": "1",
    }  # fmt: skip


def _gold_v1_row(doc_id: str) -> dict[str, Any]:
    """A valid gold_v1 row: an English high-tier page of CRAWL."""
    silver = _silver_v1_row(doc_id)
    gold = {name: silver[name] for name in GOLD_V1.fieldNames() if name in silver}
    return gold | {"cluster_size": 1, "schema_version": "gold_v1"}


@pytest.fixture
def silver_v1_batch(spark: SparkSession) -> Batch:
    """silver_v1 rows from their changes, e.g. [{"doc_id": "a"}, {"doc_id": "b", "text": ""}]."""
    return _batch(spark, SILVER_V1, _silver_v1_row)


@pytest.fixture
def dedup_v1_batch(spark: SparkSession) -> Batch:
    """dedup_v1 rows from their changes, e.g. [{"doc_id": "a", "duplicate_type": "exact"}]."""
    return _batch(spark, DEDUP_V1, _dedup_v1_row)


@pytest.fixture
def gold_v1_batch(spark: SparkSession) -> Batch:
    """gold_v1 rows from their changes, e.g. [{"doc_id": "a", "crawl_id": "CC-MAIN-2026-40"}]."""
    return _batch(spark, GOLD_V1, _gold_v1_row)


# --- The test language model ----------------------------------------------------------

VOCABULARY_FILE = Path(__file__).parent / "fixtures" / "lid_vocabulary.json"

TOY_TRAINING = [
    "__label__en the cat is on the table and the dog is in the garden",
    "__label__fr le chat est sur la table et le chien est dans le jardin",
    "__label__de die katze ist auf dem tisch und der hund ist im garten",
]


def vocabulary() -> dict[str, dict[str, list[str]]]:
    """Per language: its stopwords and words (tests/fixtures/lid_vocabulary.json)."""
    data: dict[str, Any] = json.loads(VOCABULARY_FILE.read_text(encoding="utf-8"))
    return {lang: words for lang, words in data.items() if not lang.startswith("_")}


def training_lines() -> list[str]:
    """The toy sentences (S3-04), then 60 seeded sentences per language of the vocabulary
    (S7-06: the golden pages are built from the same words, Arabic included)."""
    rng = random.Random(42)
    lines = TOY_TRAINING * 20
    for lang, words in vocabulary().items():
        pool = words["stopwords"] + words["words"]
        lines += [f"__label__{lang} " + " ".join(rng.choices(pool, k=12)) for _ in range(60)]
    return lines


@pytest.fixture(scope="session")
def toy_model(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> None:
    """A toy fastText model (en, fr, ar, de), registered with Spark as lid.176.bin (S3-04).

    Trained in well under a second, one thread, seed 42: the same model on every run.
    Once per test session: Spark refuses a second, different file under the same name,
    so every test that needs language ID shares this one.
    """
    folder = tmp_path_factory.mktemp("lid")
    training = folder / "train.txt"
    training.write_text("\n".join(training_lines()) + "\n", encoding="utf-8")
    model = fasttext.train_supervised(
        input=str(training),
        epoch=25,
        lr=1.0,
        dim=8,
        minn=0,
        maxn=0,
        bucket=0,
        thread=1,
        seed=42,
        verbose=0,
    )
    path = folder / MODEL_FILE
    model.save_model(str(path))
    spark.sparkContext.addFile(str(path))
