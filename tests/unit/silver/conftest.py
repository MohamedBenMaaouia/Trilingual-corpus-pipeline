"""Fixtures shared by the silver tests."""

import gzip
import uuid
from collections.abc import Callable

import fasttext
import pytest
from pyspark.sql import SparkSession

from corpus.silver.language import MODEL_FILE

WetRecord = Callable[..., bytes]


def _wet_record(
    body: bytes, record_id: str | None = None, url: str = "http://example.com/page"
) -> bytes:
    """One hand-made conversion record as its own gzip member, like a real WET file."""
    record_id = record_id or f"<urn:uuid:{uuid.uuid4()}>"
    headers = (
        "WARC/1.0\r\n"
        "WARC-Type: conversion\r\n"
        f"WARC-Target-URI: {url}\r\n"
        "WARC-Date: 2026-09-04T12:00:00Z\r\n"
        f"WARC-Record-ID: {record_id}\r\n"
        "Content-Type: text/plain\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
    )
    return gzip.compress(headers.encode() + body + b"\r\n\r\n")


@pytest.fixture
def wet_record() -> WetRecord:
    """Build WET records by hand, to make each failure case deliberate and visible."""
    return _wet_record


TOY_TRAINING = [
    "__label__en the cat is on the table and the dog is in the garden",
    "__label__fr le chat est sur la table et le chien est dans le jardin",
    "__label__de die katze ist auf dem tisch und der hund ist im garten",
]


@pytest.fixture(scope="session")
def toy_model(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> None:
    """A toy fastText model (en, fr, de), registered with Spark as lid.176.bin (S3-04).

    Trained in 0.04 s, one thread, seed 42: the same model on every run. Once per test
    session: Spark refuses a second, different file under the same name.
    """
    folder = tmp_path_factory.mktemp("lid")
    training = folder / "train.txt"
    training.write_text("\n".join(TOY_TRAINING * 20) + "\n", encoding="utf-8")
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
