"""Least privilege per layer (DECISIONS S2-02), checked against the real MinIO.

Two users. Processing (Spark) reads bronze and can never write it. Ingest (the
downloader, corpus.store) writes bronze and can reach nothing else. These tests are
the proof that a bug in a Spark job cannot damage bronze (invariant 1).
"""

import uuid
from collections.abc import Callable, Iterator
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError

from corpus.config import get_settings
from corpus.config.local import LocalSettings
from corpus.store import s3_client as ingest_client

pytestmark = pytest.mark.integration

BRONZE = "corpus-bronze"
DOWNSTREAM = ("corpus-silver", "corpus-gold", "corpus-meta")


@pytest.fixture(scope="module")
def processing() -> Any:
    """A client with Spark's keys. Built here, not in corpus.store: the package never
    needs one, because Spark reaches MinIO through S3A, not boto3."""
    settings = get_settings()
    assert isinstance(settings, LocalSettings)
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key.get_secret_value(),
    )


@pytest.fixture
def bronze_key() -> Iterator[str]:
    """A small object in bronze, written by the ingest user and removed afterwards."""
    key = f"_test/{uuid.uuid4().hex}"
    ingest_client().put_object(Bucket=BRONZE, Key=key, Body=b"bronze bytes")
    yield key
    ingest_client().delete_object(Bucket=BRONZE, Key=key)


def assert_denied(call: Callable[[], object]) -> None:
    with pytest.raises(ClientError) as err:
        call()
    assert err.value.response["Error"]["Code"] == "AccessDenied"


def test_processing_user_reads_bronze(processing: Any, bronze_key: str) -> None:
    body = processing.get_object(Bucket=BRONZE, Key=bronze_key)["Body"].read()
    assert body == b"bronze bytes"


def test_processing_user_cannot_write_bronze(processing: Any) -> None:
    key = f"_test/{uuid.uuid4().hex}"
    assert_denied(lambda: processing.put_object(Bucket=BRONZE, Key=key, Body=b"x"))


def test_processing_user_cannot_delete_from_bronze(processing: Any, bronze_key: str) -> None:
    assert_denied(lambda: processing.delete_object(Bucket=BRONZE, Key=bronze_key))
    ingest_client().head_object(Bucket=BRONZE, Key=bronze_key)  # still there


@pytest.mark.parametrize("bucket", DOWNSTREAM)
def test_processing_user_writes_downstream_layers(processing: Any, bucket: str) -> None:
    key = f"_test/{uuid.uuid4().hex}"
    processing.put_object(Bucket=bucket, Key=key, Body=b"x")
    processing.delete_object(Bucket=bucket, Key=key)


@pytest.mark.parametrize("bucket", DOWNSTREAM)
def test_ingest_user_cannot_reach_other_layers(bucket: str) -> None:
    key = f"_test/{uuid.uuid4().hex}"
    assert_denied(lambda: ingest_client().put_object(Bucket=bucket, Key=key, Body=b"x"))
    assert_denied(lambda: ingest_client().list_objects_v2(Bucket=bucket, MaxKeys=1))
