"""Object storage access: the only place an S3 client is built.

Mirrors corpus.db for Postgres. Keeping boto3 in one file is what makes Sprint 10
cheap: on Azure the storage protocol changes here and nowhere else.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from corpus.config import get_settings
from corpus.config.local import LocalSettings
from corpus.io import split_s3a


@lru_cache(maxsize=1)
def s3_client() -> Any:
    """One client per process; boto3 clients are meant to be reused."""
    settings = get_settings()
    if not isinstance(settings, LocalSettings):
        raise NotImplementedError("object storage on Azure (abfss://) arrives in Sprint 10")
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,  # MinIO instead of AWS
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key.get_secret_value(),
    )


def put_file(local_path: Path, uri: str) -> None:
    """Upload a local file to an s3a:// location (multipart automatically if large)."""
    bucket, key = split_s3a(uri)
    s3_client().upload_file(str(local_path), bucket, key)


def promote(temp_uri: str, final_uri: str) -> None:
    """Make a validated upload visible under its real name, then remove the temp copy.

    The copy happens inside the server: no bytes travel, and the final object either
    exists complete or not at all. Nothing is ever visible half-written (T7).
    """
    source_bucket, source_key = split_s3a(temp_uri)
    target_bucket, target_key = split_s3a(final_uri)
    s3_client().copy_object(
        Bucket=target_bucket,
        Key=target_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
    )
    s3_client().delete_object(Bucket=source_bucket, Key=source_key)


def object_exists(uri: str) -> bool:
    bucket, key = split_s3a(uri)
    try:
        s3_client().head_object(Bucket=bucket, Key=key)
    except ClientError as err:
        if err.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise  # a real problem: credentials, network, a missing bucket
    return True


def object_size(uri: str) -> int:
    bucket, key = split_s3a(uri)
    response = s3_client().head_object(Bucket=bucket, Key=key)
    return int(response["ContentLength"])


def delete_object(uri: str) -> None:
    bucket, key = split_s3a(uri)
    s3_client().delete_object(Bucket=bucket, Key=key)
