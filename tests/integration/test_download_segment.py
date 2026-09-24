"""download_segment end to end: a local HTTP server as Common Crawl, the real MinIO as bronze."""

import gzip
import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from corpus.acquisition.manifest import Segment
from corpus.ingest.download import DownloadError, download_segment, http_session, staging_path
from corpus.io import bronze_object, bronze_tmp_object
from corpus.store import delete_object, object_exists, object_size

pytestmark = pytest.mark.integration

CRAWL_ID = "CC-MAIN-9999-01"  # year 9999: no real run uses it
FILENAME = "file.warc.wet.gz"
SEGMENT = Segment("00042", f"crawl-data/CC-MAIN-9999-01/segments/1/wet/{FILENAME}")
RECORDS = [b"WARC/1.0\r\nWARC-Type: warcinfo\r\n\r\n", b"WARC/1.0\r\nWARC-Type: conversion\r\n\r\n"]
# Multi-member gzip, like a real WET file: one member per record.
WET_BYTES = b"".join(gzip.compress(record) for record in RECORDS)
DECOMPRESSED_SIZE = sum(len(record) for record in RECORDS)


@contextmanager
def crawl_server(body: bytes, etag: str | None = None) -> Iterator[str]:
    """Stand in for data.commoncrawl.org. ETag defaults to the MD5 of what is served."""
    served_etag = etag if etag is not None else f'"{hashlib.md5(body).hexdigest()}"'

    class Handler(BaseHTTPRequestHandler):
        def _send_headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", served_etag)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802
            self._send_headers()

        def do_GET(self) -> None:  # noqa: N802
            self._send_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{SEGMENT.path}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def clean_bronze() -> Iterator[None]:
    """Remove whatever the test wrote into bronze, before and after."""
    uris = (
        bronze_object(CRAWL_ID, 42, FILENAME),
        bronze_tmp_object(CRAWL_ID, 42, FILENAME),
    )
    for uri in uris:
        if object_exists(uri):
            delete_object(uri)
    yield
    for uri in uris:
        if object_exists(uri):
            delete_object(uri)


def test_a_good_segment_lands_in_bronze(tmp_path: Path) -> None:
    with crawl_server(WET_BYTES) as url:
        result = download_segment(
            http_session(backoff_factor=0),
            CRAWL_ID,
            SEGMENT,
            staging_dir=tmp_path,
            source_url=url,
        )

    final = bronze_object(CRAWL_ID, 42, FILENAME)
    assert result.final_uri == final
    assert result.size_bytes == len(WET_BYTES)
    assert result.decompressed_bytes == DECOMPRESSED_SIZE
    assert result.checksum_verified is True  # the ETag was a plain MD5 and matched
    assert object_exists(final) is True
    assert object_size(final) == len(WET_BYTES)
    assert object_exists(bronze_tmp_object(CRAWL_ID, 42, FILENAME)) is False  # no _tmp left
    assert staging_path(tmp_path, CRAWL_ID, SEGMENT).exists() is False  # local copy cleaned up


def test_a_truncated_file_never_reaches_bronze(tmp_path: Path) -> None:
    """The gzip trailer is missing, so validation fails after the download."""
    truncated = WET_BYTES[:-10]
    # The ETag matches what is served, so gzip validation is what must catch this.
    with (
        crawl_server(truncated) as url,
        pytest.raises(DownloadError, match="not a complete gzip file"),
    ):
        download_segment(
            http_session(backoff_factor=0),
            CRAWL_ID,
            SEGMENT,
            staging_dir=tmp_path,
            source_url=url,
        )

    assert object_exists(bronze_object(CRAWL_ID, 42, FILENAME)) is False
    assert object_exists(bronze_tmp_object(CRAWL_ID, 42, FILENAME)) is False
    assert staging_path(tmp_path, CRAWL_ID, SEGMENT).exists() is False  # untrusted, deleted


def test_a_checksum_mismatch_never_reaches_bronze(tmp_path: Path) -> None:
    """The server's ETag disagrees with the bytes: something altered them in transit."""
    with (
        crawl_server(WET_BYTES, etag='"0123456789abcdef0123456789abcdef"') as url,
        pytest.raises(DownloadError, match="checksum mismatch"),
    ):
        download_segment(
            http_session(backoff_factor=0),
            CRAWL_ID,
            SEGMENT,
            staging_dir=tmp_path,
            source_url=url,
        )
    assert object_exists(bronze_object(CRAWL_ID, 42, FILENAME)) is False


def test_a_compression_bomb_never_reaches_bronze(tmp_path: Path) -> None:
    bomb = gzip.compress(b"\0" * 2_000_000)
    with (
        crawl_server(bomb) as url,
        pytest.raises(DownloadError, match="decompressed size passed"),
    ):
        download_segment(
            http_session(backoff_factor=0),
            CRAWL_ID,
            SEGMENT,
            staging_dir=tmp_path,
            source_url=url,
            max_decompressed_bytes=1024,
        )
    assert object_exists(bronze_object(CRAWL_ID, 42, FILENAME)) is False
