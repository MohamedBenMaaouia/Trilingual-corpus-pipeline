"""run_download: the whole loop against real Postgres and real MinIO.

The HTTP server is local, so nothing touches the internet. Story 1.7 adds the proof
with a process that is really killed; this covers the loop's own guarantees.
"""

import gzip
import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from corpus.acquisition.control import SegmentControl
from corpus.acquisition.manifest import Segment
from corpus.ingest.download import DownloadError, http_session, run_download
from corpus.io import bronze_object, bronze_tmp_object
from corpus.store import delete_object, object_exists

pytestmark = pytest.mark.integration

RECORD = b"WARC/1.0\r\nWARC-Type: conversion\r\n\r\nsome text\r\n\r\n"
GOOD = b"".join(gzip.compress(RECORD) for _ in range(2))  # multi-member, like a WET file
SEGMENT_COUNT = 10


def segments() -> list[Segment]:
    return [
        Segment(f"{i:05d}", f"crawl-data/x/segments/1/wet/file-{i:05d}.warc.wet.gz")
        for i in range(SEGMENT_COUNT)
    ]


@contextmanager
def crawl_server(broken: set[str]) -> Iterator[str]:
    """Serve GOOD for every segment, except the ids in `broken`, which get truncated bytes."""

    def body_for(path: str) -> bytes:
        segment_id = path.split("file-")[1].split(".")[0]
        return GOOD[:-8] if segment_id in broken else GOOD

    class Handler(BaseHTTPRequestHandler):
        def _send_headers(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", f'"{hashlib.md5(body).hexdigest()}"')
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802
            self._send_headers(body_for(self.path))

        def do_GET(self) -> None:  # noqa: N802
            body = body_for(self.path)
            self._send_headers(body)
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def bronze_cleanup(crawl_id: str) -> Iterator[None]:
    yield
    for segment in segments():
        name = segment.path.rsplit("/", 1)[-1]
        for uri in (
            bronze_object(crawl_id, int(segment.segment_id), name),
            bronze_tmp_object(crawl_id, int(segment.segment_id), name),
        ):
            if object_exists(uri):
                delete_object(uri)


def attempts_by_segment(control: SegmentControl, crawl_id: str) -> dict[str, int]:
    with control._conn.cursor() as cur:  # noqa: SLF001 - inspecting stored state
        cur.execute(
            "SELECT segment_id, attempts FROM segments WHERE crawl_id = %s ORDER BY segment_id",
            (crawl_id,),
        )
        return dict(cur.fetchall())


def test_a_clean_run_downloads_every_segment(
    control: SegmentControl, crawl_id: str, bronze_cleanup: None, tmp_path: Path
) -> None:
    control.register_intent(crawl_id, segments())
    with crawl_server(broken=set()) as base_url:
        summary = run_download(
            control,
            http_session(backoff_factor=0),
            crawl_id,
            staging_dir=tmp_path,
            base_url=base_url,
        )
    assert (summary.complete, summary.failed) == (SEGMENT_COUNT, 0)
    assert summary.bytes_downloaded == SEGMENT_COUNT * len(GOOD)
    assert control.status_counts(crawl_id) == {"complete": SEGMENT_COUNT}
    # Spot-check that the files really are in bronze under their published names.
    assert object_exists(bronze_object(crawl_id, 7, "file-00007.warc.wet.gz")) is True
    assert list(tmp_path.rglob("*.gz")) == []  # every staging file cleaned up


def test_a_second_run_downloads_nothing_again(
    control: SegmentControl, crawl_id: str, bronze_cleanup: None, tmp_path: Path
) -> None:
    """Restart-safety: finished work is never repeated, and attempts prove it."""
    control.register_intent(crawl_id, segments())
    with crawl_server(broken=set()) as base_url:
        run_download(
            control,
            http_session(backoff_factor=0),
            crawl_id,
            staging_dir=tmp_path,
            base_url=base_url,
        )
        after_first = attempts_by_segment(control, crawl_id)
        summary = run_download(
            control,
            http_session(backoff_factor=0),
            crawl_id,
            staging_dir=tmp_path,
            base_url=base_url,
        )
    assert (summary.complete, summary.failed) == (0, 0)  # nothing left to do
    assert attempts_by_segment(control, crawl_id) == after_first  # no segment tried again
    assert set(after_first.values()) == {1}


def test_one_bad_segment_does_not_stop_the_others(
    control: SegmentControl, crawl_id: str, bronze_cleanup: None, tmp_path: Path
) -> None:
    control.register_intent(crawl_id, segments())
    with crawl_server(broken={"00003"}) as base_url:
        summary = run_download(
            control,
            http_session(backoff_factor=0),
            crawl_id,
            staging_dir=tmp_path,
            base_url=base_url,
        )
    assert (summary.complete, summary.failed) == (SEGMENT_COUNT - 1, 1)
    assert control.status_counts(crawl_id) == {"complete": SEGMENT_COUNT - 1, "failed": 1}
    assert object_exists(bronze_object(crawl_id, 3, "file-00003.warc.wet.gz")) is False


def test_too_many_failures_stop_the_run(
    control: SegmentControl, crawl_id: str, bronze_cleanup: None, tmp_path: Path
) -> None:
    """More than 10% failing means something systemic: stop instead of hammering on."""
    control.register_intent(crawl_id, segments())
    with (
        crawl_server(broken={"00000", "00001", "00002"}) as base_url,
        pytest.raises(DownloadError, match="stopping the run"),
    ):
        run_download(
            control,
            http_session(backoff_factor=0),
            crawl_id,
            staging_dir=tmp_path,
            base_url=base_url,
        )
    counts = control.status_counts(crawl_id)
    assert counts.get("failed", 0) >= 2
    assert counts.get("pending", 0) > 0  # it stopped early, work is left
