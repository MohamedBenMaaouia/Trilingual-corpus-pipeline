"""Sprint 1's headline artefact (task 1.7.1): kill the download, restart, lose nothing.

The download runs as a real subprocess and is killed with SIGKILL, so none of our
cleanup code runs: no exception handler, no finally block. That is the worst case,
equivalent to the machine losing power mid-upload.

Four assertions after the restart:
  1. all segments are in bronze;
  2. no _tmp object is left behind;
  3. no duplicates;
  4. segments finished before the kill were NOT fetched again (attempts still 1).
"""

import gzip
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from psycopg2.extensions import connection as Connection

from corpus.acquisition.control import SegmentControl
from corpus.acquisition.manifest import Segment
from corpus.io import bronze_object, bronze_tmp_object, split_s3a
from corpus.store import delete_object, object_exists, s3_client

pytestmark = pytest.mark.integration

SEGMENT_COUNT = 10
THREADS = 2
SECONDS_PER_SEGMENT = 0.5  # the server is deliberately slow, so there is time to kill it
RECORD = b"WARC/1.0\r\nWARC-Type: conversion\r\n\r\ntext\r\n\r\n"
BODY = b"".join(gzip.compress(RECORD) for _ in range(2))


def segments() -> list[Segment]:
    return [
        Segment(f"{i:05d}", f"crawl-data/x/segments/1/wet/file-{i:05d}.warc.wet.gz")
        for i in range(SEGMENT_COUNT)
    ]


@contextmanager
def slow_crawl_server() -> Iterator[str]:
    """Serves each segment after a delay, so the kill lands mid-run."""

    class Handler(BaseHTTPRequestHandler):
        def _send_headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.send_header("ETag", f'"{hashlib.md5(BODY).hexdigest()}"')
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802
            self._send_headers()

        def do_GET(self) -> None:  # noqa: N802
            time.sleep(SECONDS_PER_SEGMENT)
            self._send_headers()
            self.wfile.write(BODY)

        def log_message(self, *args: object) -> None:
            pass

    # Threading: a plain HTTPServer serves one request at a time, which would make the
    # parallel downloads queue behind each other instead of overlapping.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def start_download(crawl_id: str, base_url: str, staging_dir: Path) -> subprocess.Popen[bytes]:
    """The same command Airflow runs, as its own process so it can be killed.

    CORPUS_STAGING_DIR points at a writable temp folder: in the test container the
    repo (and so the default /opt/corpus/staging) is mounted read-only.
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "corpus.jobs.download",
            "--crawl-id",
            crawl_id,
            "--base-url",
            base_url,
            "--threads",
            str(THREADS),
            "--stale-lease-seconds",
            "1",  # a real restart happens seconds later, not 15 minutes later
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "CORPUS_STAGING_DIR": str(staging_dir)},
    )


def complete_segments(control: SegmentControl, crawl_id: str) -> dict[str, int]:
    """segment_id -> attempts, for segments already complete."""
    with control._conn.cursor() as cur:  # noqa: SLF001
        cur.execute(
            "SELECT segment_id, attempts FROM segments WHERE crawl_id = %s AND status = 'complete'",
            (crawl_id,),
        )
        return dict(cur.fetchall())


def bronze_keys(crawl_id: str) -> list[str]:
    """Every object stored under this crawl, including any _tmp leftovers."""
    bucket, prefix = split_s3a(bronze_object(crawl_id, 0, "x").rsplit("/segment=", 1)[0] + "/")
    pages = s3_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for page in pages for obj in page.get("Contents", [])]


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


def test_killing_the_download_loses_no_work(
    control: SegmentControl,
    conn: Connection,
    crawl_id: str,
    bronze_cleanup: None,
    tmp_path: Path,
) -> None:
    control.register_intent(crawl_id, segments())

    with slow_crawl_server() as base_url:
        process = start_download(crawl_id, base_url, tmp_path)
        try:
            # Wait until a few segments are done, then kill without warning.
            deadline = time.monotonic() + 60
            while len(complete_segments(control, crawl_id)) < 4:
                if process.poll() is not None:
                    output = process.stdout.read().decode() if process.stdout else ""
                    pytest.fail(f"the download ended before it could be killed:\n{output}")
                assert time.monotonic() < deadline, "no segment completed in time"
                time.sleep(0.05)
            os.kill(process.pid, signal.SIGKILL)
        finally:
            process.wait(timeout=30)

        finished_before_kill = complete_segments(control, crawl_id)
        assert 4 <= len(finished_before_kill) < SEGMENT_COUNT, "the kill must land mid-run"

        # The killed process left claims behind; the lease is 1 s, so wait it out.
        time.sleep(1.5)
        restarted = start_download(crawl_id, base_url, tmp_path)
        output = restarted.communicate(timeout=180)[0].decode()
        assert restarted.returncode == 0, output

    # 1. every segment is in bronze, and the control table agrees
    assert control.status_counts(crawl_id) == {"complete": SEGMENT_COUNT}

    # 2 and 3. one object per segment, no _tmp left behind, no duplicates
    keys = bronze_keys(crawl_id)
    assert len([key for key in keys if "_tmp" in key]) == 0
    assert len(keys) == SEGMENT_COUNT
    assert len(set(keys)) == len(keys)

    # 4. work finished before the kill was not repeated
    after = complete_segments(control, crawl_id)
    for segment_id, attempts_before in finished_before_kill.items():
        assert after[segment_id] == attempts_before == 1, (
            f"segment {segment_id} was fetched again after the restart"
        )

    # The numbers this run observed, so the artefact is evidence and not just a tick
    # (pytest shows this with -s, and on failure always).
    print(
        f"\nkill-and-restart proof: {len(finished_before_kill)} of {SEGMENT_COUNT} segments "
        f"were complete when SIGKILL landed "
        f"(ids {sorted(finished_before_kill)}), the restart finished the remaining "
        f"{SEGMENT_COUNT - len(finished_before_kill)}; "
        f"{len(keys)} objects in bronze, 0 _tmp, "
        f"attempts unchanged (=1) for every segment finished before the kill"
    )
