"""corpus.ingest.download head/stream_to_file, against a local HTTP server."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from corpus.ingest.download import head, http_session, stream_to_file

BODY = bytes(range(256)) * 40  # 10 240 bytes of recognisable data


@contextmanager
def file_server(
    *, support_ranges: bool = True, honour_ranges: bool = True
) -> Iterator[tuple[str, list[str | None]]]:
    """Serve BODY. Yields (base_url, range_headers_seen_on_GET).

    support_ranges: advertise Accept-Ranges in HEAD.
    honour_ranges:  actually answer 206 to a Range request (some servers don't).
    """
    ranges_seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.send_header("ETag", '"abc123"')
            if support_ranges:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            requested = self.headers.get("Range")
            ranges_seen.append(requested)
            if requested and honour_ranges:
                start = int(requested.removeprefix("bytes=").split("-")[0])
                part = BODY[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                self.wfile.write(part)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/file.wet.gz", ranges_seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_head_reports_size_etag_and_resumability() -> None:
    with file_server() as (url, _):
        remote = head(http_session(backoff_factor=0), url)
    assert remote.size_bytes == len(BODY)
    assert remote.etag == '"abc123"'
    assert remote.resumable is True


def test_head_knows_when_a_server_cannot_resume() -> None:
    with file_server(support_ranges=False) as (url, _):
        assert head(http_session(backoff_factor=0), url).resumable is False


def test_stream_writes_the_whole_file(tmp_path: Path) -> None:
    destination = tmp_path / "sub" / "file.wet.gz"  # the parent folder does not exist yet
    with file_server() as (url, ranges):
        size = stream_to_file(http_session(backoff_factor=0), url, destination)
    assert size == len(BODY)
    assert destination.read_bytes() == BODY
    assert ranges == [None]  # nothing to resume, so no Range was asked for


def test_an_interrupted_download_resumes(tmp_path: Path) -> None:
    destination = tmp_path / "file.wet.gz"
    destination.write_bytes(BODY[:4096])  # what a killed process left behind
    with file_server() as (url, ranges):
        size = stream_to_file(http_session(backoff_factor=0), url, destination, resumable=True)
    assert ranges == ["bytes=4096-"]  # asked only for the missing part
    assert size == len(BODY)
    assert destination.read_bytes() == BODY


def test_a_server_that_ignores_the_range_restarts_cleanly(tmp_path: Path) -> None:
    """A 200 answer to a Range request means the whole file is coming: start over,
    otherwise the partial bytes and the full body would be spliced together."""
    destination = tmp_path / "file.wet.gz"
    destination.write_bytes(BODY[:4096])
    with file_server(honour_ranges=False) as (url, _):
        size = stream_to_file(http_session(backoff_factor=0), url, destination, resumable=True)
    assert size == len(BODY)
    assert destination.read_bytes() == BODY  # not 4096 bytes longer


def test_a_partial_file_is_discarded_when_the_server_cannot_resume(tmp_path: Path) -> None:
    destination = tmp_path / "file.wet.gz"
    destination.write_bytes(b"junk from an older attempt")
    with file_server() as (url, ranges):
        size = stream_to_file(http_session(backoff_factor=0), url, destination, resumable=False)
    assert ranges == [None]
    assert size == len(BODY)
    assert destination.read_bytes() == BODY
