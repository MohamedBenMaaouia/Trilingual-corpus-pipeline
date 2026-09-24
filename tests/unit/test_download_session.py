"""corpus.ingest.download.http_session: retries, backoff scope and courtesy header.

A tiny HTTP server runs inside the test, so the retry behaviour is exercised for
real without touching the network. backoff_factor=0 keeps it instant.
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from corpus.ingest.download import USER_AGENT, http_session


@contextmanager
def flaky_server(statuses: Sequence[int]) -> Iterator[tuple[str, list[str]]]:
    """Serve the given statuses in order, then 200. Yields (base_url, user_agents_seen)."""
    remaining = list(statuses)
    seen_agents: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
            seen_agents.append(self.headers.get("User-Agent", ""))
            status = remaining.pop(0) if remaining else 200
            body = b"ok"
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass  # keep the test output clean

    server = HTTPServer(("127.0.0.1", 0), Handler)  # port 0: the OS picks a free one
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen_agents
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_busy_server_is_retried_until_it_answers() -> None:
    with flaky_server([503, 503]) as (url, agents):
        response = http_session(backoff_factor=0).get(f"{url}/wet.paths.gz")
        assert response.status_code == 200
        assert len(agents) == 3  # two refusals, then the successful attempt


def test_retries_stop_at_the_cap_and_hand_back_the_last_response() -> None:
    """raise_on_status=False: after the cap, the caller decides, via raise_for_status()."""
    with flaky_server([503] * 20) as (url, agents):
        response = http_session(total_retries=3, backoff_factor=0).get(url)
        assert len(agents) == 4  # the first attempt plus 3 retries
        assert response.status_code == 503
        with pytest.raises(requests.exceptions.HTTPError):
            response.raise_for_status()


def test_a_missing_file_is_not_retried() -> None:
    """404 means the file is not there; retrying wastes time and is impolite."""
    with flaky_server([404]) as (url, agents):
        response = http_session(backoff_factor=0).get(url)
        assert response.status_code == 404
        assert len(agents) == 1


def test_requests_identify_this_project() -> None:
    with flaky_server([]) as (url, agents):
        http_session(backoff_factor=0).get(url)
        assert agents == [USER_AGENT]
