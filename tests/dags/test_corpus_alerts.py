"""dags/corpus_alerts.py: the failure alert (Story 7.1.4, DECISIONS S7-03).

The module lives beside the DAG (Airflow's Python imports it, not ours), so it is loaded
here from its file. Standard library only, so these tests need no Airflow.
"""

import importlib.util
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

SOURCE = Path(__file__).parents[2] / "dags" / "corpus_alerts.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("corpus_alerts", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


alerts = _load()


class Capture(BaseHTTPRequestHandler):
    """A stand-in webhook: keeps every body it receives."""

    received: list[dict[str, Any]] = []
    status = 204

    def do_POST(self) -> None:  # noqa: N802 (the name http.server calls)
        length = int(self.headers["Content-Length"])
        Capture.received.append(json.loads(self.rfile.read(length)))
        self.send_response(Capture.status)
        self.end_headers()

    def log_message(self, *args: Any) -> None:  # keep the test output quiet
        pass


@pytest.fixture
def webhook() -> Iterator[str]:
    Capture.received, Capture.status = [], 204
    server = HTTPServer(("127.0.0.1", 0), Capture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/hook"
    server.shutdown()


def test_discord_and_slack_get_the_body_they_expect() -> None:
    assert alerts.payload("https://discord.com/api/webhooks/1/abc", "hi") == {"content": "hi"}
    assert alerts.payload("https://hooks.slack.com/services/T/B/x", "hi") == {"text": "hi"}
    assert len(alerts.payload("https://discord.com/x", "a" * 5_000)["content"]) == 1_900


def test_the_message_says_which_run_which_task_and_why() -> None:
    text = alerts.failure_message(
        dag_id="corpus_monthly",
        run_id="manual__2026-10-06",
        crawl_id="CC-MAIN-2026-39",
        task_id="gate_b",
        try_number=1,
        reason="x" * 1_000 + "the real cause",
        log_url="http://localhost:8090/log",
    )
    assert "task `gate_b` failed" in text and "crawl CC-MAIN-2026-39" in text
    assert text.count("x") <= 600 and text.endswith("log: http://localhost:8090/log")
    assert "the real cause" in text  # the end of a long error is what explains it


def test_a_task_failure_is_posted(webhook: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(alerts.WEBHOOK_ENV, webhook)
    ti = SimpleNamespace(
        dag_id="corpus_monthly", task_id="silver", try_number=2, run_id="r1", log_url=None
    )
    context = {
        "ti": ti,
        "dag_run": SimpleNamespace(run_id="r1"),
        "params": {"crawl_id": "CC-MAIN-2026-39"},
        "exception": RuntimeError("executor lost"),
    }
    alerts.on_task_failure(context)
    [body] = Capture.received
    assert "task `silver` failed" in body["text"] and "executor lost" in body["text"]


def test_a_run_that_timed_out_is_posted(webhook: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(alerts.WEBHOOK_ENV, webhook)
    run = SimpleNamespace(dag_id="corpus_monthly", run_id="r2")
    alerts.on_dag_failure({"dag_run": run, "reason": "timed_out", "params": {}})
    [body] = Capture.received
    assert "the run failed" in body["text"] and "reason: timed_out" in body["text"]


def test_no_webhook_means_a_log_line_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(alerts.WEBHOOK_ENV, raising=False)
    assert alerts.send("something broke") is False


def test_a_webhook_that_fails_never_raises(webhook: str, monkeypatch: pytest.MonkeyPatch) -> None:
    Capture.status = 500
    monkeypatch.setenv(alerts.WEBHOOK_ENV, webhook)
    assert alerts.send("something broke") is False
    monkeypatch.setenv(alerts.WEBHOOK_ENV, "http://127.0.0.1:9/nothing-listens-here")
    assert alerts.send("something broke") is False
