"""Failure alerts for the corpus DAG (Story 7.1.4, D13; DECISIONS S7-03).

A failed task, or a run that outlives its time limit, posts one message to a webhook:
Discord or Slack, whichever URL is set in CORPUS_ALERT_WEBHOOK_URL (.env). With no URL
the alert is only logged. An alert that cannot be sent never fails anything.

Standard library only: Airflow runs callbacks in its own Python, which does not have
the corpus package (two separate environments, DECISIONS S0-05).
"""

import json
import logging
import os
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

WEBHOOK_ENV = "CORPUS_ALERT_WEBHOOK_URL"
MAX_CHARS = 1_900  # Discord refuses messages over 2,000 characters
ERROR_CHARS = 600  # the end of the error is what explains it

log = logging.getLogger(__name__)


def payload(url: str, text: str) -> dict[str, str]:
    """The JSON body each service expects: Discord reads "content", Slack "text"."""
    host = urlparse(url).hostname or ""
    key = "content" if host == "discord.com" or host.endswith(".discord.com") else "text"
    return {key: text[:MAX_CHARS]}


def failure_message(
    *,
    dag_id: str,
    run_id: str,
    crawl_id: str | None,
    task_id: str | None,
    try_number: int | None,
    reason: str,
    log_url: str | None,
) -> str:
    """One alert's text: which run, which stage, why, where to look."""
    what = f"task `{task_id}` failed" if task_id else "the run failed"
    lines = [
        f"corpus pipeline: {what}",
        f"DAG {dag_id}, run {run_id}" + (f", crawl {crawl_id}" if crawl_id else ""),
    ]
    if try_number is not None:
        lines.append(f"attempt {try_number} (retries exhausted)")
    lines.append(f"reason: {reason[-ERROR_CHARS:]}")
    if log_url:
        lines.append(f"log: {log_url}")
    return "\n".join(lines)


def post(url: str, text: str, timeout: float = 10.0) -> None:
    """Send one message. Raises on any HTTP or network error."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload(url, text)).encode("utf-8"),
        # Discord's front door rejects Python's default User-Agent.
        headers={"Content-Type": "application/json", "User-Agent": "corpus-pipeline-alert"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def send(text: str) -> bool:
    """Post `text` to the configured webhook. Returns whether it was sent; never raises."""
    url = os.environ.get(WEBHOOK_ENV, "").strip()
    if not url:
        log.warning("no %s set: alert not sent:\n%s", WEBHOOK_ENV, text)
        return False
    try:
        post(url, text)
    except Exception as err:  # an alert must never turn into a second failure
        log.error("alert not sent (%r):\n%s", err, text)
        return False
    log.info("alert sent:\n%s", text)
    return True


def _crawl_id(context: Mapping[str, Any]) -> str | None:
    params = context.get("params") or {}
    crawl = params.get("crawl_id") if isinstance(params, Mapping) else None
    return str(crawl) if crawl else None


def on_task_failure(context: Mapping[str, Any]) -> None:
    """Airflow's on_failure_callback for every task: runs once its retries are spent."""
    ti = context.get("ti") or context.get("task_instance")
    dag_run = context.get("dag_run")
    send(
        failure_message(
            dag_id=str(getattr(ti, "dag_id", "?")),
            run_id=str(getattr(dag_run, "run_id", None) or getattr(ti, "run_id", "?")),
            crawl_id=_crawl_id(context),
            task_id=getattr(ti, "task_id", None),
            try_number=getattr(ti, "try_number", None),
            reason=repr(context.get("exception") or "unknown error"),
            log_url=getattr(ti, "log_url", None),
        )
    )


def on_dag_failure(context: Mapping[str, Any]) -> None:
    """Airflow's DAG-level on_failure_callback: the run failed, e.g. it outlived its
    72-hour limit (dagrun_timeout: reason "timed_out")."""
    dag_run = context.get("dag_run")
    send(
        failure_message(
            dag_id=str(getattr(dag_run, "dag_id", "?")),
            run_id=str(getattr(dag_run, "run_id", "?")),
            crawl_id=_crawl_id(context),
            task_id=None,
            try_number=None,
            reason=str(context.get("reason") or "the run failed"),
            log_url=None,
        )
    )
