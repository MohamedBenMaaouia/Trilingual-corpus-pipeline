"""What Spark measured, per step of a job: shuffle, spill, longest task (DECISIONS S4-08).

A job wraps each step in `step(...)`, which names the Spark jobs it starts (a job
group) and times it. Before the session stops, `step_metrics` reads the driver's own
monitoring REST API (Spark docs, "Monitoring and Instrumentation", REST API: the same
numbers as the Spark UI's Stages tab) and sums each step's completed stages. These are
spec 12.4's "runtime and shuffle spill per stage", and Sprint 5's before/after numbers.

Best effort: if the UI cannot be reached (tests switch it off), the job loses these
numbers, never its run.
"""

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import requests
from pyspark.sql import SparkSession

_STEP_METRICS = (
    "tasks",
    "executor_seconds",
    "shuffle_read_bytes",
    "shuffle_write_bytes",
    "spill_bytes",
    "longest_task_seconds",
)


@contextmanager
def step(spark: SparkSession, prefix: str, name: str, metrics: dict[str, float]) -> Iterator[None]:
    """Tag the Spark jobs started inside the block with `name`; record its wall time as
    <prefix>_step_<name>_seconds. Put the action that computes the step (a count on a
    persisted result) inside the block, or its work lands in a later step."""
    spark.sparkContext.setJobGroup(name, f"{prefix}: {name}")
    started = time.monotonic()
    try:
        yield
    finally:
        metrics[f"{prefix}_step_{name}_seconds"] = round(time.monotonic() - started, 1)


def _get(url: str) -> Any:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _number(value: Any, where: str) -> float:
    """A number from the API, 0.0 if there is none: reported, then counted as 0."""
    number = _as_float(value)
    if not math.isfinite(number):
        print(f"step_metrics: {where}: {value!r} is not a number, counted as 0")
        return 0.0
    return number


def _longest_task_ms(summary: Any, where: str) -> float:
    """The stage's longest task. Spark can report the duration's maximum as the text
    "NaN" (seen on stages of hundreds of near-empty tasks, live and in the history
    server, S4-08); its run time is then used, which leaves out only scheduling and
    deserialization."""
    duration = _as_float(summary["duration"][0])
    if math.isfinite(duration):
        return duration
    return _number(summary["executorRunTime"][0], f"{where} longest task")


def step_metrics(spark: SparkSession, prefix: str) -> dict[str, float]:
    """For each job group: tasks, executor time, shuffle read and write, spill, and the
    longest task, as <prefix>_step_<group>_<metric>. Call it before spark.stop().

    A stage counts for the job that ran it: later jobs that reuse its output list it as
    skipped, so each stage is given to the first job (lowest id) that lists it.
    """
    context = spark.sparkContext
    if not context.uiWebUrl:
        return {}
    api = f"{context.uiWebUrl}/api/v1/applications/{context.applicationId}"
    totals: dict[str, dict[str, float]] = {}
    try:
        jobs = sorted(_get(f"{api}/jobs"), key=lambda job: job["jobId"])
        stages = {(s["stageId"], s["attemptId"]): s for s in _get(f"{api}/stages?status=complete")}
        owner: dict[int, str] = {}
        for job in jobs:
            for stage_id in job["stageIds"]:
                owner.setdefault(stage_id, job.get("jobGroup") or "")
        for (stage_id, attempt), stage in stages.items():
            group = owner.get(stage_id)
            if not group:
                continue
            where = f"stage {stage_id}.{attempt}"
            summary = _get(f"{api}/stages/{stage_id}/{attempt}/taskSummary?quantiles=1.0")
            t = totals.setdefault(group, dict.fromkeys(_STEP_METRICS, 0.0))
            t["tasks"] += _number(stage["numCompleteTasks"], where)
            t["executor_seconds"] += _number(stage["executorRunTime"], where) / 1000
            t["shuffle_read_bytes"] += _number(stage["shuffleReadBytes"], where)
            t["shuffle_write_bytes"] += _number(stage["shuffleWriteBytes"], where)
            t["spill_bytes"] += _number(stage["memoryBytesSpilled"], where) + _number(
                stage["diskBytesSpilled"], where
            )
            longest = _longest_task_ms(summary, where) / 1000
            t["longest_task_seconds"] = max(t["longest_task_seconds"], longest)
    except Exception as err:  # best effort: a monitoring hiccup never fails the run
        print(f"step_metrics: Spark's REST API unusable ({err!r}); no per-step numbers")
        return {}
    return {
        f"{prefix}_step_{group}_{name}": round(value, 1)
        for group, values in totals.items()
        for name, value in values.items()
    }
