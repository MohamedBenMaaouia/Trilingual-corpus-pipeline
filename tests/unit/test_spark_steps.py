"""metrics.spark_steps: per-step numbers from Spark's REST API (DECISIONS S4-08)."""

from types import SimpleNamespace
from typing import Any

import pytest
from pyspark.sql import SparkSession

from corpus.metrics import spark_steps
from corpus.metrics.spark_steps import step, step_metrics

API = "http://driver:4040/api/v1/applications/app-1"


def stage(stage_id: int, run_ms: int, read: int, write: int, spilled: int) -> dict[str, Any]:
    return {
        "stageId": stage_id,
        "attemptId": 0,
        "numCompleteTasks": 4,
        "executorRunTime": run_ms,
        "shuffleReadBytes": read,
        "shuffleWriteBytes": write,
        "memoryBytesSpilled": spilled,
        "diskBytesSpilled": spilled,
    }


RESPONSES: dict[str, Any] = {
    f"{API}/jobs": [
        # Job 2 lists stage 1 again (it reuses its output: skipped), so stage 1 stays
        # with job 1's group.
        {"jobId": 2, "jobGroup": "pairs", "stageIds": [1, 2]},
        {"jobId": 1, "jobGroup": "bands", "stageIds": [0, 1]},
        {"jobId": 3, "stageIds": [3]},  # no group: ignored
    ],
    f"{API}/stages?status=complete": [
        stage(0, 2000, 0, 100, 0),
        stage(1, 3000, 100, 50, 0),
        stage(2, 9000, 50, 0, 10),
        stage(3, 1000, 0, 0, 0),
    ],
    **{
        f"{API}/stages/{i}/0/taskSummary?quantiles=1.0": {"duration": [ms]}
        for i, ms in [(0, 700), (1, 1500), (2, 6000), (3, 100)]
    },
}


def test_stages_are_summed_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spark_steps, "_get", RESPONSES.__getitem__)
    spark = SimpleNamespace(
        sparkContext=SimpleNamespace(uiWebUrl="http://driver:4040", applicationId="app-1")
    )
    metrics = step_metrics(spark, "dedup")  # type: ignore[arg-type]
    assert metrics["dedup_step_bands_executor_seconds"] == 5.0
    assert metrics["dedup_step_bands_shuffle_write_bytes"] == 150
    assert metrics["dedup_step_bands_longest_task_seconds"] == 1.5
    assert metrics["dedup_step_pairs_shuffle_read_bytes"] == 50
    assert metrics["dedup_step_pairs_spill_bytes"] == 20
    assert metrics["dedup_step_pairs_longest_task_seconds"] == 6.0
    assert metrics["dedup_step_pairs_tasks"] == 4
    assert not [k for k in metrics if "step__" in k]


@pytest.mark.parametrize(
    ("summary", "longest"),
    [
        ({"duration": ["NaN"], "executorRunTime": [5500.0]}, 5.5),  # run time instead
        ({"duration": ["NaN"], "executorRunTime": ["NaN"]}, 0.0),  # nothing usable
    ],
)
def test_a_duration_sent_as_text(
    monkeypatch: pytest.MonkeyPatch, summary: dict[str, Any], longest: float
) -> None:
    """Spark reported a longest-task duration as the text "NaN" (S4-08)."""
    responses = dict(RESPONSES)
    responses[f"{API}/stages/2/0/taskSummary?quantiles=1.0"] = summary
    monkeypatch.setattr(spark_steps, "_get", responses.__getitem__)
    spark = SimpleNamespace(
        sparkContext=SimpleNamespace(uiWebUrl="http://driver:4040", applicationId="app-1")
    )
    metrics = step_metrics(spark, "dedup")  # type: ignore[arg-type]
    assert metrics["dedup_step_pairs_longest_task_seconds"] == longest
    assert metrics["dedup_step_pairs_shuffle_read_bytes"] == 50  # the rest still counts


def test_no_ui_no_numbers_and_the_run_goes_on(spark: SparkSession) -> None:
    """Tests run Spark without its UI: the job simply gets no per-step numbers."""
    assert step_metrics(spark, "dedup") == {}


def test_step_names_the_jobs_and_times_them(spark: SparkSession) -> None:
    metrics: dict[str, float] = {}
    with step(spark, "dedup", "count", metrics):
        spark.range(10).count()
    assert spark.sparkContext.getLocalProperty("spark.jobGroup.id") == "count"
    assert metrics["dedup_step_count_seconds"] >= 0
