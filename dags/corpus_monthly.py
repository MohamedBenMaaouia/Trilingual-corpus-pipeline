"""The pipeline's DAG (Sprint 7: assembled, DECISIONS S7-01 to S7-03).

acquire -> download -> gate_a -> silver -> silver_v1 -> gate_b -> dedup -> gold.
No gate_c (cut, S3-05); corpus_stats is written by the gold task itself (S6-07).

Never on a timer on this machine (the user's decision, S7-01): every run is started by
hand, for the crawl given as the run's `crawl_id` parameter. A run for a crawl not yet
in bronze downloads it; a run for one already there downloads nothing.

This file says WHAT to run and WHEN, never HOW (invariant 13): the logic lives in
the corpus package. On Databricks (Sprint 10) only the launching lines change.
"""

from datetime import datetime, timedelta
from typing import Any

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param
from corpus_alerts import on_dag_failure, on_task_failure

# The pipeline's own interpreter, not Airflow's: two separate virtualenvs in one
# image, so Airflow's dependencies and ours never have to agree (DECISIONS S0-05).
CORPUS_PYTHON = "/opt/corpus-venv/bin/python"

DEFAULT_CRAWL_ID = "CC-MAIN-2026-39"  # the crawl in bronze since Sprint 1
SEGMENT_COUNT = 50  # N (D3)
SAMPLE_SEED = 42  # fixes WHICH segments; constant across crawls so sampling never changes

# What every task is given: the run's crawl (validated below, so it is safe in a shell
# command) and Airflow's run id, which ties together a run's rows in every table (T8).
CRAWL = '--crawl-id "{{ params.crawl_id }}"'
RUN_ID = '--run-id "{{ run_id }}"'

# Retries (plan 7.1.3). Ingestion talks to the internet: 3 retries, waits doubling from
# 1 minute. A Spark job rarely succeeds on a second try, and a try can take 15 minutes:
# 1 retry. A gate's failure is a verdict on the data, not an accident: no retry.
INGESTION: dict[str, Any] = {
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
}
SPARK: dict[str, Any] = {"retries": 1, "retry_delay": timedelta(minutes=2)}
GATE: dict[str, Any] = {"retries": 0}

with DAG(
    dag_id="corpus_monthly",
    description="Trilingual web corpus: bronze, silver, Gate B, dedup, gold (one crawl)",
    schedule=None,  # started by hand only, never on a timer here (S7-01)
    start_date=datetime(2026, 9, 1),
    catchup=False,
    # One run at a time: gold is a Delta table on S3A, safe for a single writing driver
    # only (trap S6, DECISIONS S6-02). Two runs would mean two writers.
    max_active_runs=1,
    # The 72-hour freshness target (plan 7.1.5): a run still going after 72 hours is
    # failed by the scheduler, its unfinished tasks skipped, and an alert sent.
    dagrun_timeout=timedelta(hours=72),
    params={
        "crawl_id": Param(
            DEFAULT_CRAWL_ID,
            type="string",
            # Checked when the run is triggered: nothing else can reach the commands.
            pattern="^CC-MAIN-[0-9]{4}-[0-9]{2}$",
            description="The Common Crawl crawl to process, e.g. CC-MAIN-2026-39",
        )
    },
    default_args={"on_failure_callback": on_task_failure},  # after its last retry
    on_failure_callback=on_dag_failure,  # the run failed (e.g. timed out)
    tags=["corpus", "sprint-7"],
):
    # Decide what this run consists of: one pending row per sampled segment.
    # Safe to rerun: ON CONFLICT DO NOTHING leaves existing rows alone.
    acquire = BashOperator(
        task_id="acquire",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.acquire {CRAWL} "
            f"--segments {SEGMENT_COUNT} --seed {SAMPLE_SEED} {RUN_ID}"
        ),
        **INGESTION,
    )

    # Fetch every pending segment into bronze. Prints one line per segment, so the
    # task log shows progress. Retryable and restartable by design (Story 1.3).
    download = BashOperator(
        task_id="download",
        bash_command=f"{CORPUS_PYTHON} -m corpus.jobs.download {CRAWL} {RUN_ID}",
        **INGESTION,
    )

    # Gate A: is bronze complete and trustworthy? A failure stops the DAG here, so
    # nothing downstream ever reads a half-downloaded crawl (invariant 8).
    gate_a = BashOperator(
        task_id="gate_a",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.gate --gate gate_a {CRAWL} "
            f"--expected-segments {SEGMENT_COUNT} {RUN_ID}"
        ),
        **GATE,
    )

    # Silver, Sprint 2 part: parse, normalize, remove boilerplate, write the interim
    # output and its dead letters for all N segments (plan 2.4.3). A Spark job: its
    # driver runs here in the scheduler container, its executors on the workers.
    silver = BashOperator(
        task_id="silver",
        bash_command=f"{CORPUS_PYTHON} -m corpus.jobs.run_silver {CRAWL} {RUN_ID} --keep-run-open",
        **SPARK,
    )

    # Silver, Sprint 3 part: language, PII redaction, quality, the silver_v1 contract,
    # written crawl by crawl (S3-04, S3-09). Reads the interim output, never bronze.
    silver_v1 = BashOperator(
        task_id="silver_v1",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.run_silver_v1 {CRAWL} {RUN_ID} --keep-run-open"
        ),
        **SPARK,
    )

    # Gate B: is this run's silver complete and inside its contract? A failure stops
    # the DAG before dedup (invariant 8) and closes the run as failed.
    gate_b = BashOperator(
        task_id="gate_b",
        bash_command=f"{CORPUS_PYTHON} -m corpus.jobs.gate --gate gate_b {CRAWL} {RUN_ID}",
        **GATE,
    )

    # Dedup, Sprint 4: exact and near duplicates within the crawl; writes which kept
    # silver pages gold keeps (S4-06). Leaves the run open for gold.
    dedup = BashOperator(
        task_id="dedup",
        bash_command=f"{CORPUS_PYTHON} -m corpus.jobs.run_dedup {CRAWL} {RUN_ID} --keep-run-open",
        **SPARK,
    )

    # Gold, Sprint 6: the crawl's kept, deduplicated, non-excluded pages replace that
    # crawl in the gold Delta table, compacted; corpus_stats recorded (S6-02 to S6-08).
    # Refuses to read silver without a passing Gate B. The last task: closes the run.
    gold = BashOperator(
        task_id="gold",
        bash_command=f"{CORPUS_PYTHON} -m corpus.jobs.run_gold {CRAWL} {RUN_ID}",
        **SPARK,
    )

    # Each task runs only if the previous one succeeded.
    acquire >> download >> gate_a >> silver >> silver_v1 >> gate_b >> dedup >> gold
