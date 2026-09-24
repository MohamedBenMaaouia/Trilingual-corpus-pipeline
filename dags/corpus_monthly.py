"""The pipeline's DAG.

Sprint 1: acquire -> download. Sprint 7 completes the chain:
acquire -> download -> gate_a -> silver -> gate_b -> dedup -> gate_c -> gold -> stats.

This file says WHAT to run and WHEN, never HOW (invariant 13): the logic lives in
the corpus package. On Databricks (Sprint 10) only the launching lines change.
"""

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# The pipeline's own interpreter, not Airflow's: two separate virtualenvs in one
# image, so Airflow's dependencies and ours never have to agree (DECISIONS S0-05).
CORPUS_PYTHON = "/opt/corpus-venv/bin/python"

# Sprint 1 only: fixed values. Sprint 7 derives the crawl from the run's date.
CRAWL_ID = "CC-MAIN-2026-39"
SEGMENT_COUNT = 50  # N (D3)
SAMPLE_SEED = 42  # fixes WHICH segments; constant across crawls so sampling never changes

with DAG(
    dag_id="corpus_monthly",
    description="Trilingual web corpus pipeline (S1: bronze ingestion)",
    schedule=None,  # manual trigger only; scheduling is Sprint 7
    start_date=datetime(2026, 9, 1),
    catchup=False,  # never backfill past runs on its own
    tags=["corpus", "sprint-1"],
):
    # Decide what this run consists of: one pending row per sampled segment.
    # Safe to rerun: ON CONFLICT DO NOTHING leaves existing rows alone.
    acquire = BashOperator(
        task_id="acquire",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.acquire "
            f"--crawl-id {CRAWL_ID} --segments {SEGMENT_COUNT} --seed {SAMPLE_SEED} "
            '--run-id "{{ run_id }}"'
        ),
    )

    # Fetch every pending segment into bronze. Prints one line per segment, so the
    # task log shows progress. Retryable and restartable by design (Story 1.3).
    download = BashOperator(
        task_id="download",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.download --crawl-id {CRAWL_ID} "
            '--run-id "{{ run_id }}"'
        ),
    )

    # Gate A: is bronze complete and trustworthy? A failure stops the DAG here, so
    # nothing downstream ever reads a half-downloaded crawl (invariant 8).
    # {{ run_id }} is Airflow's own run identifier, so the verdict in gate_results
    # can be traced back to this exact run (T8).
    gate_a = BashOperator(
        task_id="gate_a",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.gate --gate gate_a "
            f"--crawl-id {CRAWL_ID} --expected-segments {SEGMENT_COUNT} "
            '--run-id "{{ run_id }}"'
        ),
    )

    acquire >> download >> gate_a  # each task runs only if the previous succeeded
