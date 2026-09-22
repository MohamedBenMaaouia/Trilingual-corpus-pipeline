"""The pipeline's DAG. Sprint 0: one task, the smoke job (task 0.5.2).

Sprint 7 turns this into the real chain:
acquire -> download -> gate_a -> silver -> gate_b -> dedup -> gate_c -> gold -> stats.

This file says WHAT to run and WHEN, never HOW (invariant 13): the logic lives in
the corpus package. On Databricks (Sprint 10) only the launching line changes.
"""

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# The pipeline's own interpreter, not Airflow's: two separate virtualenvs in one
# image, so Airflow's dependencies and ours never have to agree (DECISIONS S0-05).
CORPUS_PYTHON = "/opt/corpus-venv/bin/python"

# Sprint 0 only: fixed values. Sprint 7 derives the crawl from the run's date.
CRAWL_ID = "CC-MAIN-2026-39"
SEGMENTS = "0 1 2"

with DAG(
    dag_id="corpus_monthly",
    description="Trilingual web corpus pipeline (S0: smoke test only)",
    schedule=None,  # manual trigger only; scheduling is Sprint 7
    start_date=datetime(2026, 9, 1),
    catchup=False,  # never backfill past runs on its own
    tags=["corpus", "sprint-0"],
):
    BashOperator(
        task_id="smoke",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.smoke --crawl-id {CRAWL_ID} --segments {SEGMENTS}"
        ),
    )
