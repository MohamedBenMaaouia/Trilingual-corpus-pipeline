"""The pipeline's DAG.

Sprint 6: acquire -> download -> gate_a -> silver -> silver_v1 -> gate_b -> dedup -> gold.
No gate_c (cut, S3-05); corpus_stats is written by the gold task itself (S6-07).

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
    description="Trilingual web corpus pipeline (S6: bronze, silver, Gate B, dedup, gold)",
    schedule=None,  # manual trigger only; scheduling is Sprint 7
    start_date=datetime(2026, 9, 1),
    catchup=False,  # never backfill past runs on its own
    # One run at a time: gold is a Delta table on S3A, safe for a single writing driver
    # only (trap S6, DECISIONS S6-02). Two runs would mean two writers.
    max_active_runs=1,
    tags=["corpus", "sprint-6"],
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

    # Silver, Sprint 2 part: parse, normalize, remove boilerplate, write the interim
    # output and its dead letters for all N segments (plan 2.4.3). A Spark job: its
    # driver runs here in the scheduler container, its executors on the workers. Not
    # the last task any more: it leaves the run open (a crash still closes it failed).
    silver = BashOperator(
        task_id="silver",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.run_silver --crawl-id {CRAWL_ID} "
            '--run-id "{{ run_id }}" --keep-run-open'
        ),
    )

    # Silver, Sprint 3 part: language, PII redaction, quality, the silver_v1 contract,
    # written crawl by crawl (S3-04, S3-09). Reads the interim output, never bronze.
    silver_v1 = BashOperator(
        task_id="silver_v1",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.run_silver_v1 --crawl-id {CRAWL_ID} "
            '--run-id "{{ run_id }}" --keep-run-open'
        ),
    )

    # Gate B: is this run's silver complete and inside its contract? A failure stops
    # the DAG before dedup (invariant 8) and closes the run as failed.
    gate_b = BashOperator(
        task_id="gate_b",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.gate --gate gate_b --crawl-id {CRAWL_ID} "
            '--run-id "{{ run_id }}"'
        ),
    )

    # Dedup, Sprint 4: exact and near duplicates within the crawl; writes which kept
    # silver pages gold keeps (S4-06). Leaves the run open for gold.
    dedup = BashOperator(
        task_id="dedup",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.run_dedup --crawl-id {CRAWL_ID} "
            '--run-id "{{ run_id }}" --keep-run-open'
        ),
    )

    # Gold, Sprint 6: the crawl's kept, deduplicated, non-excluded pages replace that
    # crawl in the gold Delta table, compacted; corpus_stats recorded (S6-02 to S6-08).
    # Refuses to read silver without a passing Gate B. The last task: closes the run.
    gold = BashOperator(
        task_id="gold",
        bash_command=(
            f"{CORPUS_PYTHON} -m corpus.jobs.run_gold --crawl-id {CRAWL_ID} "
            '--run-id "{{ run_id }}"'
        ),
    )

    # Each task runs only if the previous one succeeded.
    acquire >> download >> gate_a >> silver >> silver_v1 >> gate_b >> dedup >> gold
