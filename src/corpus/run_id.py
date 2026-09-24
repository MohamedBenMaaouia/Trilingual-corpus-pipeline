"""The identifier that ties a run's control rows, metrics and gate verdicts together.

Inside Airflow it is dag_run.run_id, passed with --run-id (T8). For a run started by
hand there is no such id, so one is generated from the clock.
"""

from datetime import UTC, datetime


def default_run_id() -> str:
    """e.g. local__2026-09-24T14:35:02Z. Sorts chronologically and says where it came from."""
    return "local__" + datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
