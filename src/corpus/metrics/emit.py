"""Write run telemetry: pipeline_runs (what the run was) and run_metrics (what it measured).

Every function is an upsert, so a retried task updates its rows instead of adding a
second set (invariant 7, T8). Numbers here come only from real runs: nothing estimates.
"""

from collections.abc import Mapping

from psycopg2.extensions import connection as Connection
from psycopg2.extras import execute_values


def start_run(
    conn: Connection,
    run_id: str,
    crawl_id: str,
    *,
    segment_count: int | None = None,
    sample_seed: int | None = None,
) -> None:
    """Open (or reopen, on a retry) this run. The seed is what makes it reproducible."""
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_runs (run_id, crawl_id, status, segment_count, sample_seed)
            VALUES (%s, %s, 'running', %s, %s)
            ON CONFLICT (run_id) DO UPDATE
                SET status = 'running',
                    ended_at = NULL,
                    segment_count = COALESCE(EXCLUDED.segment_count, pipeline_runs.segment_count),
                    sample_seed = COALESCE(EXCLUDED.sample_seed, pipeline_runs.sample_seed)
            """,
            (run_id, crawl_id, segment_count, sample_seed),
        )


def finish_run(conn: Connection, run_id: str, status: str) -> None:
    """Close the run: 'success' or 'failed'. Feeds the success-rate metric (S8)."""
    if status not in ("success", "failed"):
        raise ValueError(f"status must be 'success' or 'failed', got {status!r}")
    with conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE pipeline_runs SET status = %s, ended_at = now() WHERE run_id = %s",
            (status, run_id),
        )
        if cur.rowcount == 0:
            # An UPDATE matching nothing is not an SQL error, so a wrong run_id would
            # leave the run open forever and nobody would notice. Fail loudly instead.
            raise ValueError(f"no pipeline_runs row for run_id={run_id!r}: nothing was closed")


def emit(
    conn: Connection, run_id: str, crawl_id: str, stage: str, metrics: Mapping[str, float]
) -> int:
    """Record one stage's numbers. Returns how many metrics were written."""
    rows = [(run_id, crawl_id, stage, name, float(value)) for name, value in metrics.items()]
    if not rows:
        return 0
    with conn, conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO run_metrics (run_id, crawl_id, stage, metric, value)
            VALUES %s
            ON CONFLICT (run_id, stage, metric) DO UPDATE
                SET value = EXCLUDED.value, recorded_at = now()
            """,
            rows,
        )
    return len(rows)
