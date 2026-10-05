"""Write run telemetry: pipeline_runs (what the run was), run_metrics (what it measured)
and corpus_stats (what it put in gold).

Every function is an upsert or a replace, so a retried task updates its rows instead of
adding a second set (invariant 7, T8). Numbers here come only from real runs: nothing
estimates.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from psycopg2.extensions import connection as Connection
from psycopg2.extras import execute_values


def start_run(
    conn: Connection,
    run_id: str,
    crawl_id: str,
    *,
    segment_count: int | None = None,
    sample_seed: int | None = None,
    dev_mode: bool = False,
) -> None:
    """Open (or reopen, on a retry) this run. The seed is what makes it reproducible.

    dev_mode marks a dev run (5 segments), which never feeds gate baselines (T11). Once
    marked, a run stays marked: a later call without the flag cannot unmark it.
    """
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_runs
                (run_id, crawl_id, status, segment_count, sample_seed, dev_mode)
            VALUES (%s, %s, 'running', %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE
                SET status = 'running',
                    ended_at = NULL,
                    segment_count = COALESCE(EXCLUDED.segment_count, pipeline_runs.segment_count),
                    sample_seed = COALESCE(EXCLUDED.sample_seed, pipeline_runs.sample_seed),
                    dev_mode = pipeline_runs.dev_mode OR EXCLUDED.dev_mode
            """,
            (run_id, crawl_id, segment_count, sample_seed, dev_mode),
        )


def stamp_versions(
    conn: Connection, run_id: str, pipeline_version: str, schema_version: str
) -> None:
    """Record which logic (D8) and which contract produced this run's silver data."""
    with conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE pipeline_runs SET pipeline_version = %s, schema_version = %s WHERE run_id = %s",
            (pipeline_version, schema_version, run_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"no pipeline_runs row for run_id={run_id!r}: start_run first")


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


# corpus_stats' measured columns, in table order (migration 006).
CORPUS_STATS_COLUMNS = (
    "language",
    "quality_tier",
    "documents",
    "chars_total",
    "words_total",
    "mean_chars",
    "first_fetch",
    "last_fetch",
)


def replace_corpus_stats(
    conn: Connection,
    run_id: str,
    crawl_id: str,
    pipeline_version: str,
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Record what this run put in gold, one row per language x tier (C21). Replaces the
    run's rows in one transaction: a retry that no longer produces a tier leaves no
    stale row behind. Returns how many rows were written."""
    values = [
        (run_id, crawl_id, *(row[name] for name in CORPUS_STATS_COLUMNS), pipeline_version)
        for row in rows
    ]
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_stats WHERE run_id = %s", (run_id,))
        if values:
            execute_values(
                cur,
                f"INSERT INTO corpus_stats (run_id, crawl_id, {', '.join(CORPUS_STATS_COLUMNS)},"
                " pipeline_version) VALUES %s",
                values,
            )
    return len(values)
