"""corpus.metrics.emit against real Postgres: upserts, so retries never double a number."""

from collections.abc import Iterator

import pytest
from psycopg2.extensions import connection as Connection

from corpus.metrics.emit import emit, finish_run, start_run

pytestmark = pytest.mark.integration

RUN_ID = "integration-metrics-run"
STAGE = "bronze"


@pytest.fixture(autouse=True)
def clean_rows(conn: Connection) -> Iterator[None]:
    yield
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM run_metrics WHERE run_id = %s", (RUN_ID,))
        cur.execute("DELETE FROM pipeline_runs WHERE run_id = %s", (RUN_ID,))


def metrics_of(conn: Connection) -> dict[str, float]:
    with conn.cursor() as cur:
        cur.execute("SELECT metric, value FROM run_metrics WHERE run_id = %s", (RUN_ID,))
        return dict(cur.fetchall())


def run_row(conn: Connection) -> tuple[str, int | None, int | None, bool]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, segment_count, sample_seed, ended_at IS NOT NULL"
            " FROM pipeline_runs WHERE run_id = %s",
            (RUN_ID,),
        )
        row = cur.fetchone()
        assert row is not None
        status, segment_count, sample_seed, ended = row
        return status, segment_count, sample_seed, ended


def test_a_run_is_opened_then_closed(conn: Connection, crawl_id: str) -> None:
    start_run(conn, RUN_ID, crawl_id, segment_count=50, sample_seed=42)
    assert run_row(conn) == ("running", 50, 42, False)
    finish_run(conn, RUN_ID, "success")
    assert run_row(conn) == ("success", 50, 42, True)


def test_reopening_a_run_keeps_what_it_already_knew(conn: Connection, crawl_id: str) -> None:
    """A retried task must not erase the segment count and seed of the run."""
    start_run(conn, RUN_ID, crawl_id, segment_count=50, sample_seed=42)
    finish_run(conn, RUN_ID, "failed")
    start_run(conn, RUN_ID, crawl_id)  # retry, without repeating the details
    assert run_row(conn) == ("running", 50, 42, False)  # ended_at cleared again


def test_an_unknown_status_is_refused(conn: Connection, crawl_id: str) -> None:
    start_run(conn, RUN_ID, crawl_id)
    with pytest.raises(ValueError, match="status must be"):
        finish_run(conn, RUN_ID, "done")


def test_closing_an_unknown_run_is_an_error(conn: Connection) -> None:
    """A wrong run_id would otherwise update nothing, silently, and leave a run open
    forever: exactly the bug the first real telemetry run exposed (S1-07)."""
    with pytest.raises(ValueError, match="no pipeline_runs row"):
        finish_run(conn, "a-run-that-was-never-started", "success")


def test_metrics_are_stored_as_one_row_each(conn: Connection, crawl_id: str) -> None:
    written = emit(
        conn,
        RUN_ID,
        crawl_id,
        STAGE,
        {"segments_intended": 50, "segments_complete": 50, "bytes_downloaded": 3_242_609_889},
    )
    assert written == 3
    assert metrics_of(conn) == {
        "segments_intended": 50.0,
        "segments_complete": 50.0,
        "bytes_downloaded": 3_242_609_889.0,
    }


def test_re_emitting_updates_instead_of_duplicating(conn: Connection, crawl_id: str) -> None:
    """The retry-safety that T8 asks for: without it every later average is wrong."""
    emit(conn, RUN_ID, crawl_id, STAGE, {"segments_complete": 7})
    emit(conn, RUN_ID, crawl_id, STAGE, {"segments_complete": 50})
    assert metrics_of(conn) == {"segments_complete": 50.0}


def test_emitting_nothing_is_a_no_op(conn: Connection, crawl_id: str) -> None:
    assert emit(conn, RUN_ID, crawl_id, STAGE, {}) == 0
    assert metrics_of(conn) == {}
