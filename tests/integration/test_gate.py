"""corpus.jobs.gate.record_result against real Postgres.

The Soda run itself needs the Airflow image (Soda has its own virtualenv there), so
it is exercised by hand and recorded in DECISIONS S1-06; here we cover the verdict
bookkeeping, which is what Sprint 8's dashboards will read.
"""

from collections.abc import Iterator

import pytest
from psycopg2.extensions import connection as Connection

from corpus.jobs.gate import record_result

pytestmark = pytest.mark.integration

RUN_ID = "integration-test-run"


@pytest.fixture(autouse=True)
def clean_gate_rows(conn: Connection) -> Iterator[None]:
    yield
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM gate_results WHERE run_id = %s", (RUN_ID,))


def stored(conn: Connection) -> list[tuple[bool, str | None]]:
    with conn.cursor() as cur:
        cur.execute("SELECT passed, failure_detail FROM gate_results WHERE run_id = %s", (RUN_ID,))
        return [(row[0], row[1]) for row in cur.fetchall()]


def test_a_passing_gate_records_no_detail(conn: Connection, crawl_id: str) -> None:
    record_result(RUN_ID, crawl_id, "gate_a", passed=True, detail="4/4 checks PASSED")
    assert stored(conn) == [(True, None)]


def test_a_failing_gate_records_the_report(conn: Connection, crawl_id: str) -> None:
    record_result(RUN_ID, crawl_id, "gate_a", passed=False, detail="No segment failed [FAILED]")
    assert stored(conn) == [(False, "No segment failed [FAILED]")]


def test_a_retry_replaces_the_row_instead_of_adding_one(conn: Connection, crawl_id: str) -> None:
    """Retry-safety (invariant 7): one row per run and gate, holding the last verdict."""
    record_result(RUN_ID, crawl_id, "gate_a", passed=False, detail="transient failure")
    record_result(RUN_ID, crawl_id, "gate_a", passed=True, detail="4/4 checks PASSED")
    assert stored(conn) == [(True, None)]
