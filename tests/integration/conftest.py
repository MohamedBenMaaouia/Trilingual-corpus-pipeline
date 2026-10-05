"""Fixtures for tests that talk to the real Postgres on the compose network.

They need the stack up (make up) and CORPUS_METRICS_DSN, which docker-compose.yml
gives the `tests` service. Every test works under its own throwaway crawl id and
deletes its rows afterwards, so tests never disturb real runs or each other.
"""

import random
from collections.abc import Iterator

import pytest
from psycopg2.extensions import connection as Connection

from corpus.bronze.control import SegmentControl
from corpus.db import connect

pytestmark = pytest.mark.integration


@pytest.fixture
def conn() -> Iterator[Connection]:
    with connect() as connection:
        yield connection


@pytest.fixture
def crawl_id(conn: Connection) -> Iterator[str]:
    """A crawl id no real run uses. Its rows are removed when the test ends.

    Digits only: a real id is CC-MAIN-YYYY-WW, and io.check_crawl_id enforces it.
    """
    fake = f"CC-MAIN-{random.randint(9000, 9999)}-{random.randint(10, 99)}"
    yield fake
    # Every table keyed by crawl: jobs run by a test as subprocesses (kill-and-restart)
    # open their own runs and write metrics, not only segments rows (S2-10: before this,
    # each `make test` left two runs stuck at 'running' in the real telemetry).
    with conn, conn.cursor() as cur:
        # Table names are constants of this file, never input; the value stays a parameter.
        for table in ("segments", "run_metrics", "gate_results", "corpus_stats", "pipeline_runs"):
            cur.execute(f"DELETE FROM {table} WHERE crawl_id = %s", (fake,))


@pytest.fixture
def control(conn: Connection) -> SegmentControl:
    return SegmentControl(conn)
