"""The control plane: read and write the `segments` table.

Every method is safe to rerun (invariant 7). Nothing here downloads anything;
the downloader (Story 1.3) asks this class what to do and reports back.
"""

from collections.abc import Iterable

from psycopg2.extensions import connection as Connection
from psycopg2.extras import execute_values

from corpus.acquisition.manifest import Segment


class SegmentControl:
    """Repository over `segments`. One instance per open connection."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def register_intent(self, crawl_id: str, segments: Iterable[Segment]) -> int:
        """Record which segments this crawl should end up with. Returns rows inserted.

        Rerunning with the same sample inserts nothing: the primary key
        (crawl_id, segment_id) already holds those rows, and ON CONFLICT DO NOTHING
        leaves their status and attempt counts untouched.
        """
        rows = [(crawl_id, s.segment_id, s.source_url) for s in segments]
        if not rows:
            return 0
        with self._conn, self._conn.cursor() as cur:
            execute_values(  # one statement for all rows, instead of one per row
                cur,
                """
                INSERT INTO segments (crawl_id, segment_id, source_url, status)
                VALUES %s
                ON CONFLICT (crawl_id, segment_id) DO NOTHING
                """,
                rows,
                template="(%s, %s, %s, 'pending')",
            )
            return cur.rowcount  # rows actually inserted, conflicts excluded

    def claim_pending(self, crawl_id: str, limit: int) -> list[Segment]:
        """Take up to `limit` pending segments for this worker and return them.

        One statement does the selecting and the claiming, so two workers can never
        get the same segment:
          - FOR UPDATE locks the chosen rows;
          - SKIP LOCKED makes a second worker step over rows already locked instead
            of waiting for them;
          - claimed_at records when the work started, which is what lets a restart
            tell "in progress" from "the process that held this died" (T8);
          - attempts counts tries, so the kill-and-restart proof can show that
            already-complete segments were not fetched again (task 1.7.1).
        """
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE segments SET status = 'downloading',
                                    claimed_at = now(),
                                    attempts = attempts + 1
                WHERE (crawl_id, segment_id) IN (
                    SELECT crawl_id, segment_id FROM segments
                    WHERE crawl_id = %s AND status = 'pending'
                    ORDER BY segment_id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING segment_id, source_url
                """,
                (crawl_id, limit),
            )
            return [Segment.from_source_url(segment_id, url) for segment_id, url in cur.fetchall()]

    def mark_complete(
        self, crawl_id: str, segment_id: str, *, size_bytes: int, etag: str | None, final_key: str
    ) -> None:
        """The file is validated and stored: record what landed and where."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE segments SET status = 'complete', bytes = %s, etag = %s, final_key = %s,
                                    completed_at = now(), claimed_at = NULL, error = NULL
                WHERE crawl_id = %s AND segment_id = %s
                """,
                (size_bytes, etag, final_key, crawl_id, segment_id),
            )

    def mark_failed(self, crawl_id: str, segment_id: str, error: str) -> None:
        """This attempt failed. The run continues; Gate A later refuses any failure."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE segments SET status = 'failed', error = %s, claimed_at = NULL
                WHERE crawl_id = %s AND segment_id = %s
                """,
                (error[:2000], crawl_id, segment_id),  # keep one huge traceback out of the row
            )

    def reclaim_stale(self, crawl_id: str, older_than_seconds: int) -> int:
        """Put rows claimed too long ago back to pending. Returns how many.

        A killed process leaves its rows in 'downloading' forever, and nothing would
        ever pick them up again (task 1.3.5): the restart calls this first.
        """
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE segments SET status = 'pending', claimed_at = NULL
                WHERE crawl_id = %s AND status = 'downloading'
                  AND claimed_at < now() - make_interval(secs => %s)
                """,
                (crawl_id, older_than_seconds),
            )
            return cur.rowcount

    def retry_failed(self, crawl_id: str, max_attempts: int) -> int:
        """Put failed segments back to pending while they are under the attempt cap.

        Gate A requires zero failures, so failures need a bounded retry path before
        the gate runs (task 1.3.5).
        """
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE segments SET status = 'pending', claimed_at = NULL
                WHERE crawl_id = %s AND status = 'failed' AND attempts < %s
                """,
                (crawl_id, max_attempts),
            )
            return cur.rowcount

    def status_counts(self, crawl_id: str) -> dict[str, int]:
        """How many segments are in each status, e.g. {'pending': 48, 'complete': 2}."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "SELECT status, count(*) FROM segments WHERE crawl_id = %s GROUP BY status",
                (crawl_id,),
            )
            return {status: count for status, count in cur.fetchall()}

    def pending_count(self, crawl_id: str) -> int:
        """Segments still waiting to be downloaded."""
        return self.status_counts(crawl_id).get("pending", 0)
