"""corpus.acquisition.control against real Postgres (task 1.2.3)."""

import pytest

from corpus.acquisition.control import SegmentControl
from corpus.acquisition.manifest import Segment

pytestmark = pytest.mark.integration

SAMPLE = [
    Segment("00007", "crawl-data/x/wet/a-00007.warc.wet.gz"),
    Segment("00042", "crawl-data/x/wet/b-00042.warc.wet.gz"),
    Segment("01337", "crawl-data/x/wet/c-01337.warc.wet.gz"),
]


def test_register_intent_writes_one_pending_row_per_segment(
    control: SegmentControl, crawl_id: str
) -> None:
    assert control.register_intent(crawl_id, SAMPLE) == 3
    assert control.status_counts(crawl_id) == {"pending": 3}
    assert control.pending_count(crawl_id) == 3


def test_register_intent_twice_adds_no_duplicates(control: SegmentControl, crawl_id: str) -> None:
    control.register_intent(crawl_id, SAMPLE)
    assert control.register_intent(crawl_id, SAMPLE) == 0
    assert control.pending_count(crawl_id) == 3


def test_register_intent_keeps_the_source_url(control: SegmentControl, crawl_id: str) -> None:
    control.register_intent(crawl_id, SAMPLE[:1])
    with control._conn.cursor() as cur:  # noqa: SLF001 - checking what was stored
        cur.execute(
            "SELECT source_url, attempts FROM segments WHERE crawl_id = %s AND segment_id = %s",
            (crawl_id, "00007"),
        )
        assert cur.fetchone() == (SAMPLE[0].source_url, 0)


def test_register_intent_with_no_segments_is_a_no_op(
    control: SegmentControl, crawl_id: str
) -> None:
    assert control.register_intent(crawl_id, []) == 0
    assert control.status_counts(crawl_id) == {}


def test_claim_pending_takes_work_and_marks_it_downloading(
    control: SegmentControl, crawl_id: str
) -> None:
    control.register_intent(crawl_id, SAMPLE)
    claimed = control.claim_pending(crawl_id, limit=2)
    # ORDER BY in the query picks WHICH rows are taken (the lowest ids); the order
    # RETURNING gives them back in is not defined by SQL, so compare as a set.
    assert sorted(s.segment_id for s in claimed) == ["00007", "00042"]
    assert {s.path for s in claimed} == {SAMPLE[0].path, SAMPLE[1].path}  # rebuilt from the URL
    assert control.status_counts(crawl_id) == {"downloading": 2, "pending": 1}


def test_claim_pending_never_hands_the_same_segment_twice(
    control: SegmentControl, crawl_id: str
) -> None:
    control.register_intent(crawl_id, SAMPLE)
    first = control.claim_pending(crawl_id, limit=2)
    second = control.claim_pending(crawl_id, limit=2)
    assert {s.segment_id for s in first} & {s.segment_id for s in second} == set()
    assert len(second) == 1  # only one was left
    assert control.claim_pending(crawl_id, limit=2) == []  # nothing pending now


def test_claim_pending_counts_attempts(control: SegmentControl, crawl_id: str) -> None:
    control.register_intent(crawl_id, SAMPLE[:1])
    control.claim_pending(crawl_id, limit=1)
    control.reclaim_stale(crawl_id, older_than_seconds=0)
    control.claim_pending(crawl_id, limit=1)
    with control._conn.cursor() as cur:  # noqa: SLF001
        cur.execute(
            "SELECT attempts FROM segments WHERE crawl_id = %s AND segment_id = %s",
            (crawl_id, "00007"),
        )
        assert cur.fetchone() == (2,)


def test_mark_complete_records_what_landed(control: SegmentControl, crawl_id: str) -> None:
    control.register_intent(crawl_id, SAMPLE[:1])
    control.claim_pending(crawl_id, limit=1)
    control.mark_complete(
        crawl_id, "00007", size_bytes=66_761_801, etag='"abc123"', final_key="bronze/.../x.wet.gz"
    )
    assert control.status_counts(crawl_id) == {"complete": 1}
    with control._conn.cursor() as cur:  # noqa: SLF001
        cur.execute(
            """SELECT bytes, etag, final_key, claimed_at, completed_at IS NOT NULL
               FROM segments WHERE crawl_id = %s AND segment_id = %s""",
            (crawl_id, "00007"),
        )
        assert cur.fetchone() == (66_761_801, '"abc123"', "bronze/.../x.wet.gz", None, True)


def test_mark_complete_records_whether_the_checksum_was_verified(
    control: SegmentControl, crawl_id: str
) -> None:
    """True = ETag was an MD5 and matched; False = not comparable; never a false claim."""
    control.register_intent(crawl_id, SAMPLE[:2])
    control.claim_pending(crawl_id, limit=2)
    control.mark_complete(
        crawl_id, "00007", size_bytes=1, etag='"x"', final_key="k", checksum_verified=True
    )
    control.mark_complete(
        crawl_id, "00042", size_bytes=1, etag='"y-3"', final_key="k", checksum_verified=False
    )
    with control._conn.cursor() as cur:  # noqa: SLF001
        cur.execute(
            "SELECT segment_id, checksum_verified FROM segments"
            " WHERE crawl_id = %s ORDER BY segment_id",
            (crawl_id,),
        )
        assert cur.fetchall() == [("00007", True), ("00042", False)]


def test_complete_segments_are_never_handed_out_again(
    control: SegmentControl, crawl_id: str
) -> None:
    """The heart of restart-safety: finished work is not repeated (task 1.7.1)."""
    control.register_intent(crawl_id, SAMPLE[:1])
    control.claim_pending(crawl_id, limit=1)
    control.mark_complete(crawl_id, "00007", size_bytes=1, etag=None, final_key="k")
    control.register_intent(crawl_id, SAMPLE[:1])  # a rerun registers intent again
    assert control.claim_pending(crawl_id, limit=10) == []
    assert control.status_counts(crawl_id) == {"complete": 1}


def test_mark_failed_stores_the_error(control: SegmentControl, crawl_id: str) -> None:
    control.register_intent(crawl_id, SAMPLE[:1])
    control.claim_pending(crawl_id, limit=1)
    control.mark_failed(crawl_id, "00007", "ConnectionError: 503 SlowDown")
    assert control.status_counts(crawl_id) == {"failed": 1}


def test_reclaim_stale_only_touches_old_claims(control: SegmentControl, crawl_id: str) -> None:
    control.register_intent(crawl_id, SAMPLE)
    control.claim_pending(crawl_id, limit=2)
    assert control.reclaim_stale(crawl_id, older_than_seconds=3600) == 0  # just claimed
    assert control.reclaim_stale(crawl_id, older_than_seconds=0) == 2  # a dead worker's rows
    assert control.status_counts(crawl_id) == {"pending": 3}


def test_retry_failed_respects_the_attempt_cap(control: SegmentControl, crawl_id: str) -> None:
    control.register_intent(crawl_id, SAMPLE[:1])
    control.claim_pending(crawl_id, limit=1)  # attempts = 1
    control.mark_failed(crawl_id, "00007", "boom")
    assert control.retry_failed(crawl_id, max_attempts=3) == 1
    control.claim_pending(crawl_id, limit=1)  # attempts = 2
    control.mark_failed(crawl_id, "00007", "boom")
    assert control.retry_failed(crawl_id, max_attempts=2) == 0  # cap reached, stays failed
    assert control.status_counts(crawl_id) == {"failed": 1}
