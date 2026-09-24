"""Stage 1 entry point: fetch this crawl's pending segments into bronze (Story 1.3).

    python -m corpus.jobs.download --crawl-id CC-MAIN-2026-39

Safe to kill and rerun: it reclaims stale claims, skips complete segments, and
never leaves a half-written object in bronze.
"""

import argparse
from pathlib import Path

from corpus.acquisition.control import SegmentControl
from corpus.config import get_settings
from corpus.config.local import LocalSettings
from corpus.db import connect
from corpus.ingest.download import (
    DEFAULT_THREADS,
    STALE_LEASE_SECONDS,
    http_session,
    run_download,
)
from corpus.metrics.emit import emit, start_run
from corpus.run_id import default_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--run-id", default=None, help="Airflow's run_id; generated if omitted")
    parser.add_argument(
        "--base-url", default=None, help="override the source host (kill-restart proof only)"
    )
    parser.add_argument(
        "--stale-lease-seconds",
        type=int,
        default=None,
        help="reclaim claims older than this (default 15 min; the proof uses 1)",
    )
    args = parser.parse_args()
    run_id = args.run_id or default_run_id()

    settings = get_settings()
    if not isinstance(settings, LocalSettings):
        # staging_dir and the S3 client are local-profile concepts.
        raise NotImplementedError("downloading on Azure arrives in Sprint 10")

    with connect() as conn:
        # Safe on a retry: start_run reopens the same row rather than adding one.
        start_run(conn, run_id, args.crawl_id)
        control = SegmentControl(conn)
        summary = run_download(
            control,
            http_session(),
            args.crawl_id,
            staging_dir=Path(settings.staging_dir),
            threads=args.threads,
            base_url=args.base_url,  # None means the real data.commoncrawl.org
            stale_lease_seconds=(
                STALE_LEASE_SECONDS
                if args.stale_lease_seconds is None
                else args.stale_lease_seconds
            ),
        )
        counts = control.status_counts(args.crawl_id)
        emit(
            conn,
            run_id,
            args.crawl_id,
            "bronze",
            {
                "segments_intended": summary.intended,
                "segments_complete": counts.get("complete", 0),  # total, not just this run
                "segments_failed": counts.get("failed", 0),
                "bytes_downloaded": summary.bytes_downloaded,
                "duration_seconds": round(summary.duration_seconds, 1),
            },
        )

    print(
        f"download {args.crawl_id}: {summary.complete} complete, {summary.failed} failed, "
        f"{summary.bytes_downloaded / 1e9:.2f} GB in {summary.duration_seconds:.0f}s, "
        f"statuses {counts}"
    )


if __name__ == "__main__":
    main()
