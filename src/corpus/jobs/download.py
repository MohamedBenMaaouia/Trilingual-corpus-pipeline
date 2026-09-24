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
from corpus.ingest.download import DEFAULT_THREADS, http_session, run_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = parser.parse_args()

    settings = get_settings()
    if not isinstance(settings, LocalSettings):
        # staging_dir and the S3 client are local-profile concepts.
        raise NotImplementedError("downloading on Azure arrives in Sprint 10")

    with connect() as conn:
        control = SegmentControl(conn)
        summary = run_download(
            control,
            http_session(),
            args.crawl_id,
            staging_dir=Path(settings.staging_dir),
            threads=args.threads,
        )
        counts = control.status_counts(args.crawl_id)

    print(
        f"download {args.crawl_id}: {summary.complete} complete, {summary.failed} failed, "
        f"{summary.bytes_downloaded / 1e9:.2f} GB in {summary.duration_seconds:.0f}s, "
        f"statuses {counts}"
    )


if __name__ == "__main__":
    main()
