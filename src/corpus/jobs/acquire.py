"""Stage 0 entry point: decide which segments this run consists of (Stories 1.1-1.2).

    python -m corpus.jobs.acquire --crawl-id CC-MAIN-2026-39 --segments 50 --seed 42

Touches no files: it reads the crawl's manifest and writes one pending row per
chosen segment. Running it twice changes nothing (ON CONFLICT DO NOTHING).
"""

import argparse

from corpus.acquisition.control import SegmentControl
from corpus.acquisition.manifest import fetch_wet_paths, sample_segments
from corpus.db import connect
from corpus.metrics.emit import emit, start_run
from corpus.run_id import default_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True, help="e.g. CC-MAIN-2026-39")
    parser.add_argument("--segments", type=int, required=True, help="how many to sample (N)")
    # The seed makes a run reproducible and keeps the dev sample a prefix of the
    # full one (T6). It is stored in pipeline_runs, so a run stays reproducible.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None, help="Airflow's run_id; generated if omitted")
    args = parser.parse_args()
    run_id = args.run_id or default_run_id()

    paths = fetch_wet_paths(args.crawl_id)
    sample = sample_segments(paths, args.segments, args.seed)

    with connect() as conn:
        # Open the run first: the seed and segment count are what make it reproducible.
        start_run(conn, run_id, args.crawl_id, segment_count=args.segments, sample_seed=args.seed)
        control = SegmentControl(conn)
        inserted = control.register_intent(args.crawl_id, sample)
        counts = control.status_counts(args.crawl_id)
        emit(conn, run_id, args.crawl_id, "bronze", {"segments_intended": len(sample)})

    print(
        f"acquire {args.crawl_id} (run {run_id}): manifest has {len(paths)} segments, "
        f"sampled {len(sample)} with seed {args.seed}, {inserted} new rows, statuses {counts}"
    )


if __name__ == "__main__":
    main()
