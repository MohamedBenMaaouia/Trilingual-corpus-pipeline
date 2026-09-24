"""Stage 0, part 1: which WET files a crawl has, and which ones this run takes.

The list ("manifest") is published by Common Crawl as one gzipped text file with
one relative path per line: 100,000 lines for CC-MAIN-2026-39. Downloading it is
cheap (~200 KB), so a run always works from the real list, never a stored copy.
"""

import gzip
import random
from collections.abc import Sequence
from typing import NamedTuple

import requests

from corpus.io import check_crawl_id

# Verified against the site on 2026-09-22 (the plan warns this URL has changed before).
BASE_URL = "https://data.commoncrawl.org"
WET_PATHS_URL = BASE_URL + "/crawl-data/{crawl_id}/wet.paths.gz"

# The manifest is small; a slow or hanging server must not block a task forever.
TIMEOUT_SECONDS = (10, 60)  # (connect, read)


class Segment(NamedTuple):
    """One WET file, as this pipeline identifies it.

    segment_id: 5-digit position of the path in the crawl's manifest (T6). It is a
    string, so it keeps its leading zeros and matches the control table's TEXT column.
    """

    segment_id: str
    # Relative path exactly as published, e.g.
    # crawl-data/CC-MAIN-2026-39/segments/1788492699475.64/wet/CC-MAIN-...-00042.warc.wet.gz
    path: str

    @property
    def source_url(self) -> str:
        return f"{BASE_URL}/{self.path}"

    @classmethod
    def from_source_url(cls, segment_id: str, source_url: str) -> "Segment":
        """Rebuild a Segment from what the control table stores (its URL)."""
        if not source_url.startswith(BASE_URL + "/"):
            raise ValueError(f"not a {BASE_URL} url: {source_url!r}")
        return cls(segment_id=segment_id, path=source_url[len(BASE_URL) + 1 :])


def parse_wet_paths(raw_gzip: bytes) -> list[str]:
    """Decompress the manifest and return its lines, blank lines dropped."""
    text = gzip.decompress(raw_gzip).decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def fetch_wet_paths(crawl_id: str) -> list[str]:
    """Download the crawl's manifest. Raises on any HTTP error."""
    check_crawl_id(crawl_id)
    response = requests.get(WET_PATHS_URL.format(crawl_id=crawl_id), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_wet_paths(response.content)


def sample_segments(paths: Sequence[str], n: int, seed: int) -> list[Segment]:
    """Pick n segments at random, reproducibly (T6).

    Shuffle all positions with a seeded generator and take the first n. Two properties
    follow, and both are required:
      - same seed, same result: a run can be repeated exactly (needed for the S7
        backfill proof and S10 parity);
      - prefix stability: the 5-segment dev sample is the start of the 50-segment
        sample, so development works on a subset of the real run. random.sample()
        does NOT guarantee this.

    Sequential selection is forbidden: the manifest is ordered by crawl directory,
    so the first n files would over-represent whatever was crawled first.
    """
    if not 0 < n <= len(paths):
        raise ValueError(f"n must be 1..{len(paths)}, got {n}")
    positions = list(range(len(paths)))
    random.Random(seed).shuffle(positions)
    return [Segment(segment_id=f"{i:05d}", path=paths[i]) for i in positions[:n]]
