"""corpus.acquisition.manifest: manifest parsing and reproducible sampling (task 1.1.3)."""

import gzip

import pytest

from corpus.acquisition.manifest import (
    BASE_URL,
    Segment,
    parse_wet_paths,
    sample_segments,
)

# Shaped like the real manifest: relative paths, one per line.
PATHS = [
    f"crawl-data/CC-MAIN-2026-39/segments/1788492699{i % 7}.6"
    f"/wet/CC-MAIN-x-{i % 1000:05d}.warc.wet.gz"
    for i in range(2_000)
]


def test_parse_wet_paths_drops_blank_lines() -> None:
    raw = gzip.compress(b"a/one.wet.gz\n\nb/two.wet.gz\n  \n")
    assert parse_wet_paths(raw) == ["a/one.wet.gz", "b/two.wet.gz"]


def test_same_seed_gives_the_same_sample() -> None:
    assert sample_segments(PATHS, 50, seed=7) == sample_segments(PATHS, 50, seed=7)


def test_different_seed_gives_a_different_sample() -> None:
    assert sample_segments(PATHS, 50, seed=7) != sample_segments(PATHS, 50, seed=8)


def test_dev_sample_is_a_prefix_of_the_full_sample() -> None:
    """The 5-segment dev run must work on a subset of the 50-segment run (T6)."""
    assert sample_segments(PATHS, 5, seed=7) == sample_segments(PATHS, 50, seed=7)[:5]


def test_sample_is_not_the_start_of_the_manifest() -> None:
    """Sequential selection is forbidden: it biases the domain mix."""
    sample = sample_segments(PATHS, 50, seed=7)
    assert [s.path for s in sample] != PATHS[:50]


def test_segment_id_is_the_zero_padded_position_in_the_manifest() -> None:
    for segment in sample_segments(PATHS, 20, seed=7):
        assert len(segment.segment_id) == 5
        assert PATHS[int(segment.segment_id)] == segment.path


def test_sample_has_no_duplicates() -> None:
    sample = sample_segments(PATHS, 200, seed=7)
    assert len({s.segment_id for s in sample}) == 200


def test_source_url_is_the_path_under_the_public_base_url() -> None:
    segment = Segment("00042", "crawl-data/x/wet/file.warc.wet.gz")
    assert segment.source_url == f"{BASE_URL}/crawl-data/x/wet/file.warc.wet.gz"


def test_segment_round_trips_through_its_source_url() -> None:
    """The control table stores the URL; claim_pending rebuilds the Segment from it."""
    segment = Segment("00042", "crawl-data/x/wet/file.warc.wet.gz")
    assert Segment.from_source_url("00042", segment.source_url) == segment


def test_from_source_url_refuses_a_foreign_url() -> None:
    with pytest.raises(ValueError, match="not a"):
        Segment.from_source_url("00042", "https://example.com/crawl-data/x.wet.gz")


@pytest.mark.parametrize("bad_n", [0, -1, 2_001])
def test_n_outside_the_manifest_is_refused(bad_n: int) -> None:
    with pytest.raises(ValueError, match="n must be"):
        sample_segments(PATHS, bad_n, seed=7)
