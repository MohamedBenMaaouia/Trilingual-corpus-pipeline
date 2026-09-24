"""corpus.ingest.download validation: size, gzip completeness, decompression cap."""

import gzip
from pathlib import Path

import pytest

from corpus.ingest.download import DownloadError, validate_gzip, validate_size


def write_multi_member_gzip(path: Path, parts: list[bytes]) -> None:
    """Like a real WET file: several gzip members concatenated, one per record."""
    path.write_bytes(b"".join(gzip.compress(part) for part in parts))


def test_size_matches() -> None:
    validate_size(66_761_801, 66_761_801)  # no exception


def test_size_mismatch_is_refused() -> None:
    with pytest.raises(DownloadError, match="size mismatch"):
        validate_size(1_000, 66_761_801)


def test_missing_content_length_is_refused() -> None:
    """Without the server's size we have nothing to check against."""
    with pytest.raises(DownloadError, match="no Content-Length"):
        validate_size(1_000, None)


def test_complete_gzip_passes_and_reports_its_size(tmp_path: Path) -> None:
    path = tmp_path / "good.wet.gz"
    write_multi_member_gzip(path, [b"WARC/1.0\r\nrecord one\r\n", b"WARC/1.0\r\nrecord two\r\n"])
    assert validate_gzip(path) == len(b"WARC/1.0\r\nrecord one\r\nWARC/1.0\r\nrecord two\r\n")


def test_all_members_are_read_not_only_the_first(tmp_path: Path) -> None:
    """A truncated file often keeps a valid first member; the end is what proves it."""
    path = tmp_path / "two_members.wet.gz"
    write_multi_member_gzip(path, [b"a" * 1000, b"b" * 1000])
    assert validate_gzip(path) == 2000


def test_truncated_gzip_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cut.wet.gz"
    write_multi_member_gzip(path, [b"x" * 10_000])
    path.write_bytes(path.read_bytes()[:-20])  # drop the checksum and length trailer
    with pytest.raises(DownloadError, match="not a complete gzip file"):
        validate_gzip(path)


def test_a_file_that_is_not_gzip_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "html_error_page.gz"
    path.write_text("<html><body>503 Slow Down</body></html>")  # what a proxy might return
    with pytest.raises(DownloadError, match="not a complete gzip file"):
        validate_gzip(path)


def test_compression_bomb_is_refused(tmp_path: Path) -> None:
    """5 MB of zeros compress to a few KB; with a 1 KB cap it must be refused."""
    path = tmp_path / "bomb.wet.gz"
    path.write_bytes(gzip.compress(b"\0" * 5_000_000))
    assert path.stat().st_size < 10_000  # tiny on disk, huge when decompressed
    with pytest.raises(DownloadError, match="decompressed size passed"):
        validate_gzip(path, max_decompressed_bytes=1024)
