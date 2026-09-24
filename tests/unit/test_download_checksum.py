"""corpus.ingest.download: ETag verification, when the ETag is usable as a checksum."""

import hashlib
from pathlib import Path

import pytest

from corpus.ingest.download import DownloadError, md5_of_file, verify_etag


@pytest.fixture
def a_file(tmp_path: Path) -> Path:
    path = tmp_path / "segment.wet.gz"
    path.write_bytes(b"pretend this is a WET file" * 1000)
    return path


def test_md5_matches_an_independent_calculation(a_file: Path) -> None:
    assert md5_of_file(a_file) == hashlib.md5(a_file.read_bytes()).hexdigest()


def test_a_single_part_etag_is_verified(a_file: Path) -> None:
    etag = f'"{hashlib.md5(a_file.read_bytes()).hexdigest()}"'  # how S3 sends it
    assert verify_etag(a_file, etag) is True


def test_an_etag_without_quotes_also_works(a_file: Path) -> None:
    assert verify_etag(a_file, hashlib.md5(a_file.read_bytes()).hexdigest()) is True


def test_a_wrong_checksum_is_refused(a_file: Path) -> None:
    with pytest.raises(DownloadError, match="checksum mismatch"):
        verify_etag(a_file, '"0123456789abcdef0123456789abcdef"')


@pytest.mark.parametrize(
    "not_comparable",
    [
        None,  # no ETag at all
        '"b6db5ec815419c88bdcd56ddb9937dc0-12"',  # multipart: a hash of hashes
        '"W/weak-etag"',
        '""',
    ],
)
def test_an_etag_that_is_not_an_md5_reports_unverified(a_file: Path, not_comparable: str) -> None:
    """False means "could not check", never "checked and fine" (C2: claim nothing)."""
    assert verify_etag(a_file, not_comparable) is False
