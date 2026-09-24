"""corpus.store against the real MinIO: upload, promote, and the temp copy going away."""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from corpus.store import delete_object, object_exists, object_size, promote, put_file

pytestmark = pytest.mark.integration

CONTENT = b"pretend this is a validated WET file" * 100


@pytest.fixture
def uris() -> Iterator[tuple[str, str]]:
    """A throwaway temp/final pair in corpus-meta, removed afterwards."""
    name = f"_test/{uuid.uuid4().hex}.wet.gz"
    temp = f"s3a://corpus-meta/{name}.tmp"
    final = f"s3a://corpus-meta/{name}"
    yield temp, final
    for uri in (temp, final):
        if object_exists(uri):
            delete_object(uri)


def test_upload_then_promote_publishes_exactly_once(uris: tuple[str, str], tmp_path: Path) -> None:
    temp, final = uris
    local = tmp_path / "staged.wet.gz"
    local.write_bytes(CONTENT)

    put_file(local, temp)
    assert object_exists(temp) is True
    assert object_exists(final) is False  # nothing visible under the real name yet

    promote(temp, final)
    assert object_exists(final) is True
    assert object_size(final) == len(CONTENT)
    assert object_exists(temp) is False  # the temp copy is gone (no _tmp leftovers)


def test_object_exists_is_false_for_something_never_written() -> None:
    assert object_exists(f"s3a://corpus-meta/_test/{uuid.uuid4().hex}") is False
