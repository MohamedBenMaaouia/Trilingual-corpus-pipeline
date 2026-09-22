"""corpus.io: path layout and input validation."""

from collections.abc import Iterator

import pytest

from corpus.config import get_settings
from corpus.io import bronze_path, smoke_output_path, spark_events_path


@pytest.fixture(autouse=True)
def local_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CORPUS_ENV", "local")
    monkeypatch.setenv("CORPUS_S3_ACCESS_KEY", "test-user")
    monkeypatch.setenv("CORPUS_S3_SECRET_KEY", "test-secret")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_bronze_path_layout() -> None:
    assert (
        bronze_path("CC-MAIN-2026-18", 42)
        == "s3a://corpus-bronze/common_crawl/crawl_id=CC-MAIN-2026-18/segment=00042"
    )


def test_bronze_path_follows_the_configured_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_BRONZE_ROOT", "abfss://corpus-bronze@acct.dfs.core.windows.net")
    get_settings.cache_clear()
    assert bronze_path("CC-MAIN-2026-18", 0).startswith("abfss://corpus-bronze@acct")


@pytest.mark.parametrize("bad", ["CC-MAIN-2026-5", "cc-main-2026-18", "CC-MAIN-2026-18/..", ""])
def test_bronze_path_rejects_bad_crawl_ids(bad: str) -> None:
    with pytest.raises(ValueError, match="Common Crawl id"):
        bronze_path(bad, 1)


@pytest.mark.parametrize("bad", [-1, 100_000])
def test_bronze_path_rejects_out_of_range_segments(bad: int) -> None:
    with pytest.raises(ValueError, match="segment"):
        bronze_path("CC-MAIN-2026-18", bad)


def test_smoke_and_event_log_paths() -> None:
    assert smoke_output_path() == "s3a://corpus-silver/smoke"
    assert spark_events_path() == "s3a://corpus-meta/spark-events"
