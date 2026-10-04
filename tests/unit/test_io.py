"""corpus.io: path layout and input validation."""

from collections.abc import Iterator

import pytest

from corpus.config import get_settings
from corpus.io import (
    bronze_object,
    bronze_path,
    bronze_tmp_object,
    dead_letter_path,
    dedup_path,
    lid_model_path,
    silver_stage1_path,
    spark_events_path,
    split_s3a,
)
from corpus.silver.language import MODEL_FILE


@pytest.fixture(autouse=True)
def local_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CORPUS_ENV", "local")
    monkeypatch.setenv("CORPUS_S3_ACCESS_KEY", "test-user")
    monkeypatch.setenv("CORPUS_S3_SECRET_KEY", "test-secret")
    monkeypatch.setenv("CORPUS_INGEST_S3_ACCESS_KEY", "test-ingest-user")
    monkeypatch.setenv("CORPUS_INGEST_S3_SECRET_KEY", "test-ingest-secret")
    monkeypatch.setenv("CORPUS_METRICS_DSN", "postgresql://u:p@postgres:5432/corpus")
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


def test_bronze_object_and_its_temporary_name() -> None:
    final = bronze_object("CC-MAIN-2026-39", 42, "CC-MAIN-x-00042.warc.wet.gz")
    temp = bronze_tmp_object("CC-MAIN-2026-39", 42, "CC-MAIN-x-00042.warc.wet.gz")
    assert final == (
        "s3a://corpus-bronze/common_crawl/crawl_id=CC-MAIN-2026-39/segment=00042"
        "/CC-MAIN-x-00042.warc.wet.gz"
    )
    assert temp == final.replace("/CC-MAIN-x", "/_tmp/CC-MAIN-x")


@pytest.mark.parametrize("bad", ["../../etc/passwd", "sub/dir.gz", "", ".", ".."])
def test_object_names_cannot_escape_their_folder(bad: str) -> None:
    """The name comes from the crawl manifest, which is untrusted input (S1-03)."""
    with pytest.raises(ValueError, match="plain file name"):
        bronze_object("CC-MAIN-2026-39", 1, bad)


def test_split_s3a_gives_bucket_and_key() -> None:
    assert split_s3a("s3a://corpus-bronze/common_crawl/x.gz") == (
        "corpus-bronze",
        "common_crawl/x.gz",
    )


@pytest.mark.parametrize("bad", ["s3://b/k", "https://b/k", "s3a://bucket-only", "s3a://"])
def test_split_s3a_refuses_anything_else(bad: str) -> None:
    with pytest.raises(ValueError):
        split_s3a(bad)


def test_event_log_path() -> None:
    assert spark_events_path() == "s3a://corpus-meta/spark-events"


# Silver output and dead letters (Story 2.4, S2-10) -------------------------------------

CRAWL = "CC-MAIN-2026-39"


def test_silver_stage1_layout() -> None:
    assert silver_stage1_path(CRAWL, dev=False) == (
        "s3a://corpus-silver/_stage1/crawl_id=CC-MAIN-2026-39"
    )
    assert silver_stage1_path(CRAWL, dev=True) == (
        "s3a://corpus-silver/_dev/_stage1/crawl_id=CC-MAIN-2026-39"
    )


def test_dead_letter_layout() -> None:
    assert dead_letter_path("silver", CRAWL, dev=False) == (
        "s3a://corpus-meta/dead_letter/silver/crawl_id=CC-MAIN-2026-39"
    )
    assert dead_letter_path("silver", CRAWL, dev=True) == (
        "s3a://corpus-meta/_dev/dead_letter/silver/crawl_id=CC-MAIN-2026-39"
    )


def test_dev_and_full_outputs_never_overlap() -> None:
    """Neither path is inside the other, so overwriting one can never touch the other (T11)."""
    full, dev = silver_stage1_path(CRAWL, dev=False), silver_stage1_path(CRAWL, dev=True)
    assert not dev.startswith(full + "/") and not full.startswith(dev + "/")


def test_output_paths_reject_a_bad_crawl_id() -> None:
    with pytest.raises(ValueError):
        silver_stage1_path("CC-MAIN-2026", dev=False)
    with pytest.raises(ValueError):
        dead_letter_path("silver", "../etc", dev=False)


def test_dead_letters_only_for_known_stages() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        dead_letter_path("silvr", CRAWL, dev=False)


def test_dedup_layout() -> None:
    """One crawl's folder beside gold (S4-06); dev runs under _dev (T11)."""
    assert dedup_path(CRAWL, dev=False) == "s3a://corpus-gold/dedup/crawl_id=CC-MAIN-2026-39"
    assert dedup_path(CRAWL, dev=True) == "s3a://corpus-gold/_dev/dedup/crawl_id=CC-MAIN-2026-39"
    with pytest.raises(ValueError):
        dedup_path("CC-MAIN-2026-39/..", dev=False)


# Language ID model (Story 3.1) ---------------------------------------------------------


def test_lid_model_path() -> None:
    assert lid_model_path() == "s3a://corpus-meta/models/lid.176.bin"


def test_lid_model_file_name_is_what_the_executors_look_for() -> None:
    # addFile keeps the file name; SparkFiles.get(MODEL_FILE) must find it on the executor.
    assert lid_model_path().rsplit("/", 1)[1] == MODEL_FILE
