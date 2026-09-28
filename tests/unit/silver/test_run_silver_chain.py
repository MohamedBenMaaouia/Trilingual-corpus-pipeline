"""corpus.jobs.run_silver.build_stage1: the whole Sprint 2 chain, end to end, locally.

Runs on the committed fixture plus a hand-made file with a bad record, writing into a
temporary folder (no MinIO). Spark reports a local file's path as "file:/...", which
is what the control table would hold here; on MinIO both sides say "s3a://...".
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from corpus.jobs.run_silver import build_stage1
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA

SAMPLE = Path(__file__).parents[2] / "fixtures" / "sample.wet.gz"
CRAWL = "CC-MAIN-2026-39"

WetRecord = Callable[..., bytes]  # the `wet_record` fixture, from conftest.py


@pytest.fixture
def files(wet_record: WetRecord, tmp_path: Path) -> list[tuple[str, str]]:
    """Two bronze files as the control table lists them: (segment_id, path)."""
    mixed = tmp_path / "mixed.wet.gz"
    mixed.write_bytes(
        wet_record(b"The committee met on Friday and approved the budget.")
        + wet_record(b"caf\xe9")  # not UTF-8: a dead letter
        + wet_record(b"It will meet again next month to review progress.")
    )
    return [("00048", f"file:{SAMPLE.resolve()}"), ("01326", f"file:{mixed}")]


def test_the_chain_writes_documents_and_dead_letters(
    spark: SparkSession, files: list[tuple[str, str]], tmp_path: Path
) -> None:
    stage1_out, dead_out = str(tmp_path / "stage1"), str(tmp_path / "dead")
    metrics = build_stage1(spark, files, CRAWL, stage1_out, dead_out)

    assert metrics["files_read"] == 2
    assert metrics["documents"] == 5  # 3 in the fixture + 2 good hand-made records
    assert metrics["dead_letters"] == 1
    assert metrics["documents_written"] == 5

    written = spark.read.parquet(stage1_out)
    # Parquet reads every column back as nullable, so compare names and types.
    assert [(f.name, f.dataType) for f in written.schema.fields] == [
        (f.name, f.dataType) for f in STAGE1_SCHEMA.fields
    ]
    by_segment = {r.segment_id: r["count"] for r in written.groupBy("segment_id").count().collect()}
    assert by_segment == {"00048": 3, "01326": 2}
    assert written.where(F.col("crawl_id") != CRAWL).count() == 0

    dead = spark.read.parquet(dead_out).collect()
    assert len(dead) == 1
    assert dead[0].exception.startswith("UnicodeDecodeError")
    assert bytes(dead[0].raw) == b"caf\xe9"


def test_a_rerun_replaces_instead_of_adding(
    spark: SparkSession, files: list[tuple[str, str]], tmp_path: Path
) -> None:
    """Invariants 6 and 7: the same crawl written twice holds the rows once."""
    stage1_out, dead_out = str(tmp_path / "stage1"), str(tmp_path / "dead")
    build_stage1(spark, files, CRAWL, stage1_out, dead_out)
    build_stage1(spark, files, CRAWL, stage1_out, dead_out)
    assert spark.read.parquet(stage1_out).count() == 5
    assert spark.read.parquet(dead_out).count() == 1


def test_documents_that_cannot_be_traced_to_a_segment_are_never_written(
    spark: SparkSession, tmp_path: Path
) -> None:
    """A path that does not match the control table (here: no "file:" scheme) is refused
    before anything is written."""
    stage1_out, dead_out = tmp_path / "stage1", tmp_path / "dead"
    with pytest.raises(RuntimeError, match="no segment_id"):
        build_stage1(
            spark, [("00048", str(SAMPLE.resolve()))], CRAWL, str(stage1_out), str(dead_out)
        )
    assert not stage1_out.exists() and not dead_out.exists()
