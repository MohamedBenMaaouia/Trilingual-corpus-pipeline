"""corpus.silver.parse.parse_files: the Spark wiring, on the real binaryFile source."""

from collections.abc import Callable
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.silver.parse import PARSED_SCHEMA, parse_files

SAMPLE = Path(__file__).parents[2] / "fixtures" / "sample.wet.gz"

WetRecord = Callable[..., bytes]  # the `wet_record` fixture, from conftest.py


def read_files(spark: SparkSession, *paths: Path) -> DataFrame:
    return spark.read.format("binaryFile").load([str(p) for p in paths])


def test_output_has_the_declared_schema(spark: SparkSession) -> None:
    assert parse_files(read_files(spark, SAMPLE)).schema == PARSED_SCHEMA


def test_fixture_gives_three_documents_and_no_dead_letters(spark: SparkSession) -> None:
    parsed = parse_files(read_files(spark, SAMPLE))
    rows = parsed.select("url", "domain", "exception", "path").collect()
    assert sorted(r.url for r in rows) == [
        "http://01xq.com/xqplayer/xqplayer.asp?pid=110764",
        "http://comblan.com/village-martyr-comblanchien/photos-1945/vue-du-village-detruit/",
        "http://www.7awaya.com/poetPage.do;jsessionid=DD8B5DC82D9B8BFCEEBBE5AA03358BB1?poetId=365",
    ]
    assert sorted(r.domain for r in rows) == ["01xq.com", "7awaya.com", "comblan.com"]
    assert all(r.exception is None for r in rows)
    assert all(r.path.endswith("sample.wet.gz") for r in rows)  # lineage to the file


def test_timestamps_and_lengths_survive_the_trip_through_arrow(spark: SparkSession) -> None:
    parsed = parse_files(read_files(spark, SAMPLE))
    row = (
        parsed.where(F.col("doc_id") == "dbc0bc3a-c7db-4a16-a82d-420a7eb85610")
        .select(F.col("fetch_date").cast("string").alias("fetch_date"), "content_length")
        .first()
    )
    assert row is not None
    assert row.fetch_date == "2026-09-04 14:40:25"  # the session time zone is UTC
    assert row.content_length == 3496


def test_a_bad_record_becomes_a_dead_letter_row(
    spark: SparkSession, wet_record: WetRecord, tmp_path: Path
) -> None:
    mixed = tmp_path / "mixed.wet.gz"
    bad_body = b"caf\xe9"  # not valid UTF-8
    mixed.write_bytes(wet_record(b"good one") + wet_record(bad_body) + wet_record(b"good two"))

    parsed = parse_files(read_files(spark, mixed))
    documents = parsed.where(F.col("exception").isNull())
    dead = parsed.where(F.col("exception").isNotNull()).collect()

    assert documents.count() == 2
    assert len(dead) == 1
    assert dead[0].exception.startswith("UnicodeDecodeError")
    assert bytes(dead[0].raw) == bad_body
    assert dead[0].doc_id is None and dead[0].text is None  # document columns stay empty
    assert dead[0].record_offset > 0
    assert dead[0].path.endswith("mixed.wet.gz")


def test_a_file_of_only_bad_records(
    spark: SparkSession, wet_record: WetRecord, tmp_path: Path
) -> None:
    """Every document column is empty for the whole chunk: types must still hold."""
    bad = tmp_path / "bad.wet.gz"
    bad.write_bytes(wet_record(b"\xff\xfe") + wet_record(b"ok", record_id="<urn:uuid:nope>"))
    rows = parse_files(read_files(spark, bad)).collect()
    assert len(rows) == 2
    assert all(r.exception is not None and r.fetch_date is None for r in rows)
