"""tests/fixtures/sample.wet.gz loads (task 0.4.3): 1 warcinfo + en/fr/ar conversion records."""

from pathlib import Path

from fastwarc import ArchiveIterator, WarcRecordType
from pyspark.sql import SparkSession

SAMPLE = Path(__file__).parent.parent / "fixtures" / "sample.wet.gz"


def test_fixture_parses_as_warc() -> None:
    with SAMPLE.open("rb") as stream:
        records = [
            (r.record_type, r.headers.get("WARC-Identified-Content-Language") or "")
            for r in ArchiveIterator(stream, record_types=WarcRecordType.any_type)
        ]
    assert [t for t, _ in records] == [WarcRecordType.warcinfo] + [WarcRecordType.conversion] * 3
    assert [lang.split(",")[0] for _, lang in records[1:]] == ["eng", "fra", "ara"]


def test_spark_reads_the_gzipped_fixture(spark: SparkSession) -> None:
    lines = spark.read.text(str(SAMPLE))
    assert lines.count() > 0
    assert lines.filter(lines.value.startswith("WARC/1.0")).count() == 4  # one per record
