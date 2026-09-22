"""The shared spark fixture: settings applied, Delta usable, no cluster needed."""

from pathlib import Path

from pyspark.sql import SparkSession


def test_fixture_settings(spark: SparkSession) -> None:
    assert spark.sparkContext.master == "local[2]"
    assert spark.conf.get("spark.sql.shuffle.partitions") == "4"


def test_delta_round_trip(spark: SparkSession, tmp_path: Path) -> None:
    table = str(tmp_path / "delta_table")
    spark.range(10).write.format("delta").save(table)
    assert spark.read.format("delta").load(table).count() == 10
    assert (tmp_path / "delta_table" / "_delta_log" / "00000000000000000000.json").exists()
