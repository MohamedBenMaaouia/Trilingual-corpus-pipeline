"""corpus.schemas.contract.assert_schema: the check before every write (invariant 3)."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType

from corpus.schemas.contract import SchemaMismatch, assert_schema

EXPECTED = StructType(
    [StructField("doc_id", StringType(), False), StructField("n", LongType(), True)]
)


def test_a_matching_dataframe_passes(spark: SparkSession) -> None:
    assert_schema(spark.createDataFrame([("a", 1)], EXPECTED), EXPECTED)


@pytest.mark.parametrize(
    ("actual", "message"),
    [
        (StructType([StructField("doc_id", StringType(), False)]), "missing column 'n'"),
        (
            StructType([*EXPECTED.fields, StructField("extra", StringType(), True)]),
            "unexpected column 'extra'",
        ),
        (
            StructType([EXPECTED.fields[0], StructField("n", StringType(), True)]),
            "'n': type string, expected bigint",
        ),
        (
            StructType([StructField("doc_id", StringType(), True), EXPECTED.fields[1]]),
            "'doc_id': nullable=True, expected False",
        ),
        (StructType([EXPECTED.fields[1], EXPECTED.fields[0]]), "column order"),
    ],
)
def test_every_kind_of_mismatch_is_refused(
    spark: SparkSession, actual: StructType, message: str
) -> None:
    with pytest.raises(SchemaMismatch, match=message):
        assert_schema(spark.createDataFrame([], actual), EXPECTED)
