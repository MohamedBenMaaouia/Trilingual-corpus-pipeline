"""assert_schema: the check run before every write (invariant 3, plan task 3.4.2)."""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType


class SchemaMismatch(ValueError):
    """A DataFrame does not match its declared schema: the write must not happen."""


def assert_schema(df: DataFrame, expected: StructType, *, check_nullability: bool = True) -> None:
    """Raise unless df has exactly the expected columns: same names, types, nullability,
    and order. Order matters: it is the column order stored in the files.

    check_nullability=False (silver_v1, S3-09): Spark marks computed columns nullable
    and stores every Parquet column as nullable, so the flag cannot carry the promise.
    The caller must then prove it by counting nulls (null_count below) before writing.
    """
    actual = df.schema
    if actual == expected or (
        not check_nullability
        and [(f.name, f.dataType) for f in actual.fields]
        == [(f.name, f.dataType) for f in expected.fields]
    ):
        return
    problems = []
    actual_fields = {f.name: f for f in actual.fields}
    expected_fields = {f.name: f for f in expected.fields}
    for name in expected_fields.keys() - actual_fields.keys():
        problems.append(f"missing column {name!r}")
    for name in actual_fields.keys() - expected_fields.keys():
        problems.append(f"unexpected column {name!r}")
    for name in expected_fields.keys() & actual_fields.keys():
        want, got = expected_fields[name], actual_fields[name]
        if want.dataType != got.dataType:
            problems.append(
                f"{name!r}: type {got.dataType.simpleString()}, "
                f"expected {want.dataType.simpleString()}"
            )
        if check_nullability and want.nullable != got.nullable:
            problems.append(f"{name!r}: nullable={got.nullable}, expected {want.nullable}")
    if not problems and actual.fieldNames() != expected.fieldNames():
        problems.append(f"column order {actual.fieldNames()}, expected {expected.fieldNames()}")
    raise SchemaMismatch("schema does not match the contract: " + "; ".join(sorted(problems)))


def null_count(expected: StructType) -> Column:
    """An aggregate: how many nulls sit in the contract's non-null columns, all summed.
    Must be 0 before a write (invariant 3); Gate B checks it again (S3-09)."""
    counts = [F.sum(F.col(f.name).isNull().cast("long")) for f in expected.fields if not f.nullable]
    return sum(counts[1:], counts[0])
