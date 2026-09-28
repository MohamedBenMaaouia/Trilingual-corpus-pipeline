"""assert_schema: the check run before every write (invariant 3, plan task 3.4.2)."""

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType


class SchemaMismatch(ValueError):
    """A DataFrame does not match its declared schema: the write must not happen."""


def assert_schema(df: DataFrame, expected: StructType) -> None:
    """Raise unless df has exactly the expected columns: same names, types, nullability,
    and order. Order matters: it is the column order stored in the files."""
    actual = df.schema
    if actual == expected:
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
        if want.nullable != got.nullable:
            problems.append(f"{name!r}: nullable={got.nullable}, expected {want.nullable}")
    if not problems and actual.fieldNames() != expected.fieldNames():
        problems.append(f"column order {actual.fieldNames()}, expected {expected.fieldNames()}")
    raise SchemaMismatch("schema does not match the contract: " + "; ".join(sorted(problems)))
