"""The schema contract check (Story 7.4.1, DECISIONS S7-05).

Every declared contract is compared with its committed snapshot, the contract as JSON
(names, types, nullability, order) in tests/fixtures/contracts/<name>.json, named after
the contract's version. A contract never changes under the same name: changing one means
a new version (silver_v2, gold_v2, ...), a new snapshot, and for gold a pipeline-version
bump and backfill (invariant 9). Runs as its own CI step, and in every `make test`.
"""

import json
from pathlib import Path

import pytest
from pyspark.sql.types import StructType

from corpus.schemas.dedup_v1 import DEDUP_SCHEMA_VERSION, DEDUP_V1
from corpus.schemas.gold_v1 import GOLD_SCHEMA_VERSION, GOLD_V1
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.schemas.silver_v1 import SCHEMA_VERSION, SILVER_V1

SNAPSHOTS = Path(__file__).parents[2] / "fixtures" / "contracts"

CONTRACTS: dict[str, StructType] = {
    "silver_stage1": STAGE1_SCHEMA,  # Sprint 2's interim output: no version column
    SCHEMA_VERSION: SILVER_V1,
    DEDUP_SCHEMA_VERSION: DEDUP_V1,
    GOLD_SCHEMA_VERSION: GOLD_V1,
}


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_a_contract_matches_its_snapshot(name: str) -> None:
    declared = CONTRACTS[name].jsonValue()
    committed = json.loads((SNAPSHOTS / f"{name}.json").read_text(encoding="utf-8"))
    assert declared == committed, (
        f"the {name} contract no longer matches tests/fixtures/contracts/{name}.json. "
        "A contract never changes under the same name: declare a new version instead. "
        f"The declared contract is:\n{json.dumps(declared, indent=2)}"
    )


def test_every_snapshot_belongs_to_a_contract() -> None:
    """A contract removed from the code must not leave its snapshot behind."""
    assert {path.stem for path in SNAPSHOTS.glob("*.json")} == set(CONTRACTS)


@pytest.mark.parametrize("schema", [SILVER_V1, DEDUP_V1, GOLD_V1])
def test_every_versioned_row_says_which_contract_and_logic_made_it(schema: StructType) -> None:
    """Invariant 3: schema_version and pipeline_version on every row."""
    assert {"schema_version", "pipeline_version"} <= set(schema.fieldNames())


def test_gold_takes_every_column_from_silver_or_dedup_with_its_type() -> None:
    """gold_v1 is a projection (C24, S6-03): no column is invented or retyped on the way."""
    sources = {f.name: f.dataType for f in [*SILVER_V1.fields, *DEDUP_V1.fields]}
    for field in GOLD_V1.fields:
        assert sources.get(field.name) == field.dataType, field.name
