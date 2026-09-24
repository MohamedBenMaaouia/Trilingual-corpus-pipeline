"""corpus.jobs.gate: the Soda connection file is derived from our one DSN."""

import pytest

from corpus.jobs.gate import DATA_SOURCE, soda_configuration


def test_configuration_is_built_from_the_dsn() -> None:
    config = soda_configuration("postgresql://corpus:s3cret@postgres:5432/corpus")
    assert config.splitlines() == [
        f"data_source {DATA_SOURCE}:",
        "  type: postgres",
        "  host: postgres",
        "  port: 5432",
        "  username: corpus",
        "  password: s3cret",
        "  database: corpus",
        "  schema: public",
    ]


def test_a_dsn_without_a_port_gets_the_postgres_default() -> None:
    assert "  port: 5432" in soda_configuration("postgresql://u:p@db/corpus")


@pytest.mark.parametrize(
    ("dsn", "expected_database"),
    [
        ("postgresql://u:p@db:5432/corpus", "corpus"),
        ("postgresql://u:p@db:5432/other_db", "other_db"),
    ],
)
def test_the_database_name_comes_from_the_dsn_path(dsn: str, expected_database: str) -> None:
    assert f"  database: {expected_database}" in soda_configuration(dsn)
