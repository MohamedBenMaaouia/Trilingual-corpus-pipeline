"""Fixtures shared by every test. pytest finds this file automatically."""

from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

from corpus.session import DELTA_CONF


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """One real, in-process Spark for the whole test run (starting one costs seconds).

    Deliberately not corpus.session.get_session(): tests must not need the cluster
    or MinIO. local[2] = driver + 2 worker threads in this one process.
    """
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("corpus-tests")
        # Default 200 shuffle partitions would make every tiny test run 200 tasks.
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")  # no web UI needed in tests
        .config("spark.sql.session.timeZone", "UTC")  # same results on every machine
    )
    for key, value in DELTA_CONF.items():
        builder = builder.config(key, value)
    session = builder.getOrCreate()
    yield session
    session.stop()
