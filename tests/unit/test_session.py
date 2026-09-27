"""corpus.session: local configuration and Databricks reuse. No Spark is started here;
the real local session is exercised on the cluster by every Spark job (DECISIONS S0-08, S2-02)."""

from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

import corpus.session as session_module
from corpus.config import get_settings
from corpus.config.local import LocalSettings
from corpus.session import get_session, local_conf


@pytest.fixture(autouse=True)
def reset_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_ENV", "local")
    monkeypatch.setenv("CORPUS_S3_ACCESS_KEY", "test-user")
    monkeypatch.setenv("CORPUS_S3_SECRET_KEY", "test-secret")
    monkeypatch.setenv("CORPUS_INGEST_S3_ACCESS_KEY", "test-ingest-user")
    monkeypatch.setenv("CORPUS_INGEST_S3_SECRET_KEY", "test-ingest-secret")
    monkeypatch.setenv("CORPUS_METRICS_DSN", "postgresql://u:p@postgres:5432/corpus")


def test_local_conf_targets_the_docker_cluster(local_env: None) -> None:
    settings = get_settings()
    assert isinstance(settings, LocalSettings)
    conf = local_conf(settings)
    assert conf["spark.master"] == "spark://spark-master:7077"
    assert conf["spark.driver.host"] == "airflow-scheduler"
    assert conf["spark.executor.memory"] == "3g"  # the workers' --memory, not Spark's 1g
    assert conf["spark.eventLog.enabled"] == "true"
    assert conf["spark.eventLog.dir"] == "s3a://corpus-meta/spark-events"
    assert conf["spark.hadoop.fs.s3a.endpoint"] == "http://minio:9000"
    assert conf["spark.hadoop.fs.s3a.path.style.access"] == "true"
    assert conf["spark.sql.extensions"] == "io.delta.sql.DeltaSparkSessionExtension"


def test_spark_gets_the_processing_user_never_the_ingest_user(local_env: None) -> None:
    """Spark must not hold credentials that can write bronze (DECISIONS S2-02)."""
    settings = get_settings()
    assert isinstance(settings, LocalSettings)
    conf = local_conf(settings)
    assert conf["spark.hadoop.fs.s3a.access.key"] == "test-user"
    assert conf["spark.hadoop.fs.s3a.secret.key"] == "test-secret"
    assert not {"test-ingest-user", "test-ingest-secret"} & set(conf.values())


def test_azure_reuses_the_active_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_ENV", "azure")
    monkeypatch.setenv("CORPUS_METRICS_DSN", "postgresql://u:p@host:5432/corpus")
    for layer in ("BRONZE", "SILVER", "GOLD", "META"):
        monkeypatch.setenv(f"CORPUS_{layer}_ROOT", "abfss://x@acct.dfs.core.windows.net")
    runtime_session = object()
    monkeypatch.setattr(SparkSession, "getActiveSession", staticmethod(lambda: runtime_session))
    # If get_session tried to build a session, this would blow up.
    monkeypatch.setattr(session_module, "local_conf", lambda _: pytest.fail("built a session"))
    assert get_session("any-app") is runtime_session
