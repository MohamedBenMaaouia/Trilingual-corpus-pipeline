"""corpus.session: local configuration and Databricks reuse. No Spark is started here;
the real local session is exercised end to end by the smoke job (Story 0.5)."""

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
    monkeypatch.setenv("CORPUS_METRICS_DSN", "postgresql://u:p@postgres:5432/corpus")


def test_local_conf_targets_the_docker_cluster(local_env: None) -> None:
    settings = get_settings()
    assert isinstance(settings, LocalSettings)
    conf = local_conf(settings)
    assert conf["spark.master"] == "spark://spark-master:7077"
    assert conf["spark.driver.host"] == "airflow-scheduler"
    assert conf["spark.eventLog.enabled"] == "true"
    assert conf["spark.eventLog.dir"] == "s3a://corpus-meta/spark-events"
    assert conf["spark.hadoop.fs.s3a.endpoint"] == "http://minio:9000"
    assert conf["spark.hadoop.fs.s3a.path.style.access"] == "true"
    assert conf["spark.sql.extensions"] == "io.delta.sql.DeltaSparkSessionExtension"


def test_local_conf_uses_the_service_user_keys(local_env: None) -> None:
    settings = get_settings()
    assert isinstance(settings, LocalSettings)
    conf = local_conf(settings)
    assert conf["spark.hadoop.fs.s3a.access.key"] == "test-user"
    assert conf["spark.hadoop.fs.s3a.secret.key"] == "test-secret"


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
