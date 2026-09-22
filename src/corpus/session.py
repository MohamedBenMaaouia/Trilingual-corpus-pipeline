"""The only place a SparkSession is created (CLAUDE.md §8)."""

from pyspark.sql import SparkSession

from corpus.config import get_settings
from corpus.config.local import LocalSettings
from corpus.io import spark_events_path


def get_session(app_name: str) -> SparkSession:
    """Return a session. On Databricks, reuse the runtime's active session."""
    settings = get_settings()
    if isinstance(settings, LocalSettings):
        builder = SparkSession.builder.appName(app_name)
        for key, value in local_conf(settings).items():
            builder = builder.config(key, value)
        return builder.getOrCreate()
    # Databricks starts Spark before our code runs. getActiveSession() can be None
    # in a fresh wheel-task process; getOrCreate() then attaches to the runtime's.
    return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()


def local_conf(settings: LocalSettings) -> dict[str, str]:
    """Spark settings for the Docker cluster. Proven by the S0 probe (DECISIONS S0-01)."""
    return {
        # Client mode: the driver runs where the job is launched (airflow-scheduler)
        # and executors on the workers connect back to it by this hostname.
        "spark.master": settings.spark_master,
        "spark.driver.host": settings.driver_host,
        # Event logs are a driver setting; the history server reads the same folder.
        "spark.eventLog.enabled": "true",
        "spark.eventLog.dir": spark_events_path(),
        # Delta Lake: SQL extensions + a catalog that understands Delta tables.
        "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
        "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        # S3A -> MinIO. Path-style: bucket in the path (minio:9000/corpus-bronze/...),
        # because MinIO has no per-bucket hostnames (corpus-bronze.minio:9000).
        "spark.hadoop.fs.s3a.endpoint": settings.s3_endpoint,
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        # The "corpus" user's keys. Spark's default redaction hides both in the UI
        # and in event logs (spark.redaction.regex matches "secret" and "access.key").
        "spark.hadoop.fs.s3a.access.key": settings.s3_access_key,
        "spark.hadoop.fs.s3a.secret.key": settings.s3_secret_key.get_secret_value(),
    }
