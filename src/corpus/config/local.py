"""Local profile: Spark standalone cluster and MinIO in Docker Compose (Sprints 0-9).

Defaults match docker-compose.yml; any of them can be overridden with CORPUS_<FIELD>.
"""

from pydantic import SecretStr

from corpus.config.base import Settings


class LocalSettings(Settings):
    # One MinIO bucket per layer, created by minio-init (T1). "s3a://" makes
    # Spark use Hadoop's S3A connector, which speaks the S3 API that MinIO serves.
    bronze_root: str = "s3a://corpus-bronze"
    silver_root: str = "s3a://corpus-silver"
    gold_root: str = "s3a://corpus-gold"
    meta_root: str = "s3a://corpus-meta"

    # Standalone master; the driver runs in the airflow-scheduler container
    # (client mode, DECISIONS S0-01), and executors connect back to it by this name.
    spark_master: str = "spark://spark-master:7077"
    driver_host: str = "airflow-scheduler"

    # MinIO as the pipeline's own "corpus" user, never root (DECISIONS S0-05).
    # No defaults for the keys: a missing key must fail at startup, not mid-job.
    # Local disk for in-progress downloads (T7): object storage cannot be appended
    # to, so a resumable download needs a real file. A Docker named volume, so a
    # restarted container still finds its partial files.
    staging_dir: str = "/opt/corpus/staging"

    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str
    s3_secret_key: SecretStr
