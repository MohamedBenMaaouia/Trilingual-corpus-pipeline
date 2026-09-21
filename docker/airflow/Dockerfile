# Airflow 3 + everything a Spark driver needs (client mode: the driver runs here).
# Airflow keeps its own environment; the pipeline gets /opt/corpus-venv, never mixed
# (DECISIONS S0-05). The "spark" build context is the corpus-spark image.
FROM apache/airflow:3.3.2-python3.11

USER root

# Java 17 for the driver JVM. The Airflow image is Debian 12 (bookworm).
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /usr/local/bin/uv

# Same venv path as the Spark image: executors launch the Python path the driver
# sends them. Built from this image's own interpreter (/usr/python), same uv.lock.
ENV UV_PROJECT_ENVIRONMENT=/opt/corpus-venv \
    UV_PYTHON=/usr/python/bin/python3.11 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /opt/corpus
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/uv-cache \
    UV_CACHE_DIR=/uv-cache uv sync --frozen --no-dev --extra spark --no-install-project

# The S3A and Delta jars, taken from the Spark image where they were checksum-verified.
ENV SPARK_HOME=/opt/corpus-venv/lib/python3.11/site-packages/pyspark \
    PYSPARK_PYTHON=/opt/corpus-venv/bin/python
COPY --from=spark \
     /opt/corpus-venv/lib/python3.11/site-packages/pyspark/jars/hadoop-aws-3.3.4.jar \
     /opt/corpus-venv/lib/python3.11/site-packages/pyspark/jars/aws-java-sdk-bundle-1.12.262.jar \
     /opt/corpus-venv/lib/python3.11/site-packages/pyspark/jars/delta-spark_2.12-3.3.3.jar \
     /opt/corpus-venv/lib/python3.11/site-packages/pyspark/jars/delta-storage-3.3.3.jar \
     /opt/corpus-venv/lib/python3.11/site-packages/pyspark/jars/

COPY src ./src
RUN --mount=type=cache,target=/uv-cache \
    UV_CACHE_DIR=/uv-cache uv sync --frozen --no-dev --extra spark

# Back to Airflow's user and home. /opt/corpus-venv is deliberately NOT on PATH:
# `airflow` and `python` stay Airflow's; jobs call /opt/corpus-venv/bin/python.
USER airflow
WORKDIR /opt/airflow
