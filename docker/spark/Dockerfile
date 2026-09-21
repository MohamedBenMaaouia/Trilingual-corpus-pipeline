# One image for spark-master, spark-workers, spark-history and the test runner.
# Spark itself comes from the pip pyspark package pinned in uv.lock (DECISIONS S0-05).

# --- Layer 1: OS, Python 3.11, Java 17 ---------------------------------------
# bookworm (Debian 12): the last Debian release that packages OpenJDK 17.
FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps curl \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# --- Layer 2: corpus[spark] from uv.lock --------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /usr/local/bin/uv

# Same venv path as the Airflow image: executors run the Python path the driver
# sends them, so both images must have the interpreter at the same place.
ENV UV_PROJECT_ENVIRONMENT=/opt/corpus-venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/corpus-venv/bin:$PATH

WORKDIR /opt/corpus

# Dependencies first, project second: editing src/ doesn't reinstall pyspark.
# The cache mount keeps uv's download cache out of the image (it was 1.8 GB).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/uv-cache \
    UV_CACHE_DIR=/uv-cache uv sync --frozen --no-dev --extra spark --no-install-project

# --- Layer 3: S3A and Delta jars, baked in and checksum-verified ------------
# The pip pyspark package is a complete Spark: its folder is SPARK_HOME, and
# everything in jars/ is on the classpath of every Spark process.
ENV SPARK_HOME=/opt/corpus-venv/lib/python3.11/site-packages/pyspark

# hadoop-aws must equal PySpark's bundled Hadoop client (3.3.4); the SDK bundle
# is the version Hadoop 3.3.4 was built against. Delta 3.3.3 targets Spark 3.5,
# Scala 2.12. SHA-1s from Maven Central (DECISIONS S0-02).
ARG MAVEN=https://repo1.maven.org/maven2
RUN cd "$SPARK_HOME/jars" \
 && curl -fsSLO "$MAVEN/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar" \
 && curl -fsSLO "$MAVEN/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar" \
 && curl -fsSLO "$MAVEN/io/delta/delta-spark_2.12/3.3.3/delta-spark_2.12-3.3.3.jar" \
 && curl -fsSLO "$MAVEN/io/delta/delta-storage/3.3.3/delta-storage-3.3.3.jar" \
 && printf '%s  %s\n' \
      a65839fbf1869f81a1632e09f415e586922e4f80 hadoop-aws-3.3.4.jar \
      02deec3a0ad83d13d032b1812421b23d7a961eea aws-java-sdk-bundle-1.12.262.jar \
      8083a378e42f6d04c5a78cb6e8566f59f4e9ed1c delta-spark_2.12-3.3.3.jar \
      a5bf8cebb3a669f83b14a8f4a707545ed861303d delta-storage-3.3.3.jar \
    | sha1sum -c -

# --- Layer 4: how services start ----------------------------------------------
# spark-class (in $SPARK_HOME/bin) runs one Spark daemon in the foreground;
# docker-compose.yml picks which one per service. Executors run the venv Python.
ENV PATH=$SPARK_HOME/bin:$PATH \
    PYSPARK_PYTHON=/opt/corpus-venv/bin/python

# Copied last so editing src/ only rebuilds this small layer, not the jars.
COPY src ./src
RUN --mount=type=cache,target=/uv-cache \
    UV_CACHE_DIR=/uv-cache uv sync --frozen --no-dev --extra spark
