#!/bin/sh
# One-shot MinIO setup, safe to rerun: every step is idempotent.
set -eu

# Connect as root: only this script ever uses the root account.
mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

# One bucket per layer (T1).
for bucket in corpus-bronze corpus-silver corpus-gold corpus-meta; do
  mc mb --ignore-existing "local/$bucket"
done

# S3 has no directories. The history server fails on a prefix with no objects,
# so an empty placeholder makes spark-events/ exist from the start.
printf '' | mc pipe local/corpus-meta/spark-events/.keep

# Service user for Spark (S3A) and the pipeline: read/write on data, no admin rights.
mc admin user add local "$CORPUS_S3_ACCESS_KEY" "$CORPUS_S3_SECRET_KEY"
# Attach only if missing: attaching twice is an error. The mc image has no grep,
# so the check uses the shell's own pattern matching.
user_info=$(mc admin user info local "$CORPUS_S3_ACCESS_KEY")
case "$user_info" in
  *readwrite*) echo "policy readwrite already attached" ;;
  *) mc admin policy attach local readwrite --user "$CORPUS_S3_ACCESS_KEY" ;;
esac

echo "minio-init: done"
