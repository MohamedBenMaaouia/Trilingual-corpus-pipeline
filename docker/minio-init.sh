#!/bin/sh
# One-shot MinIO setup, safe to rerun: every step is idempotent.
# Buckets, the event-log placeholder, and the two service users with their policies.
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

# Least privilege per layer (DECISIONS S2-02): two service users, like two service
# principals on Azure. Neither has admin rights.
#   processing (CORPUS_S3_*):        Spark, history server, clean-*. Reads bronze,
#                                    can never write it; read/write on silver, gold, meta.
#   ingest (CORPUS_INGEST_S3_*):     the downloader only. Read/write on bronze, nothing else.
# "create" replaces a policy that already exists, so editing a JSON file and rerunning
# this script updates the rules.
mc admin policy create local corpus-processing /policies/corpus-processing.json
mc admin policy create local corpus-ingest /policies/corpus-ingest.json

# `mc admin user info` prints "AccessKey: <user>" and "PolicyName: <p1>,<p2>,...".
# Read only the PolicyName line (a user called corpus-ingest must not count as having
# the corpus-ingest policy), then look for ",<policy>," in ",<list>," so the policy
# matches a whole entry in any position. The mc image has no grep, so plain shell.
policies_of() {
  mc admin user info local "$1" | while IFS= read -r line; do
    case "$line" in "PolicyName: "*) echo "${line#PolicyName: }" ;; esac
  done
}
has_policy() {
  case ",$(policies_of "$1")," in
    *",$2,"*) return 0 ;;
    *) return 1 ;;
  esac
}

# Attaching twice is an error, so attach only what is missing; detach only what is there.
grant() {
  if has_policy "$1" "$2"; then echo "$1: $2 already attached"
  else mc admin policy attach local "$2" --user "$1"; fi
}
revoke() {
  if has_policy "$1" "$2"; then mc admin policy detach local "$2" --user "$1"
  else echo "$1: $2 not attached"; fi
}

# "user add" on an existing user just resets its secret to the .env value.
mc admin user add local "$CORPUS_S3_ACCESS_KEY" "$CORPUS_S3_SECRET_KEY"
grant "$CORPUS_S3_ACCESS_KEY" corpus-processing
revoke "$CORPUS_S3_ACCESS_KEY" readwrite   # Sprint 0-1 grant: write access to bronze

mc admin user add local "$CORPUS_INGEST_S3_ACCESS_KEY" "$CORPUS_INGEST_S3_SECRET_KEY"
grant "$CORPUS_INGEST_S3_ACCESS_KEY" corpus-ingest

echo "minio-init: done"
