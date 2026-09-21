#!/bin/sh
# Empties one layer bucket: make clean-silver / make clean-gold.
# Bronze is refused by name (invariant 1: bronze is append-only and immutable).
set -eu

bucket="${1:-}"
case "$bucket" in
  corpus-silver|corpus-gold) ;;
  corpus-bronze) echo "refused: bronze is immutable and is never cleaned (invariant 1)" >&2; exit 1 ;;
  *) echo "usage: minio-clean.sh corpus-silver|corpus-gold" >&2; exit 1 ;;
esac

# The pipeline's own user, not root: it can delete objects, not buckets or users.
mc alias set svc http://minio:9000 "$CORPUS_S3_ACCESS_KEY" "$CORPUS_S3_SECRET_KEY" >/dev/null
mc rm --recursive --force "svc/$bucket/"
echo "minio-clean: $bucket emptied"
