#!/usr/bin/env bash
# The CI steps (Story 7.4, DECISIONS S7-05), one function each, so GitHub Actions
# (.github/workflows/ci.yml) and a rehearsal on a developer's machine run the same commands:
#   .github/ci.sh all       every step in order (locally: from a clean copy of the repo)
#   .github/ci.sh <step>    one step: env build services migrate lint contracts test wheel down
# A separate compose project (corpus-ci) and no published ports (docker-compose.ci.yml):
# it never touches, and can run beside, the developer's own stack and data.
set -euo pipefail

PROJECT="${CI_PROJECT:-corpus-ci}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

compose() {
  docker compose -p "$PROJECT" -f docker-compose.yml -f docker-compose.ci.yml "$@"
}

# A throwaway .env: random values, nothing real. Compose needs every variable it names.
step_env() {
  "$PYTHON" -c '
import base64, os, secrets
values = {
    "CORPUS_ENV": "local",
    "MINIO_ROOT_USER": "ci-root",
    "MINIO_ROOT_PASSWORD": secrets.token_hex(16),
    "CORPUS_S3_ACCESS_KEY": "ci-processing",
    "CORPUS_S3_SECRET_KEY": secrets.token_hex(16),
    "CORPUS_INGEST_S3_ACCESS_KEY": "ci-ingest",
    "CORPUS_INGEST_S3_SECRET_KEY": secrets.token_hex(16),
    "POSTGRES_USER": "ci",
    "POSTGRES_PASSWORD": secrets.token_hex(16),
    "AIRFLOW_JWT_SECRET": secrets.token_hex(32),
    "AIRFLOW_FERNET_KEY": base64.urlsafe_b64encode(os.urandom(32)).decode(),
    "CORPUS_ALERT_WEBHOOK_URL": "",
}
with open(".env", "w", newline="") as env:
    for name, value in values.items():
        print(f"{name}={value}", file=env)
'
}

# The test image: the cluster's image (same Linux, Java, Spark, jars) plus the dev tools.
step_build() { compose build tests; }

# What the integration tests talk to: Postgres, MinIO with its buckets and two users.
step_services() {
  compose up -d --wait postgres minio
  compose run --rm minio-init
}

step_migrate() { compose run --rm tests corpus.db; }

step_lint() {
  compose run --rm tests ruff check .
  compose run --rm tests ruff format --check .
  compose run --rm tests mypy
}

# Every contract equals its committed snapshot (tests/fixtures/contracts).
step_contracts() { compose run --rm tests pytest tests/unit/schemas/test_contract_snapshots.py -q; }

# Unit, integration and golden tests, with coverage. The repo is mounted read-only.
step_test() {
  compose run --rm -e COVERAGE_FILE=/tmp/.coverage tests pytest --cov --cov-report=term -q -p no:warnings
}

# The wheel builds on a clean machine and ships what the jobs read at run time.
step_wheel() {
  local out
  out="$(mktemp -d)"
  uv build --wheel --out-dir "$out"
  "$PYTHON" -c '
import glob, sys, zipfile
wheel = glob.glob(sys.argv[1] + "/corpus-*.whl")[0]
names = set(zipfile.ZipFile(wheel).namelist())
needed = [
    "corpus/config/exclusions.txt",
    "corpus/checks/gate_a.yml",
    "corpus/checks/gate_b.yml",
    "corpus/migrations/006_corpus_stats.sql",
    "corpus/resources/stopwords/ar.txt",
    "corpus/jobs/run_gold.py",
]
missing = [name for name in needed if name not in names]
entry_points = [n for n in names if n.endswith("entry_points.txt")]
scripts = zipfile.ZipFile(wheel).read(entry_points[0]).decode() if entry_points else ""
if missing or "corpus-gold" not in scripts:
    sys.exit(f"wheel incomplete: missing {missing}, entry points: {scripts!r}")
print(f"{wheel}: {len(names)} files, package data and entry points present")
' "$out"
}

step_down() { compose down -v; }

if [ "${1:-}" = "all" ]; then
  trap step_down EXIT
  for step in env build services migrate lint contracts test wheel; do
    echo "=== ci: $step"
    "step_$step"
  done
else
  "step_${1:?usage: ci.sh all|env|build|services|migrate|lint|contracts|test|wheel|down}"
fi
