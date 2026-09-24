"""Run a quality gate and record its verdict (Story 1.5).

    python -m corpus.jobs.gate --gate gate_a --crawl-id CC-MAIN-2026-39 \
        --expected-segments 50 --run-id manual__2026-09-24T10:00:00

Exits non-zero when a check fails, which stops the DAG (invariant 8). Soda lives in
its own virtualenv, so it is run as a subprocess rather than imported.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

from corpus.config import get_settings
from corpus.db import connect
from corpus.metrics.emit import finish_run, start_run

SODA = "/opt/soda-venv/bin/soda"
DATA_SOURCE = "corpus"  # the name the checks files refer to
CHECKS_PACKAGE = "corpus.checks"


def soda_configuration(dsn: str) -> str:
    """The connection file Soda needs, derived from our one connection string.

    Keeping a second copy of the credentials (in git or another variable) would be
    one more place for them to drift or leak, so this is generated per run.
    """
    parsed = urlparse(dsn)
    return (
        f"data_source {DATA_SOURCE}:\n"
        "  type: postgres\n"
        f"  host: {parsed.hostname}\n"
        f"  port: {parsed.port or 5432}\n"
        f"  username: {parsed.username}\n"
        f"  password: {parsed.password}\n"
        f"  database: {(parsed.path or '').lstrip('/')}\n"
        "  schema: public\n"
    )


def run_checks(gate: str, variables: dict[str, str]) -> tuple[bool, str]:
    """Run one gate's checks. Returns (passed, the report Soda printed)."""
    dsn = get_settings().metrics_dsn.get_secret_value()
    # Owner-only, and deleted in the finally block: the file holds the password.
    handle, config_path = tempfile.mkstemp(suffix=".yml", text=True)
    try:
        with os.fdopen(handle, "w") as config:
            config.write(soda_configuration(dsn))
        os.chmod(config_path, 0o600)

        with resources.as_file(resources.files(CHECKS_PACKAGE) / f"{gate}.yml") as checks:
            command = [SODA, "scan", "-d", DATA_SOURCE, "-c", config_path]
            for name, value in variables.items():
                command += ["-v", f"{name}={value}"]
            command.append(str(checks))
            result = subprocess.run(command, capture_output=True, text=True, check=False)
    finally:
        Path(config_path).unlink(missing_ok=True)

    report = (result.stdout + result.stderr).strip()
    return result.returncode == 0, report


def record_result(run_id: str, crawl_id: str, gate: str, passed: bool, detail: str) -> None:
    """Store the verdict. A retried task replaces its row instead of adding one."""
    with connect() as conn, conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO gate_results (run_id, crawl_id, gate_name, passed, failure_detail)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id, gate_name) DO UPDATE
                SET passed = EXCLUDED.passed,
                    failure_detail = EXCLUDED.failure_detail,
                    checked_at = now()
            """,
            (run_id, crawl_id, gate, passed, None if passed else detail[:4000]),
        )


def close_run(run_id: str, crawl_id: str, passed: bool) -> None:
    """Close the run: this gate is Sprint 1's last task. Feeds the success rate (S8).

    start_run first, so the gate still works when run on its own (by hand, or before
    Sprint 7 rearranges the DAG): it is an upsert and keeps any seed already recorded.
    """
    with connect() as conn:
        start_run(conn, run_id, crawl_id)
        finish_run(conn, run_id, "success" if passed else "failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="gate_a", help="name of the checks file")
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--expected-segments", type=int, required=True)
    parser.add_argument("--run-id", required=True, help="Airflow's dag_run.run_id")
    args = parser.parse_args()

    passed, report = run_checks(
        args.gate,
        {"crawl_id": args.crawl_id, "expected_segments": str(args.expected_segments)},
    )
    record_result(args.run_id, args.crawl_id, args.gate, passed, report)
    close_run(args.run_id, args.crawl_id, passed)

    print(report)
    print(f"{args.gate} for {args.crawl_id}: {'PASSED' if passed else 'FAILED'}")
    if not passed:
        sys.exit(1)  # stop the DAG


if __name__ == "__main__":
    main()
