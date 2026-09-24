"""Postgres access: the connection and the schema migrations.

Run the migrations with:  python -m corpus.db      (or: make migrate)
"""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources

import psycopg2
from psycopg2.extensions import connection as Connection

from corpus.config import get_settings

MIGRATIONS_PACKAGE = "corpus.migrations"

# Any integer; it only has to be the same in every process that migrates.
_MIGRATION_LOCK = 4_019_260_924

_SCHEMA_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   TEXT        PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


@contextmanager
def connect() -> Iterator[Connection]:
    """A connection to the control-plane database, always closed afterwards."""
    conn = psycopg2.connect(get_settings().metrics_dsn.get_secret_value())
    try:
        yield conn
    finally:
        conn.close()


def migration_files() -> list[tuple[str, str]]:
    """(filename, sql) for every migration, in filename order: 001_, 002_, ..."""
    folder = resources.files(MIGRATIONS_PACKAGE)
    names = sorted(f.name for f in folder.iterdir() if f.name.endswith(".sql"))
    return [(name, (folder / name).read_text(encoding="utf-8")) for name in names]


def apply_migrations() -> list[str]:
    """Apply the migrations this database has not seen yet. Returns their filenames.

    Safe to run twice, and safe to run from two processes at once.
    """
    applied: list[str] = []
    with connect() as conn:
        # `with conn` is a transaction: it commits on success, rolls back on error.
        with conn, conn.cursor() as cur:
            cur.execute(_SCHEMA_MIGRATIONS)

        for filename, sql in migration_files():
            with conn, conn.cursor() as cur:
                # Two Airflow tasks could migrate at the same time; the first one
                # here holds the lock until its transaction ends.
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
                cur.execute("SELECT 1 FROM schema_migrations WHERE filename = %s", (filename,))
                if cur.fetchone() is not None:
                    continue  # already applied
                cur.execute(sql)
                cur.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (filename,))
                applied.append(filename)
    return applied


if __name__ == "__main__":
    done = apply_migrations()
    print("applied: " + ", ".join(done) if done else "nothing to do, schema is up to date")
