from __future__ import annotations

import sqlite3
from contextlib import suppress
from importlib import resources

from mmrag.db.connection import connect
from mmrag.logging import get_logger

log = get_logger("migrations")


def _migration_files() -> list[tuple[str, str]]:
    """Return [(name, sql), ...] sorted by filename."""
    pkg = resources.files("mmrag.db").joinpath("sql")
    out: list[tuple[str, str]] = []
    for entry in sorted(pkg.iterdir(), key=lambda p: p.name):
        name = entry.name
        if not name.endswith(".sql"):
            continue
        out.append((name, entry.read_text(encoding="utf-8")))
    return out


def _applied(conn) -> set[str]:
    try:
        cur = conn.execute("SELECT name FROM schema_migrations")
    except Exception:
        return set()
    return {row["name"] for row in cur.fetchall()}


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def apply_migrations() -> list[str]:
    """Apply any pending migrations. Returns the names that ran this call."""
    ran: list[str] = []
    with connect() as conn:
        # Bootstrap migration table outside any other migration so we can
        # query it before the very first run.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
            """
        )
        already = _applied(conn)
        for name, sql in _migration_files():
            if name in already:
                continue
            log.info("applying migration", name=name)
            # executescript() implicitly commits any active transaction before
            # running, so the BEGIN/COMMIT must live inside the script. This
            # keeps migration DDL and the schema_migrations row atomic.
            script = f"""
            BEGIN;
            {sql}
            INSERT OR IGNORE INTO schema_migrations(name)
            VALUES ({_sql_literal(name)});
            COMMIT;
            """
            try:
                conn.executescript(script)
            except sqlite3.Error:
                with suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                raise
            ran.append(name)
    return ran
