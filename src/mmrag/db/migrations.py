from __future__ import annotations

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
            # running, so we cannot wrap it in our own BEGIN/COMMIT. The
            # connection runs in autocommit mode (isolation_level=None), so
            # the executescript and the insert each commit independently.
            conn.executescript(sql)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(name) VALUES (?)",
                (name,),
            )
            ran.append(name)
    return ran
