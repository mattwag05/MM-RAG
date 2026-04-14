from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator

from mmrag.config import get_settings
from mmrag.logging import get_logger

log = get_logger("db.connection")

_VEC_LOAD_WARNED = False


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension on the given connection when available.

    Degrades silently (one warning per process) if the m3-visual extra is
    not installed, so core-only installs can still run MCP tools in FTS mode.
    """
    global _VEC_LOAD_WARNED
    try:
        import sqlite_vec
    except ImportError:
        if not _VEC_LOAD_WARNED:
            log.warning("sqlite_vec.unavailable", hint="install with: make sync-m3")
            _VEC_LOAD_WARNED = True
        return
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _open(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(
        db_path,
        isolation_level=None,  # autocommit; we manage transactions explicitly
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    _load_sqlite_vec(conn)
    return conn


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    settings = get_settings()
    settings.ensure_dirs()
    conn = _open(str(settings.db_path))
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
