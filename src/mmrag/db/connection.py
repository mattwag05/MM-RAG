from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from mmrag.config import get_settings
from mmrag.logging import get_logger

log = get_logger("db.connection")

# Not lock-guarded: worst case is two warnings emitted at startup.
# The worker currently runs single-async-loop; acceptable for now.
_VEC_LOAD_WARNED = False
_VEC_EXT_DISABLED_WARNED = False


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension on the given connection when available.

    Degrades silently (one warning per process) in two scenarios:
    - the ``m3-visual`` extra isn't installed (``ImportError`` on ``sqlite_vec``)
    - the running Python was built without loadable-extension support
      (``AttributeError`` on ``conn.enable_load_extension``)

    Either way, core-only installs and stripped-down platforms can still run
    MCP tools in FTS-only mode without taking down every caller of connect().
    """
    global _VEC_LOAD_WARNED, _VEC_EXT_DISABLED_WARNED
    try:
        import sqlite_vec
    except ImportError:
        if not _VEC_LOAD_WARNED:
            log.warning("sqlite_vec.unavailable", hint="install with: make sync-m3")
            _VEC_LOAD_WARNED = True
        return
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        if not _VEC_EXT_DISABLED_WARNED:
            log.warning(
                "sqlite_vec.load_extension_disabled",
                hint="Python built without loadable-extension support",
            )
            _VEC_EXT_DISABLED_WARNED = True
        return
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
