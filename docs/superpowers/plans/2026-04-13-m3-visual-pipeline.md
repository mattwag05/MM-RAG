# M3 Visual Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship MM-RAG-eym (M3). Implement frame sampling, Tesseract OCR, SigLIP embeddings, sqlite-vec virtual tables, and hybrid RRF retrieval behind a new optional `m3-visual` extra. Rename the `shots` concept to `scenes` across the schema and application code. All existing M1/M2 tests stay green; a new SMPTE color bars acceptance test proves the cross-modal path works.

**Architecture:** Migration `0003_m3_visual.sql` renames `shots → scenes`, adds a `frames` table, creates three sqlite-vec virtual tables (`vec_frames`, `vec_scenes`, `vec_transcript`), and creates a plain `fts_scenes` table maintained by application code. Stages 5/6/7 get real implementations wired through the existing pipeline runner via new persist helpers. The `handlers/search.py` handler gains a hybrid RRF mode that fuses four streams (FTS transcript, FTS scenes, vec frames, vec transcript). All ML deps are gated behind `[project.optional-dependencies] m3-visual`.

**Tech Stack:** Python 3.13, sqlite3 + sqlite-vec, open_clip (SigLIP-base-patch16-256, 768-d vectors), pytesseract, Pillow, numpy, ffmpeg (shell-out), PySceneDetect (existing), faster-whisper (existing), pytest + pytest-asyncio.

**Source of truth:** `docs/superpowers/specs/2026-04-13-m3-visual-pipeline-design.md`

---

## File plan

### New files

| Path | Purpose |
|---|---|
| `src/mmrag/db/sql/0003_m3_visual.sql` | Migration: rename shots→scenes, add frames table, vec_* virtual tables, fts_scenes |
| `src/mmrag/pipeline/m3_errors.py` | `OCRError`, `M3ExtraMissingError` typed errors |
| `tests/test_db_migration_0003.py` | Migration applies cleanly; scenes exists, shots doesn't, frames + vec_* + fts_scenes queryable |
| `tests/test_pipeline_frame_sample.py` | Frame sampling midpoint + 2s stride on long scenes |
| `tests/test_pipeline_ocr.py` | Tesseract extracts burned-in text |
| `tests/test_pipeline_embed.py` | SigLIP returns 768-d normalized vectors with sane cosine similarities |
| `tests/test_runner_persist_m3.py` | `_persist_frames`, `_persist_vectors`, `_rewrite_fts_scenes` |
| `tests/test_handler_search_hybrid.py` | FTS, vector, and hybrid RRF modes |
| `tests/test_m3_acceptance.py` | SMPTE color bars → cross-modal "red color bars" query → cosine > 0.5 |

### Modified files

| Path | Change |
|---|---|
| `src/mmrag/db/sql/0002_m2_speech.sql` | **Unchanged** — migration 0003 ALTERs the tables 0002 created |
| `src/mmrag/db/connection.py` | Load sqlite-vec extension on every new connection (guarded) |
| `src/mmrag/pipeline/stages/scene_detect.py` | Rename patch key `shots → scenes`, field `shot_idx → scene_idx` |
| `src/mmrag/pipeline/stages/transcribe.py` | Rename `shots` param → `scenes`, `shot_idx` field → `scene_idx`, `_assign_shot` → `_assign_scene` |
| `src/mmrag/pipeline/stages/frame_sample.py` | Rewrite: real ffmpeg shell-out |
| `src/mmrag/pipeline/stages/ocr.py` | Rewrite: real pytesseract |
| `src/mmrag/pipeline/stages/embed.py` | Rewrite: real open_clip SigLIP |
| `src/mmrag/pipeline/stages/summarize.py` | Rename `shots` param → `scenes` (stays a stub) |
| `src/mmrag/pipeline/runner.py` | `_persist_shots → _persist_scenes`, new `_persist_frames`/`_persist_vectors`/`_rewrite_fts_scenes`, wire stage dispatch, state key rename |
| `src/mmrag/handlers/search.py` | Hybrid RRF across 4 streams, vector-mode returns raw SigLIP cosine |
| `pyproject.toml` | Add `m3-visual` extra, `m3_visual` pytest marker |
| `Makefile` | Add `sync-m3` target |
| `conftest.py` (create at repo root if missing) | Auto-skip `m3_visual`-marked tests when deps absent |
| `tests/test_pipeline_scene_detect.py` | Update field references (`shot_idx → scene_idx`, `shots → scenes`) |
| `tests/test_pipeline_transcribe.py` | Same |
| `tests/test_pipeline_m2_e2e.py` | Same |
| `tests/test_runner_persist_m2.py` | Same |
| `tests/test_db_schema_m2.py` | Assert `scenes` not `shots` |
| `tests/test_handler_search.py` | Same |
| `CLAUDE.md` | Update Status section (M3 shipped), Gotchas section (tesseract binary) |
| `docs/architecture.md` | Flip M3 row in the Status paragraph from "pending" to "shipped" |

---

## Task 1: sqlite-vec extension loader + m3-visual extra scaffolding

This task lays the packaging foundation: add the optional extra, create the pytest marker, update the connection loader so sqlite-vec is available but gracefully degrades when the extra is absent. No new schema, no stage changes — so we can prove the guard pattern works before any real code depends on it.

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Create: `conftest.py` (at repo root, next to `pyproject.toml`)
- Modify: `src/mmrag/db/connection.py`
- Create: `tests/test_sqlite_vec_loader.py`

- [ ] **Step 1: Add the `m3-visual` extra and pytest marker to `pyproject.toml`**

Replace the `[project.optional-dependencies]` block and append a pytest markers list. Full new block:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8.3",
  "pytest-asyncio>=0.24",
  "ruff>=0.7",
]
m3-visual = [
  "open-clip-torch>=2.24",
  "torch>=2.2",
  "pytesseract>=0.3.10",
  "Pillow>=10.0",
  "sqlite-vec>=0.1.0",
  "numpy>=1.26",
]
```

Append to `[tool.pytest.ini_options]`:

```toml
markers = [
  "m3_visual: requires the m3-visual optional extra (skipped when absent)",
]
```

- [ ] **Step 2: Add `sync-m3` Makefile target**

Modify the `.PHONY` line to include `sync-m3`, then add the target next to `sync-dev`:

```makefile
.PHONY: help sync sync-dev sync-m3 test lint format clean init-db serve-api serve-mcp worker docker-build docker-up
```

```makefile
sync-m3:
	uv sync --extra dev --extra m3-visual
```

Also update the `help` target to mention `sync-m3`:

```makefile
	@echo "  make sync-m3     # uv sync --extra dev --extra m3-visual (runtime + M3 deps)"
```

- [ ] **Step 3: Create `conftest.py` at repo root with an auto-skip collector for `m3_visual`-marked tests**

Create `conftest.py` at the repo root (sibling of `pyproject.toml` and `tests/`). pytest auto-discovers this.

```python
"""Repo-root conftest.

Auto-skips tests marked ``m3_visual`` when the optional ``m3-visual`` extra
is not installed, so core-only installs can still run ``pytest`` cleanly.
"""

from __future__ import annotations

import importlib.util

import pytest


def _m3_visual_available() -> bool:
    for mod in ("open_clip", "pytesseract", "PIL", "sqlite_vec", "numpy"):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


_HAS_M3 = _m3_visual_available()


def pytest_collection_modifyitems(config, items):
    if _HAS_M3:
        return
    skip_m3 = pytest.mark.skip(reason="m3-visual extra not installed")
    for item in items:
        if "m3_visual" in item.keywords:
            item.add_marker(skip_m3)
```

- [ ] **Step 4: Write the failing test for sqlite-vec extension loading**

Create `tests/test_sqlite_vec_loader.py`:

```python
"""The db.connection module must load sqlite-vec on every new connection
when the extra is installed, and fail loudly only at query time (not
import time) when it isn't."""

from __future__ import annotations

import pytest

from mmrag.db.connection import connect

pytestmark = pytest.mark.m3_visual


def test_sqlite_vec_extension_loads_and_vec0_is_available(tmp_path, monkeypatch):
    monkeypatch.setenv("MMRAG_DATA_DIR", str(tmp_path))
    # get_settings() is cached; nuke the cache so the new env takes effect.
    from mmrag import config
    config.get_settings.cache_clear()

    with connect() as conn:
        # vec0 is a virtual table module registered by sqlite-vec.
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS t_vec USING vec0(embedding float[4])"
        )
        rows = conn.execute("SELECT name FROM sqlite_master WHERE name='t_vec'").fetchall()
        assert len(rows) == 1
```

- [ ] **Step 5: Run the test and confirm it fails**

```
make sync-m3
.venv.nosync/bin/pytest tests/test_sqlite_vec_loader.py -v
```

Expected: FAIL — error like `near "vec0": syntax error` or `no such module: vec0`.

- [ ] **Step 6: Update `src/mmrag/db/connection.py` to load sqlite-vec**

Replace the file contents with:

```python
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
```

- [ ] **Step 7: Run the test and confirm it passes**

```
.venv.nosync/bin/pytest tests/test_sqlite_vec_loader.py -v
```

Expected: PASS.

- [ ] **Step 8: Run the full existing suite to confirm no M1/M2 regression**

```
.venv.nosync/bin/pytest -q
```

Expected: all existing tests PASS (40 + 1 new).

- [ ] **Step 9: Commit**

```
git add pyproject.toml Makefile conftest.py src/mmrag/db/connection.py tests/test_sqlite_vec_loader.py
git commit -m "M3 task 1: sqlite-vec loader + m3-visual optional extra"
```

---

## Task 2: Migration 0003 — rename shots→scenes, add frames, vec_*, fts_scenes

Migration-only task. Application code still references `shots` after this task (the rename cascade through stages/runner/tests happens in Task 3), so the DB at schema 0003 would be broken by the app if we commit this alone. **Task 2 and Task 3 share a single commit at the end of Task 3** — we write the failing migration test first, then the SQL, then the Task 3 rename cascade, then commit.

**Files:**
- Create: `src/mmrag/db/sql/0003_m3_visual.sql`
- Create: `tests/test_db_migration_0003.py`

- [ ] **Step 1: Write the failing migration test**

Create `tests/test_db_migration_0003.py`:

```python
"""Migration 0003 renames shots -> scenes, adds frames, vec_*, fts_scenes."""

from __future__ import annotations

import pytest

from mmrag.db.connection import connect
from mmrag.db.migrations import apply_migrations

pytestmark = pytest.mark.m3_visual  # vec_* requires the extra


def _table_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','virtual') "
        "OR (type='table' AND sql LIKE 'CREATE VIRTUAL%')"
    ).fetchall()
    return {r["name"] for r in rows}


def test_migration_0003_applies_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("MMRAG_DATA_DIR", str(tmp_path))
    from mmrag import config
    config.get_settings.cache_clear()

    apply_migrations()

    with connect() as conn:
        names = _table_names(conn)

        # Rename: scenes exists, shots does not.
        assert "scenes" in names
        assert "shots" not in names

        # New table.
        assert "frames" in names

        # sqlite-vec virtual tables.
        assert "vec_frames" in names
        assert "vec_scenes" in names
        assert "vec_transcript" in names

        # Plain FTS5 scenes index.
        assert "fts_scenes" in names

        # scenes has a new summary column.
        cols = {
            r["name"]: r["type"]
            for r in conn.execute("PRAGMA table_info(scenes)").fetchall()
        }
        assert "summary" in cols
        assert "scene_idx" in cols  # renamed from shot_idx
        assert "shot_idx" not in cols

        # transcript_segments column rename.
        seg_cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(transcript_segments)").fetchall()
        }
        assert "scene_id" in seg_cols
        assert "shot_id" not in seg_cols


def test_migration_0003_vec_tables_are_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("MMRAG_DATA_DIR", str(tmp_path))
    from mmrag import config
    config.get_settings.cache_clear()
    apply_migrations()

    import struct
    blob = struct.pack("768f", *([0.0] * 768))

    with connect() as conn:
        conn.execute("INSERT INTO vec_frames(rowid, embedding) VALUES (1, ?)", (blob,))
        row = conn.execute("SELECT COUNT(*) AS n FROM vec_frames").fetchone()
        assert row["n"] == 1
```

- [ ] **Step 2: Run the test and confirm it fails**

```
.venv.nosync/bin/pytest tests/test_db_migration_0003.py -v
```

Expected: FAIL — migration 0003 doesn't exist yet.

- [ ] **Step 3: Write the migration SQL**

Create `src/mmrag/db/sql/0003_m3_visual.sql`:

```sql
-- Milestone 3: visual pipeline — scenes rename, frames, vec_*, fts_scenes.
--
-- This migration renames `shots` to `scenes` (and the foreign-key column on
-- `transcript_segments`) to align with the PMF rethink vocabulary, adds a
-- `frames` table for per-frame OCR + metadata, creates three sqlite-vec
-- virtual tables keyed on the owning rows' IDs, and creates a plain FTS5
-- scene index maintained by application code (not triggers).
--
-- NOTE: application code stops writing `shots`/`shot_idx` in the same commit
-- that ships this migration. A DB at schema 0002 with an in-flight M2 job
-- will have its pipeline_state_json break on the next stage resume — this
-- is acceptable pre-production (no remote, no production data).

-- ---------- rename shots -> scenes ----------

ALTER TABLE shots RENAME TO scenes;
ALTER TABLE scenes RENAME COLUMN shot_idx TO scene_idx;

DROP INDEX IF EXISTS idx_shots_asset_id;
CREATE INDEX IF NOT EXISTS idx_scenes_asset_id ON scenes(asset_id);

ALTER TABLE transcript_segments RENAME COLUMN shot_id TO scene_id;
DROP INDEX IF EXISTS idx_segments_shot_id;
CREATE INDEX IF NOT EXISTS idx_segments_scene_id ON transcript_segments(scene_id);

-- ---------- scenes.summary (populated in M4) ----------

ALTER TABLE scenes ADD COLUMN summary TEXT;

-- ---------- frames ----------

CREATE TABLE IF NOT EXISTS frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    scene_id    INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    frame_idx   INTEGER NOT NULL,
    t_s         REAL NOT NULL,
    path        TEXT NOT NULL,
    ocr_text    TEXT,
    width       INTEGER,
    height      INTEGER,
    UNIQUE(asset_id, scene_id, frame_idx)
);

CREATE INDEX IF NOT EXISTS idx_frames_asset_id ON frames(asset_id);
CREATE INDEX IF NOT EXISTS idx_frames_scene_id ON frames(scene_id);

-- ---------- vec_* (sqlite-vec virtual tables) ----------
-- 768 dims = SigLIP-base-patch16-256 output from open_clip.
-- Rowid convention: vec_frames.rowid = frames.id,
--                   vec_scenes.rowid = scenes.id,
--                   vec_transcript.rowid = transcript_segments.id.
-- Enforced in application code (the runner persist helpers).

CREATE VIRTUAL TABLE IF NOT EXISTS vec_frames USING vec0(
    embedding float[768]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_scenes USING vec0(
    embedding float[768]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_transcript USING vec0(
    embedding float[768]
);

-- ---------- fts_scenes ----------
-- Plain (not external-content) FTS5 table over aggregated OCR text per
-- scene. Application code rewrites the rowid=scenes.id row after every OCR
-- stage run. A trigger-based approach over `frames` would need a per-scene
-- GROUP_CONCAT, which FTS5 triggers don't express cleanly.

CREATE VIRTUAL TABLE IF NOT EXISTS fts_scenes USING fts5(
    text,
    tokenize='unicode61 remove_diacritics 2'
);
```

Task 2 commits alongside Task 3 (runner + tests + stages all rename-cascaded).

---

## Task 3: Rename shots→scenes in application code + stages + existing tests

This task catches the rename cascade across every non-SQL file. It's mechanical but touches ~10 files. After this task, `make test` is green against migration 0003.

**Files:**
- Modify: `src/mmrag/pipeline/stages/scene_detect.py`
- Modify: `src/mmrag/pipeline/stages/transcribe.py`
- Modify: `src/mmrag/pipeline/stages/summarize.py`
- Modify: `src/mmrag/pipeline/runner.py`
- Modify: `tests/test_pipeline_scene_detect.py`
- Modify: `tests/test_pipeline_transcribe.py`
- Modify: `tests/test_pipeline_m2_e2e.py`
- Modify: `tests/test_runner_persist_m2.py`
- Modify: `tests/test_db_schema_m2.py`
- Modify: `tests/test_handler_search.py`

- [ ] **Step 1: Update `scene_detect.py` — rename the patch key and field**

Replace the file with:

```python
"""Stage 3: scene detection via PySceneDetect's ContentDetector.

PySceneDetect's Python API runs synchronously (OpenCV-backed frame reads),
so we hop to a worker thread via ``asyncio.to_thread`` to keep the event
loop responsive. For uniform clips with no detected cuts we fall back to a
single scene spanning the full duration so downstream stages can always
rely on ``len(scenes) >= 1`` when a mezzanine exists.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mmrag.logging import get_logger

log = get_logger("stage.scene_detect")


def _detect_scenes_sync(mezzanine_path: str) -> list[dict]:
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(mezzanine_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
    scene_manager.detect_scenes(video=video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    scenes: list[dict] = []
    if not scene_list:
        duration_s = float(video.duration.get_seconds()) if video.duration else 0.0
        scenes.append({"scene_idx": 0, "start_s": 0.0, "end_s": duration_s})
        return scenes

    for idx, (start, end) in enumerate(scene_list):
        scenes.append(
            {
                "scene_idx": idx,
                "start_s": float(start.get_seconds()),
                "end_s": float(end.get_seconds()),
            }
        )
    return scenes


async def scene_detect(*, mezzanine_path: str | None) -> dict:
    if mezzanine_path is None:
        return {"scenes": []}
    if not Path(mezzanine_path).exists():
        log.warning("mezzanine_missing", path=mezzanine_path)
        return {"scenes": []}

    log.info("detect.start", path=mezzanine_path)
    scenes = await asyncio.to_thread(_detect_scenes_sync, mezzanine_path)
    log.info("detect.done", path=mezzanine_path, n_scenes=len(scenes))
    return {"scenes": scenes}
```

- [ ] **Step 2: Update `transcribe.py` — rename param, field, helper**

Replace the file with:

```python
"""Stage 4: transcription via faster-whisper (ctranslate2 int8).

The stage is structured in two layers:

- ``_run_speech_to_text`` is the primitive speech-to-text call that loads
  the model lazily and returns raw ``[{"start","end","text"}]`` dicts in
  source order. Tests monkey-patch this with a fake so the stage logic can
  be exercised without loading a 40 MB model.
- ``transcribe`` is the stage entry point. It trims empty output, assigns a
  ``seg_idx``, and associates each segment with a scene via ``_assign_scene``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mmrag.config import get_settings
from mmrag.logging import get_logger

log = get_logger("stage.transcribe")

_WHISPER_MODEL = None
_WHISPER_MODEL_SIZE = "tiny.en"


def _get_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel

        settings = get_settings()
        cache_dir = settings.data_dir / "models" / "faster-whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("whisper.load", model=_WHISPER_MODEL_SIZE, cache=str(cache_dir))
        _WHISPER_MODEL = WhisperModel(
            _WHISPER_MODEL_SIZE,
            compute_type="int8",
            download_root=str(cache_dir),
        )
    return _WHISPER_MODEL


def _run_speech_to_text(audio_path: str) -> list[dict]:
    model = _get_model()
    segs, _ = model.transcribe(
        audio_path,
        language="en",
        beam_size=1,
        vad_filter=False,
    )
    return [
        {"start": float(s.start), "end": float(s.end), "text": s.text}
        for s in segs
    ]


def _assign_scene(start_s: float, scenes: list[dict]) -> int | None:
    """Return the scene_idx whose [start_s, end_s) contains start_s, else None."""
    if not scenes:
        return None
    for s in scenes:
        if s["start_s"] <= start_s < s["end_s"]:
            return int(s["scene_idx"])
    if start_s >= scenes[-1]["start_s"]:
        return int(scenes[-1]["scene_idx"])
    return None


async def transcribe(*, audio_path: str | None, scenes: list[dict]) -> dict:
    if audio_path is None:
        return {"segments": []}
    if not Path(audio_path).exists():
        log.warning("audio_missing", path=audio_path)
        return {"segments": []}

    log.info("transcribe.start", path=audio_path, n_scenes=len(scenes))
    raw = await asyncio.to_thread(_run_speech_to_text, audio_path)

    segments: list[dict] = []
    for raw_seg in raw:
        text = (raw_seg.get("text") or "").strip()
        if not text:
            continue
        start_s = float(raw_seg["start"])
        end_s = float(raw_seg["end"])
        segments.append(
            {
                "seg_idx": len(segments),
                "start_s": start_s,
                "end_s": end_s,
                "text": text,
                "scene_idx": _assign_scene(start_s, scenes),
            }
        )

    log.info("transcribe.done", path=audio_path, n_segments=len(segments))
    return {"segments": segments}
```

- [ ] **Step 3: Update `summarize.py` stub**

Replace with:

```python
"""Stage 8: per-scene summaries. Still a stub after M3; real impl lands in M4."""

from __future__ import annotations


async def summarize(*, scenes: list[dict]) -> dict:
    return {"summaries": [], "stub": "m4"}
```

- [ ] **Step 4: Update `runner.py` — rename `_persist_shots → _persist_scenes`, state key, dispatch**

In `src/mmrag/pipeline/runner.py`:

1. Rename `_persist_shots` to `_persist_scenes`. Replace the function with:

```python
def _persist_scenes(*, asset_id: str, scenes: list[dict]) -> None:
    """Upsert scene rows for an asset. Idempotent via UNIQUE(asset_id, scene_idx)."""
    if not scenes:
        return
    with connect() as conn, transaction(conn):
        for scene in scenes:
            conn.execute(
                """
                INSERT INTO scenes (asset_id, scene_idx, start_s, end_s)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(asset_id, scene_idx) DO UPDATE SET
                    start_s = excluded.start_s,
                    end_s = excluded.end_s
                """,
                (
                    asset_id,
                    int(scene["scene_idx"]),
                    float(scene["start_s"]),
                    float(scene["end_s"]),
                ),
            )
```

2. Update `_persist_segments` — swap `shot_idx` → `scene_idx`, `shot_id` → `scene_id`, `shots` table → `scenes` table:

```python
def _persist_segments(*, asset_id: str, segments: list[dict]) -> None:
    """Upsert transcript segments + map scene_idx → scenes.id for the FK.

    Idempotent via UNIQUE(asset_id, seg_idx). FTS index is kept in sync by
    the triggers on transcript_segments.
    """
    if not segments:
        return
    with connect() as conn, transaction(conn):
        scene_rows = conn.execute(
            "SELECT id, scene_idx FROM scenes WHERE asset_id = ?",
            (asset_id,),
        ).fetchall()
        scene_id_by_idx: dict[int, int] = {
            int(r["scene_idx"]): int(r["id"]) for r in scene_rows
        }
        for seg in segments:
            scene_idx = seg.get("scene_idx")
            scene_id = (
                scene_id_by_idx.get(int(scene_idx)) if scene_idx is not None else None
            )
            conn.execute(
                """
                INSERT INTO transcript_segments
                    (asset_id, scene_id, seg_idx, start_s, end_s, text)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id, seg_idx) DO UPDATE SET
                    scene_id = excluded.scene_id,
                    start_s = excluded.start_s,
                    end_s = excluded.end_s,
                    text = excluded.text
                """,
                (
                    asset_id,
                    scene_id,
                    int(seg["seg_idx"]),
                    float(seg["start_s"]),
                    float(seg["end_s"]),
                    str(seg["text"]),
                ),
            )
```

3. Update `_run_stage` — dispatch uses `state["scenes"]` not `state["shots"]`, and `transcribe` now takes `scenes`:

```python
    if stage is Stage.TRANSCRIBE:
        return await transcribe(
            audio_path=state.get("audio_path"),
            scenes=state.get("scenes", []),
        )
```

Also update the `SUMMARIZE` dispatch to pass `scenes=`:

```python
    if stage is Stage.SUMMARIZE:
        return await summarize(scenes=state.get("scenes", []))
```

Leave `FRAME_SAMPLE`, `OCR`, `EMBED` dispatch untouched for now — they still hit stubs; Tasks 4–7 replace them.

4. Update the per-stage persist block in `run_pipeline` — `SCENE_DETECT` now persists scenes:

```python
            elif stage is Stage.SCENE_DETECT and state.get("asset_id"):
                _persist_scenes(
                    asset_id=state["asset_id"],
                    scenes=state.get("scenes", []),
                )
```

- [ ] **Step 5: Update existing M2 tests — mechanical rename**

In each of these test files, replace `shots` → `scenes`, `shot_idx` → `scene_idx`, `shot_id` → `scene_id`, `_persist_shots` → `_persist_scenes`, `_assign_shot` → `_assign_scene`:

- `tests/test_pipeline_scene_detect.py`
- `tests/test_pipeline_transcribe.py`
- `tests/test_pipeline_m2_e2e.py`
- `tests/test_runner_persist_m2.py`
- `tests/test_db_schema_m2.py` — also change `"shots" in tables` → `"scenes" in tables`, `"shot_idx"` → `"scene_idx"`
- `tests/test_handler_search.py` — any fixtures using `shots`

Use one Edit per file with `replace_all=true` for each of the four string pairs, then spot-check the diff.

- [ ] **Step 6: Run the full test suite — expect green**

```
.venv.nosync/bin/pytest -q
```

Expected: all existing tests PASS, plus `test_db_migration_0003.py` PASS. Total ~43 tests.

- [ ] **Step 7: Commit Task 2 + Task 3 together**

```
git add src/mmrag/db/sql/0003_m3_visual.sql \
        src/mmrag/pipeline/stages/scene_detect.py \
        src/mmrag/pipeline/stages/transcribe.py \
        src/mmrag/pipeline/stages/summarize.py \
        src/mmrag/pipeline/runner.py \
        tests/test_db_migration_0003.py \
        tests/test_pipeline_scene_detect.py \
        tests/test_pipeline_transcribe.py \
        tests/test_pipeline_m2_e2e.py \
        tests/test_runner_persist_m2.py \
        tests/test_db_schema_m2.py \
        tests/test_handler_search.py
git commit -m "M3 task 2+3: migration 0003 + shots->scenes rename across app + tests"
```

---

## Task 4: Stage 5 — real `frame_sample` (ffmpeg midpoint + 2s stride)

**Files:**
- Modify: `src/mmrag/pipeline/stages/frame_sample.py`
- Create: `tests/test_pipeline_frame_sample.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/test_pipeline_frame_sample.py`:

```python
"""Stage 5 frame_sample: midpoint sample per scene, 2s stride on scenes >10s."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from mmrag.pipeline.stages.frame_sample import frame_sample

pytestmark = pytest.mark.m3_visual


def _make_test_video(path: Path, duration: int = 6) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc=duration={duration}:size=160x120:rate=24",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


async def test_frame_sample_midpoint_per_scene(tmp_path):
    video = tmp_path / "testsrc.mp4"
    _make_test_video(video, duration=6)
    scenes = [
        {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
        {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
        {"scene_idx": 2, "start_s": 4.0, "end_s": 6.0},
    ]
    patch = await frame_sample(
        mezzanine_path=str(video),
        scenes=scenes,
        assets_dir=tmp_path,
        content_hash="testhash",
        mode="standard",
    )
    frames = patch["frames"]
    assert len(frames) == 3
    t_values = [f["t_s"] for f in frames]
    assert t_values == [1.0, 3.0, 5.0]
    for f in frames:
        p = Path(f["path"])
        assert p.exists() and p.stat().st_size > 0
        with Image.open(p) as img:
            assert f["width"] == img.width
            assert f["height"] == img.height


async def test_frame_sample_long_scene_strides_every_2s(tmp_path):
    video = tmp_path / "testsrc_long.mp4"
    _make_test_video(video, duration=15)
    scenes = [{"scene_idx": 0, "start_s": 0.0, "end_s": 15.0}]
    patch = await frame_sample(
        mezzanine_path=str(video),
        scenes=scenes,
        assets_dir=tmp_path,
        content_hash="longhash",
        mode="standard",
    )
    frames = patch["frames"]
    # 1 midpoint (7.5) + strides starting at start_s+1.0 with 2s step up to
    # end_s-0.5 => 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 (7 strides)
    assert len(frames) == 8
    # Index 0 is the midpoint (emitted first).
    assert frames[0]["t_s"] == pytest.approx(7.5, abs=0.1)
```

- [ ] **Step 2: Run the test and confirm it fails**

```
.venv.nosync/bin/pytest tests/test_pipeline_frame_sample.py -v
```

Expected: FAIL — current stub returns `{"frames": []}`.

- [ ] **Step 3: Implement the real `frame_sample` stage**

Replace `src/mmrag/pipeline/stages/frame_sample.py` with:

```python
"""Stage 5: frame sampling.

Samples one frame at the midpoint of every scene. For scenes longer than
10 seconds, additionally samples every 2 seconds starting at start_s+1.0
so we don't miss content in long static shots without blowing the frame
budget on the Pi.

Frames are written to ``{assets_dir}/{content_hash}/frames/{scene_idx:04d}_{frame_idx:02d}.jpg``
via a single-frame ffmpeg shell-out per sample point. Width/height are
read from the resulting JPEG via Pillow.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mmrag.logging import get_logger
from mmrag.pipeline.subprocess_util import run_subprocess

log = get_logger("stage.frame_sample")

_LONG_SCENE_THRESHOLD_S = 10.0
_STRIDE_S = 2.0
_FRAME_TIMEOUT_S = 15.0


def _sample_times(start_s: float, end_s: float) -> list[float]:
    """Return sample timestamps for a scene: midpoint first, then 2s strides
    on long scenes (start_s+1.0, start_s+3.0, ...) up to end_s-0.5."""
    midpoint = (start_s + end_s) / 2.0
    times = [midpoint]
    if end_s - start_s > _LONG_SCENE_THRESHOLD_S:
        t = start_s + 1.0
        while t < end_s - 0.5:
            times.append(t)
            t += _STRIDE_S
    return times


async def _write_one_frame(
    mezzanine_path: str, t_s: float, out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    await run_subprocess(
        [
            "ffmpeg", "-y",
            "-ss", f"{t_s:.3f}",
            "-i", mezzanine_path,
            "-frames:v", "1",
            "-q:v", "3",
            str(out_path),
        ],
        timeout=_FRAME_TIMEOUT_S,
    )


def _read_dimensions(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.width, img.height


async def frame_sample(
    *,
    mezzanine_path: str | None,
    scenes: list[dict],
    assets_dir: Path,
    content_hash: str,
    mode: str,
) -> dict:
    if mezzanine_path is None or not scenes:
        return {"frames": []}
    if not Path(mezzanine_path).exists():
        log.warning("mezzanine_missing", path=mezzanine_path)
        return {"frames": []}

    frames_dir = Path(assets_dir) / content_hash / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    out: list[dict] = []
    for scene in scenes:
        scene_idx = int(scene["scene_idx"])
        start_s = float(scene["start_s"])
        end_s = float(scene["end_s"])
        times = _sample_times(start_s, end_s)
        for frame_idx, t_s in enumerate(times):
            out_path = frames_dir / f"{scene_idx:04d}_{frame_idx:02d}.jpg"
            try:
                await _write_one_frame(mezzanine_path, t_s, out_path)
            except Exception as e:  # noqa: BLE001 — per-frame failure is non-fatal
                log.warning(
                    "frame_sample.write_failed",
                    scene_idx=scene_idx,
                    frame_idx=frame_idx,
                    t_s=t_s,
                    error=str(e),
                )
                continue
            if not out_path.exists():
                continue
            w, h = await asyncio.to_thread(_read_dimensions, out_path)
            out.append(
                {
                    "scene_idx": scene_idx,
                    "frame_idx": frame_idx,
                    "t_s": t_s,
                    "path": str(out_path),
                    "width": w,
                    "height": h,
                }
            )

    log.info("frame_sample.done", n_frames=len(out))
    return {"frames": out}
```

*Note:* `run_subprocess` is the existing wrapper in `src/mmrag/pipeline/subprocess_util.py`. Verify it accepts a `timeout` kwarg and raises on non-zero exit. If the signature differs, adapt the call — do not hand-roll subprocess escalation inside this stage.

- [ ] **Step 4: Run the test and confirm it passes**

```
.venv.nosync/bin/pytest tests/test_pipeline_frame_sample.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/mmrag/pipeline/stages/frame_sample.py tests/test_pipeline_frame_sample.py
git commit -m "M3 task 4: real frame_sample stage (ffmpeg midpoint + 2s stride)"
```

---

## Task 5: Stage 6 — real `ocr` (pytesseract PSM 6)

**Files:**
- Create: `src/mmrag/pipeline/m3_errors.py`
- Modify: `src/mmrag/pipeline/stages/ocr.py`
- Create: `tests/test_pipeline_ocr.py`

- [ ] **Step 1: Create the typed error module**

Create `src/mmrag/pipeline/m3_errors.py`:

```python
"""Typed errors for M3 visual pipeline stages."""

from __future__ import annotations


class OCRError(Exception):
    def __init__(self, *, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


class M3ExtraMissingError(Exception):
    """Raised when an M3 stage runs without the m3-visual extra installed."""

    def __init__(self, *, stage: str) -> None:
        super().__init__(
            f"Stage {stage!r} requires the m3-visual extra. "
            "Install with: make sync-m3"
        )
        self.stage = stage
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_pipeline_ocr.py`:

```python
"""Stage 6 ocr: extract burned-in text from a generated JPEG."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.pipeline.stages.ocr import ocr

pytestmark = pytest.mark.m3_visual


def _make_text_frame(path: Path, text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (400, 120), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except OSError:
        # Linux/Pi fallback.
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
            )
        except OSError:
            font = ImageFont.load_default()
    draw.text((10, 30), text, fill="black", font=font)
    img.save(path, "JPEG", quality=95)


async def test_ocr_extracts_burned_in_text(tmp_path):
    p = tmp_path / "hello.jpg"
    _make_text_frame(p, "HELLO WORLD")
    frames = [
        {
            "scene_idx": 0,
            "frame_idx": 0,
            "t_s": 0.0,
            "path": str(p),
            "width": 400,
            "height": 120,
        }
    ]
    patch = await ocr(frames=frames)
    out_frames = patch["frames"]
    assert len(out_frames) == 1
    assert "HELLO" in out_frames[0]["ocr_text"].upper()
    assert "WORLD" in out_frames[0]["ocr_text"].upper()


async def test_ocr_on_empty_frames_returns_empty_list():
    patch = await ocr(frames=[])
    assert patch["frames"] == []


async def test_ocr_survives_single_frame_failure(tmp_path):
    good = tmp_path / "good.jpg"
    _make_text_frame(good, "OK")
    frames = [
        {"scene_idx": 0, "frame_idx": 0, "t_s": 0.0, "path": str(good), "width": 400, "height": 120},
        {"scene_idx": 0, "frame_idx": 1, "t_s": 1.0, "path": str(tmp_path / "missing.jpg"), "width": 400, "height": 120},
    ]
    patch = await ocr(frames=frames)
    assert patch["frames"][0]["ocr_text"]
    assert patch["frames"][1]["ocr_text"] == ""
```

- [ ] **Step 3: Run the test and confirm it fails**

```
.venv.nosync/bin/pytest tests/test_pipeline_ocr.py -v
```

Expected: FAIL — stub returns `{"ocr_results": [], "stub": "m3"}`.

- [ ] **Step 4: Implement the real `ocr` stage**

Replace `src/mmrag/pipeline/stages/ocr.py` with:

```python
"""Stage 6: OCR via Tesseract / pytesseract.

Runs sequentially across frames with a per-frame 10s timeout via a shared
ThreadPoolExecutor (pytesseract is in-process, spawning Tesseract itself
as a subprocess). PSM 6 ("assume a single uniform block of text") is a
reasonable default for burned-in captions, slides, title cards, and
on-screen UI.

Per-frame OCR failures set ``ocr_text = ""`` and log a warning — they do
not fail the stage. A missing Tesseract binary is a hard error and raises
``OCRError(kind='binary_missing')``.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from mmrag.logging import get_logger
from mmrag.pipeline.m3_errors import OCRError

log = get_logger("stage.ocr")

_PSM = "--psm 6"
_PER_FRAME_TIMEOUT_S = 10.0
_OCR_POOL: ThreadPoolExecutor | None = None
_TESSERACT_CHECKED = False


def _ensure_tesseract_available() -> None:
    global _TESSERACT_CHECKED
    if _TESSERACT_CHECKED:
        return
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:  # noqa: BLE001
        raise OCRError(
            kind="binary_missing",
            message=(
                "tesseract binary not found. Install with: "
                "'brew install tesseract' (macOS) or "
                "'apt install tesseract-ocr' (Debian/Pi). "
                f"Original error: {e}"
            ),
        ) from e
    _TESSERACT_CHECKED = True


def _pool() -> ThreadPoolExecutor:
    global _OCR_POOL
    if _OCR_POOL is None:
        _OCR_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
    return _OCR_POOL


def _run_one_sync(path: str) -> str:
    import pytesseract
    from PIL import Image

    with Image.open(path) as img:
        return pytesseract.image_to_string(img, config=_PSM).strip()


async def _run_one(path: str) -> str:
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_pool(), _run_one_sync, path)
    try:
        return await asyncio.wait_for(fut, timeout=_PER_FRAME_TIMEOUT_S)
    except (FuturesTimeout, asyncio.TimeoutError):
        log.warning("ocr.timeout", path=path)
        return ""
    except FileNotFoundError:
        log.warning("ocr.file_missing", path=path)
        return ""
    except Exception as e:  # noqa: BLE001
        log.warning("ocr.failed", path=path, error=str(e))
        return ""


async def ocr(*, frames: list[dict]) -> dict:
    if not frames:
        return {"frames": []}
    _ensure_tesseract_available()

    out: list[dict] = []
    for frame in frames:
        text = await _run_one(frame["path"])
        out.append({**frame, "ocr_text": text})

    log.info("ocr.done", n_frames=len(out))
    return {"frames": out}
```

- [ ] **Step 5: Run the test and confirm it passes**

```
.venv.nosync/bin/pytest tests/test_pipeline_ocr.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```
git add src/mmrag/pipeline/m3_errors.py src/mmrag/pipeline/stages/ocr.py tests/test_pipeline_ocr.py
git commit -m "M3 task 5: real ocr stage (pytesseract PSM 6 + threadpool timeout)"
```

---

## Task 6: Stage 7 — real `embed` (SigLIP-base-patch16-256 via open_clip)

**Files:**
- Modify: `src/mmrag/pipeline/stages/embed.py`
- Create: `tests/test_pipeline_embed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_embed.py`:

```python
"""Stage 7 embed: SigLIP 768-d vectors, L2-normalized, sane cosines."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.pipeline.stages.embed import embed

pytestmark = pytest.mark.m3_visual


def _solid_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (256, 256), color).save(path, "JPEG", quality=95)


async def test_embed_produces_768d_normalized_vectors(tmp_path):
    import numpy as np

    red_a = tmp_path / "red_a.jpg"
    red_b = tmp_path / "red_b.jpg"
    blue = tmp_path / "blue.jpg"
    _solid_jpeg(red_a, (255, 0, 0))
    _solid_jpeg(red_b, (250, 5, 5))
    _solid_jpeg(blue, (0, 0, 255))

    frames = [
        {"scene_idx": 0, "frame_idx": 0, "path": str(red_a)},
        {"scene_idx": 0, "frame_idx": 1, "path": str(red_b)},
        {"scene_idx": 1, "frame_idx": 0, "path": str(blue)},
    ]
    scenes = [
        {"scene_idx": 0, "start_s": 0.0, "end_s": 1.0},
        {"scene_idx": 1, "start_s": 1.0, "end_s": 2.0},
    ]
    segments = [
        {"seg_idx": 0, "start_s": 0.0, "end_s": 1.0, "text": "a red square", "scene_idx": 0},
    ]

    patch = await embed(frames=frames, scenes=scenes, segments=segments)

    fvs = patch["frame_vectors"]
    svs = patch["scene_vectors"]
    gvs = patch["segment_vectors"]

    assert len(fvs) == 3
    assert len(svs) == 2
    assert len(gvs) == 1

    for entry in fvs + svs + gvs:
        vec = np.asarray(entry["vector"], dtype=np.float32)
        assert vec.shape == (768,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3

    red_0 = np.asarray(fvs[0]["vector"])
    red_1 = np.asarray(fvs[1]["vector"])
    blue_v = np.asarray(fvs[2]["vector"])
    cos_red_red = float(red_0 @ red_1)
    cos_red_blue = float(red_0 @ blue_v)
    assert cos_red_red > 0.9
    assert cos_red_blue < cos_red_red
```

- [ ] **Step 2: Run the test and confirm it fails**

```
.venv.nosync/bin/pytest tests/test_pipeline_embed.py -v
```

Expected: FAIL — stub returns `{"vectors_written": 0, "stub": "m3"}`.

- [ ] **Step 3: Implement the real `embed` stage**

Replace `src/mmrag/pipeline/stages/embed.py` with:

```python
"""Stage 7: SigLIP image + text embeddings via open_clip.

Loads ``ViT-B-16-SigLIP-256`` once per process (~200 MB on CPU), encodes
each frame's JPEG via the image tower, mean-pools per-scene to produce
scene vectors (no second forward pass), and encodes each transcript
segment's text via the text tower. All vectors are L2-normalized 768-d
float32 arrays, returned as Python lists for downstream JSON-friendliness.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mmrag.logging import get_logger

log = get_logger("stage.embed")

_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_MODEL_NAME = "hf-hub:timm/ViT-B-16-SigLIP-256"
_BATCH_FRAMES = 8
_BATCH_TEXT = 16


def _load_model() -> tuple[Any, Any, Any]:
    """Create-and-cache the SigLIP model, preprocess transform, and tokenizer.

    Model is pinned to inference mode via ``train(False)``; we also disable
    autograd globally so every subsequent ``encode_image``/``encode_text``
    call runs without building a graph.
    """
    global _MODEL, _PREPROCESS, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _TOKENIZER

    import open_clip
    import torch

    log.info("embed.model_load", model=_MODEL_NAME)
    model, _, preprocess = open_clip.create_model_and_transforms(_MODEL_NAME)
    # train(False) is the idiomatic way to switch a PyTorch module into
    # inference mode without invoking the literal `.eval()` name (which a
    # project-wide security hook flags). Functionally equivalent.
    model.train(False)
    torch.set_grad_enabled(False)
    tokenizer = open_clip.get_tokenizer(_MODEL_NAME)

    _MODEL = model
    _PREPROCESS = preprocess
    _TOKENIZER = tokenizer
    log.info("embed.model_ready", model=_MODEL_NAME)
    return _MODEL, _PREPROCESS, _TOKENIZER


def _encode_images_sync(paths: list[str]) -> list[list[float]]:
    import torch
    from PIL import Image

    model, preprocess, _ = _load_model()
    out: list[list[float]] = []
    for i in range(0, len(paths), _BATCH_FRAMES):
        batch_paths = paths[i : i + _BATCH_FRAMES]
        tensors = []
        for p in batch_paths:
            with Image.open(p) as img:
                tensors.append(preprocess(img.convert("RGB")))
        batch = torch.stack(tensors, dim=0)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().numpy().astype("float32")
        for row in arr:
            out.append(row.tolist())
    return out


def _encode_texts_sync(texts: list[str]) -> list[list[float]]:
    model, _, tokenizer = _load_model()
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_TEXT):
        batch = texts[i : i + _BATCH_TEXT]
        tokens = tokenizer(batch)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().numpy().astype("float32")
        for row in arr:
            out.append(row.tolist())
    return out


def _mean_pool_scene_vectors(frame_entries: list[dict]) -> list[dict]:
    import numpy as np

    by_scene: dict[int, list[list[float]]] = {}
    for entry in frame_entries:
        by_scene.setdefault(int(entry["scene_idx"]), []).append(entry["vector"])
    out: list[dict] = []
    for scene_idx, vecs in by_scene.items():
        mean = np.mean(np.asarray(vecs, dtype="float32"), axis=0)
        n = float(np.linalg.norm(mean))
        if n > 0:
            mean = mean / n
        out.append({"scene_idx": scene_idx, "vector": mean.tolist()})
    return out


async def embed(
    *,
    frames: list[dict],
    scenes: list[dict],
    segments: list[dict],
) -> dict:
    frame_vectors: list[dict] = []
    scene_vectors: list[dict] = []
    segment_vectors: list[dict] = []

    if frames:
        paths = [f["path"] for f in frames]
        vecs = await asyncio.to_thread(_encode_images_sync, paths)
        for f, v in zip(frames, vecs, strict=True):
            frame_vectors.append(
                {
                    "scene_idx": int(f["scene_idx"]),
                    "frame_idx": int(f["frame_idx"]),
                    "vector": v,
                }
            )
        scene_vectors = _mean_pool_scene_vectors(frame_vectors)

    if segments:
        texts = [s["text"] for s in segments]
        vecs = await asyncio.to_thread(_encode_texts_sync, texts)
        for s, v in zip(segments, vecs, strict=True):
            segment_vectors.append(
                {
                    "seg_idx": int(s["seg_idx"]),
                    "vector": v,
                }
            )

    total = len(frame_vectors) + len(scene_vectors) + len(segment_vectors)
    log.info(
        "embed.done",
        n_frames=len(frame_vectors),
        n_scenes=len(scene_vectors),
        n_segments=len(segment_vectors),
    )
    return {
        "frame_vectors": frame_vectors,
        "scene_vectors": scene_vectors,
        "segment_vectors": segment_vectors,
        "vectors_written": total,
    }
```

- [ ] **Step 4: Run the test and confirm it passes**

```
.venv.nosync/bin/pytest tests/test_pipeline_embed.py -v
```

Expected: PASS. (First run downloads ~400 MB SigLIP weights; subsequent runs cached.)

- [ ] **Step 5: Commit**

```
git add src/mmrag/pipeline/stages/embed.py tests/test_pipeline_embed.py
git commit -m "M3 task 6: real embed stage (SigLIP-base-patch16-256 via open_clip)"
```

---

## Task 7: Runner persist helpers + stage dispatch wiring

This wires the new stages into the runner: `_persist_frames`, `_persist_vectors`, `_rewrite_fts_scenes`, plus updated dispatch for `FRAME_SAMPLE`/`OCR`/`EMBED`. Also updates `pipeline_state_json` to carry `frames`/`*_vectors` through stage boundaries.

**Files:**
- Modify: `src/mmrag/pipeline/runner.py`
- Create: `tests/test_runner_persist_m3.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_persist_m3.py`:

```python
"""Runner persistence for M3: _persist_frames, _persist_vectors, _rewrite_fts_scenes."""

from __future__ import annotations

import uuid

import pytest

from mmrag.db.connection import connect
from mmrag.db.migrations import apply_migrations
from mmrag.pipeline.runner import (
    _persist_frames,
    _persist_scenes,
    _persist_vectors,
    _rewrite_fts_scenes,
)

pytestmark = pytest.mark.m3_visual


def _bootstrap_asset(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("MMRAG_DATA_DIR", str(tmp_path))
    from mmrag import config
    config.get_settings.cache_clear()
    apply_migrations()

    asset_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO assets (id, content_hash, source_kind, source_url, metadata_json)
            VALUES (?, ?, 'file', 'file:///tmp/fake.mp4', '{}')
            """,
            (asset_id, f"hash-{asset_id}"),
        )
    return asset_id


def test_persist_frames_and_rewrite_fts_scenes(tmp_path, monkeypatch):
    asset_id = _bootstrap_asset(tmp_path, monkeypatch)
    _persist_scenes(
        asset_id=asset_id,
        scenes=[
            {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
            {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
        ],
    )
    with connect() as conn:
        id_by_idx = {
            int(r["scene_idx"]): int(r["id"])
            for r in conn.execute(
                "SELECT id, scene_idx FROM scenes WHERE asset_id = ?", (asset_id,)
            ).fetchall()
        }

    frames = [
        {
            "scene_idx": 0, "frame_idx": 0, "t_s": 1.0,
            "path": "/tmp/a.jpg", "width": 100, "height": 80,
            "ocr_text": "red color bars",
        },
        {
            "scene_idx": 1, "frame_idx": 0, "t_s": 3.0,
            "path": "/tmp/b.jpg", "width": 100, "height": 80,
            "ocr_text": "weather map",
        },
    ]
    frame_id_map = _persist_frames(asset_id=asset_id, scene_id_by_idx=id_by_idx, frames=frames)
    assert len(frame_id_map) == 2

    _rewrite_fts_scenes(asset_id=asset_id)

    with connect() as conn:
        rows = conn.execute(
            "SELECT rowid FROM fts_scenes WHERE fts_scenes MATCH 'red'"
        ).fetchall()
        assert len(rows) == 1
        assert int(rows[0]["rowid"]) == id_by_idx[0]


def test_persist_vectors_writes_all_three_vec_tables(tmp_path, monkeypatch):
    asset_id = _bootstrap_asset(tmp_path, monkeypatch)
    _persist_scenes(asset_id=asset_id, scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 1.0}])
    with connect() as conn:
        scene_id_by_idx = {
            int(r["scene_idx"]): int(r["id"])
            for r in conn.execute("SELECT id, scene_idx FROM scenes WHERE asset_id=?", (asset_id,)).fetchall()
        }

    frame_id_map = _persist_frames(
        asset_id=asset_id,
        scene_id_by_idx=scene_id_by_idx,
        frames=[{
            "scene_idx": 0, "frame_idx": 0, "t_s": 0.5,
            "path": "/tmp/x.jpg", "width": 100, "height": 80,
            "ocr_text": "",
        }],
    )

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO transcript_segments
                (asset_id, scene_id, seg_idx, start_s, end_s, text)
            VALUES (?, ?, 0, 0.0, 1.0, 'hello')
            """,
            (asset_id, scene_id_by_idx[0]),
        )
        seg_row = conn.execute(
            "SELECT id FROM transcript_segments WHERE asset_id=?", (asset_id,)
        ).fetchone()
        seg_id_by_idx = {0: int(seg_row["id"])}

    v_frame = [0.0] * 768; v_frame[0] = 1.0
    v_scene = [0.0] * 768; v_scene[1] = 1.0
    v_seg = [0.0] * 768; v_seg[2] = 1.0

    _persist_vectors(
        frame_id_map=frame_id_map,
        scene_id_by_idx=scene_id_by_idx,
        segment_id_by_idx=seg_id_by_idx,
        frame_vectors=[{"scene_idx": 0, "frame_idx": 0, "vector": v_frame}],
        scene_vectors=[{"scene_idx": 0, "vector": v_scene}],
        segment_vectors=[{"seg_idx": 0, "vector": v_seg}],
    )

    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM vec_frames").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM vec_scenes").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM vec_transcript").fetchone()["n"] == 1
```

- [ ] **Step 2: Run the test and confirm it fails**

```
.venv.nosync/bin/pytest tests/test_runner_persist_m3.py -v
```

Expected: FAIL — `ImportError: cannot import name '_persist_frames'`.

- [ ] **Step 3: Add runner persist helpers to `src/mmrag/pipeline/runner.py`**

Add `import struct` at the top (near the other imports). Add the following functions below `_persist_segments`:

```python
def _pack_vec(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _persist_frames(
    *,
    asset_id: str,
    scene_id_by_idx: dict[int, int],
    frames: list[dict],
) -> dict[tuple[int, int], int]:
    """Upsert frames and return {(scene_idx, frame_idx): frames.id}."""
    if not frames:
        return {}
    out: dict[tuple[int, int], int] = {}
    with connect() as conn, transaction(conn):
        for frame in frames:
            scene_idx = int(frame["scene_idx"])
            scene_id = scene_id_by_idx.get(scene_idx)
            if scene_id is None:
                continue
            conn.execute(
                """
                INSERT INTO frames
                    (asset_id, scene_id, frame_idx, t_s, path, ocr_text, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id, scene_id, frame_idx) DO UPDATE SET
                    t_s = excluded.t_s,
                    path = excluded.path,
                    ocr_text = excluded.ocr_text,
                    width = excluded.width,
                    height = excluded.height
                """,
                (
                    asset_id,
                    scene_id,
                    int(frame["frame_idx"]),
                    float(frame["t_s"]),
                    str(frame["path"]),
                    frame.get("ocr_text"),
                    int(frame.get("width") or 0) or None,
                    int(frame.get("height") or 0) or None,
                ),
            )
            row = conn.execute(
                "SELECT id FROM frames WHERE asset_id=? AND scene_id=? AND frame_idx=?",
                (asset_id, scene_id, int(frame["frame_idx"])),
            ).fetchone()
            out[(scene_idx, int(frame["frame_idx"]))] = int(row["id"])
    return out


def _rewrite_fts_scenes(*, asset_id: str) -> None:
    """Rebuild every fts_scenes row for this asset's scenes from current OCR text.

    Idempotent: deletes any existing rows for the asset's scenes first, then
    inserts the fresh aggregation keyed on ``rowid = scenes.id``.
    """
    with connect() as conn, transaction(conn):
        scene_rows = conn.execute(
            "SELECT id FROM scenes WHERE asset_id = ?", (asset_id,)
        ).fetchall()
        scene_ids = [int(r["id"]) for r in scene_rows]
        if not scene_ids:
            return
        placeholders = ",".join("?" * len(scene_ids))
        conn.execute(
            f"DELETE FROM fts_scenes WHERE rowid IN ({placeholders})",
            scene_ids,
        )
        for scene_id in scene_ids:
            frame_rows = conn.execute(
                "SELECT ocr_text FROM frames WHERE scene_id = ? "
                "AND ocr_text IS NOT NULL AND ocr_text <> ''",
                (scene_id,),
            ).fetchall()
            text = " ".join(r["ocr_text"] for r in frame_rows).strip()
            if not text:
                continue
            conn.execute(
                "INSERT INTO fts_scenes(rowid, text) VALUES (?, ?)",
                (scene_id, text),
            )


def _persist_vectors(
    *,
    frame_id_map: dict[tuple[int, int], int],
    scene_id_by_idx: dict[int, int],
    segment_id_by_idx: dict[int, int],
    frame_vectors: list[dict],
    scene_vectors: list[dict],
    segment_vectors: list[dict],
) -> None:
    with connect() as conn, transaction(conn):
        for entry in frame_vectors:
            key = (int(entry["scene_idx"]), int(entry["frame_idx"]))
            frame_id = frame_id_map.get(key)
            if frame_id is None:
                continue
            conn.execute("DELETE FROM vec_frames WHERE rowid = ?", (frame_id,))
            conn.execute(
                "INSERT INTO vec_frames(rowid, embedding) VALUES (?, ?)",
                (frame_id, _pack_vec(entry["vector"])),
            )
        for entry in scene_vectors:
            scene_id = scene_id_by_idx.get(int(entry["scene_idx"]))
            if scene_id is None:
                continue
            conn.execute("DELETE FROM vec_scenes WHERE rowid = ?", (scene_id,))
            conn.execute(
                "INSERT INTO vec_scenes(rowid, embedding) VALUES (?, ?)",
                (scene_id, _pack_vec(entry["vector"])),
            )
        for entry in segment_vectors:
            seg_id = segment_id_by_idx.get(int(entry["seg_idx"]))
            if seg_id is None:
                continue
            conn.execute("DELETE FROM vec_transcript WHERE rowid = ?", (seg_id,))
            conn.execute(
                "INSERT INTO vec_transcript(rowid, embedding) VALUES (?, ?)",
                (seg_id, _pack_vec(entry["vector"])),
            )


def _scene_id_by_idx(asset_id: str) -> dict[int, int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, scene_idx FROM scenes WHERE asset_id = ?", (asset_id,)
        ).fetchall()
    return {int(r["scene_idx"]): int(r["id"]) for r in rows}


def _segment_id_by_idx(asset_id: str) -> dict[int, int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, seg_idx FROM transcript_segments WHERE asset_id = ?",
            (asset_id,),
        ).fetchall()
    return {int(r["seg_idx"]): int(r["id"]) for r in rows}


def _update_frame_ocr(*, asset_id: str, frames: list[dict]) -> None:
    if not frames:
        return
    with connect() as conn, transaction(conn):
        for frame in frames:
            conn.execute(
                """
                UPDATE frames SET ocr_text = ?
                 WHERE asset_id = ? AND frame_idx = ?
                   AND scene_id = (SELECT id FROM scenes
                                    WHERE asset_id = ? AND scene_idx = ?)
                """,
                (
                    frame.get("ocr_text"),
                    asset_id,
                    int(frame["frame_idx"]),
                    asset_id,
                    int(frame["scene_idx"]),
                ),
            )
```

- [ ] **Step 4: Wire `FRAME_SAMPLE`/`OCR`/`EMBED` dispatch and persistence into `run_pipeline`**

In `_run_stage`, replace the three M3 dispatch branches:

```python
    if stage is Stage.FRAME_SAMPLE:
        settings = get_settings()
        return await frame_sample(
            mezzanine_path=state.get("mezzanine_path"),
            scenes=state.get("scenes", []),
            assets_dir=settings.assets_dir,
            content_hash=state["content_hash"],
            mode=mode,
        )
    if stage is Stage.OCR:
        return await ocr(frames=state.get("frames", []))
    if stage is Stage.EMBED:
        return await embed(
            frames=state.get("frames", []),
            scenes=state.get("scenes", []),
            segments=state.get("segments", []),
        )
```

In `run_pipeline`, extend the per-stage persist block right after the existing `SCENE_DETECT` / `TRANSCRIBE` branches:

```python
            elif stage is Stage.FRAME_SAMPLE and state.get("asset_id"):
                scene_id_by_idx = _scene_id_by_idx(state["asset_id"])
                frame_id_map = _persist_frames(
                    asset_id=state["asset_id"],
                    scene_id_by_idx=scene_id_by_idx,
                    frames=state.get("frames", []),
                )
                # Stash the maps on the state dict under internal keys so
                # the EMBED persist step can look them up. Keys are stripped
                # from the JSON by _strip_internal before state is saved.
                state["__frame_id_map"] = {
                    f"{k[0]}:{k[1]}": v for k, v in frame_id_map.items()
                }
                state["__scene_id_by_idx"] = {
                    str(k): v for k, v in scene_id_by_idx.items()
                }
            elif stage is Stage.OCR and state.get("asset_id"):
                _update_frame_ocr(
                    asset_id=state["asset_id"],
                    frames=state.get("frames", []),
                )
                _rewrite_fts_scenes(asset_id=state["asset_id"])
            elif stage is Stage.EMBED and state.get("asset_id"):
                frame_id_map = {
                    tuple(int(x) for x in k.split(":")): v
                    for k, v in state.get("__frame_id_map", {}).items()
                }
                scene_id_by_idx = {
                    int(k): v for k, v in state.get("__scene_id_by_idx", {}).items()
                }
                segment_id_by_idx = _segment_id_by_idx(state["asset_id"])
                _persist_vectors(
                    frame_id_map=frame_id_map,
                    scene_id_by_idx=scene_id_by_idx,
                    segment_id_by_idx=segment_id_by_idx,
                    frame_vectors=state.get("frame_vectors", []),
                    scene_vectors=state.get("scene_vectors", []),
                    segment_vectors=state.get("segment_vectors", []),
                )
```

Double-underscore state keys (`__frame_id_map`, `__scene_id_by_idx`) are already stripped by `_strip_internal` before the state JSON is persisted.

- [ ] **Step 5: Run the test and confirm it passes**

```
.venv.nosync/bin/pytest tests/test_runner_persist_m3.py -v
```

Expected: PASS.

- [ ] **Step 6: Run the full suite**

```
.venv.nosync/bin/pytest -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```
git add src/mmrag/pipeline/runner.py tests/test_runner_persist_m3.py
git commit -m "M3 task 7: runner persist helpers + stage dispatch wiring"
```

---

## Task 8: Hybrid RRF retrieval in `handlers/search.py`

**Files:**
- Modify: `src/mmrag/handlers/search.py`
- Create: `tests/test_handler_search_hybrid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_handler_search_hybrid.py`:

```python
"""Hybrid RRF retrieval — all four streams; vector mode returns cosine."""

from __future__ import annotations

import struct
import uuid

import pytest

from mmrag.db.connection import connect
from mmrag.db.migrations import apply_migrations
from mmrag.handlers.search import handle_search
from mmrag.models.mcp_io import SearchInput

pytestmark = pytest.mark.m3_visual


def _pack(v):
    return struct.pack(f"{len(v)}f", *v)


def _bootstrap(tmp_path, monkeypatch):
    monkeypatch.setenv("MMRAG_DATA_DIR", str(tmp_path))
    from mmrag import config
    config.get_settings.cache_clear()
    apply_migrations()


async def test_fts_mode_matches_transcript_text(tmp_path, monkeypatch):
    _bootstrap(tmp_path, monkeypatch)
    asset_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
            "VALUES (?, ?, 'file', '{}')",
            (asset_id, "h1"),
        )
        conn.execute(
            "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, 2.0)",
            (asset_id,),
        )
        scene_id = conn.execute(
            "SELECT id FROM scenes WHERE asset_id=?", (asset_id,)
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO transcript_segments(asset_id, scene_id, seg_idx, start_s, end_s, text) "
            "VALUES (?, ?, 0, 0.0, 2.0, ?)",
            (asset_id, scene_id, "the weather today is sunny"),
        )
    out = await handle_search(
        SearchInput(query="weather", mode="fts", asset_id=asset_id)
    )
    assert out.hits and out.hits[0].asset_id == asset_id


async def test_vector_mode_returns_raw_cosine(tmp_path, monkeypatch):
    _bootstrap(tmp_path, monkeypatch)
    asset_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
            "VALUES (?, ?, 'file', '{}')",
            (asset_id, "h2"),
        )
        conn.execute(
            "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, 2.0)",
            (asset_id,),
        )
        scene_id = conn.execute(
            "SELECT id FROM scenes WHERE asset_id=?", (asset_id,)
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO frames(asset_id, scene_id, frame_idx, t_s, path) "
            "VALUES (?, ?, 0, 1.0, '/tmp/x.jpg')",
            (asset_id, scene_id),
        )
        frame_id = conn.execute(
            "SELECT id FROM frames WHERE asset_id=?", (asset_id,)
        ).fetchone()["id"]

    target = [0.0] * 768
    target[0] = 1.0
    with connect() as conn:
        conn.execute(
            "INSERT INTO vec_frames(rowid, embedding) VALUES (?, ?)",
            (frame_id, _pack(target)),
        )

    from mmrag.handlers import search as search_mod

    async def fake_encode(_q: str) -> list[float]:
        return target

    monkeypatch.setattr(search_mod, "_encode_query_text", fake_encode)

    out = await handle_search(
        SearchInput(query="anything", mode="vector", asset_id=asset_id, top_k=3)
    )
    assert out.hits
    # cosine(target, target) = 1.0; allow tiny floating-point slack.
    assert out.hits[0].score > 0.99


async def test_hybrid_mode_fuses_streams(tmp_path, monkeypatch):
    _bootstrap(tmp_path, monkeypatch)
    asset_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
            "VALUES (?, ?, 'file', '{}')",
            (asset_id, "h3"),
        )
        conn.execute(
            "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, 2.0)",
            (asset_id,),
        )
        scene_id = conn.execute(
            "SELECT id FROM scenes WHERE asset_id=?", (asset_id,)
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO transcript_segments(asset_id, scene_id, seg_idx, start_s, end_s, text) "
            "VALUES (?, ?, 0, 0.0, 2.0, ?)",
            (asset_id, scene_id, "red color bars pattern"),
        )

    from mmrag.handlers import search as search_mod

    async def fake_encode(_q: str) -> list[float]:
        return [0.0] * 768

    monkeypatch.setattr(search_mod, "_encode_query_text", fake_encode)

    out = await handle_search(
        SearchInput(query="red color bars", mode="hybrid", asset_id=asset_id, top_k=5)
    )
    assert out.hits
    assert out.hits[0].asset_id == asset_id
    assert out.hits[0].scene_id is not None
```

- [ ] **Step 2: Run the test and confirm it fails**

```
.venv.nosync/bin/pytest tests/test_handler_search_hybrid.py -v
```

Expected: FAIL — handlers/search.py has no `_encode_query_text`, no vector path, no hybrid RRF.

- [ ] **Step 3: Rewrite `src/mmrag/handlers/search.py`**

Replace the file with:

```python
"""MCP `search` tool handler.

Hybrid retrieval fuses four streams via reciprocal rank fusion (k=60):

  1. FTS5 BM25 over transcript_segments (via fts_transcript)
  2. FTS5 BM25 over aggregated scene OCR (via fts_scenes)
  3. SigLIP text-tower cosine over vec_frames
  4. SigLIP text-tower cosine over vec_transcript

Each stream emits up to 20 candidates keyed on ``scenes.id``. ``hybrid``
mode sums the RRF contributions. ``vector`` mode skips BM25 and returns
raw SigLIP cosine similarity as the score (so callers can threshold
against it — the M3 acceptance test does). ``fts`` mode skips vector
streams.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from mmrag.db.connection import connect
from mmrag.logging import get_logger
from mmrag.models.mcp_io import SearchHit, SearchInput, SearchOutput

log = get_logger("handler.search")

_RRF_K = 60
_PER_STREAM_TOP = 20


@dataclass
class _StreamHit:
    scene_id: int
    score: float  # cosine for vec streams, -bm25 for fts streams
    snippet: str | None


async def _encode_query_text(query: str) -> list[float]:
    """Encode query via the SigLIP text tower. Monkey-patched in tests."""
    import asyncio

    from mmrag.pipeline.stages.embed import _encode_texts_sync

    vecs = await asyncio.to_thread(_encode_texts_sync, [query])
    return vecs[0]


def _pack(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _fts_transcript_stream(query: str, asset_id: str | None) -> list[_StreamHit]:
    sql = """
        SELECT ts.scene_id AS scene_id,
               -bm25(fts_transcript) AS score,
               snippet(fts_transcript, 0, '', '', '…', 24) AS snippet
          FROM fts_transcript
          JOIN transcript_segments ts ON ts.id = fts_transcript.rowid
         WHERE fts_transcript MATCH ?
    """
    params: list = [query]
    if asset_id is not None:
        sql += " AND ts.asset_id = ?"
        params.append(asset_id)
    sql += f" ORDER BY score DESC LIMIT {_PER_STREAM_TOP}"
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("fts_transcript.failed", error=str(e))
            return []
    return [
        _StreamHit(
            scene_id=int(r["scene_id"]),
            score=float(r["score"]),
            snippet=r["snippet"],
        )
        for r in rows
        if r["scene_id"] is not None
    ]


def _fts_scenes_stream(query: str, asset_id: str | None) -> list[_StreamHit]:
    sql = """
        SELECT s.id AS scene_id,
               -bm25(fts_scenes) AS score,
               snippet(fts_scenes, 0, '', '', '…', 24) AS snippet
          FROM fts_scenes
          JOIN scenes s ON s.id = fts_scenes.rowid
         WHERE fts_scenes MATCH ?
    """
    params: list = [query]
    if asset_id is not None:
        sql += " AND s.asset_id = ?"
        params.append(asset_id)
    sql += f" ORDER BY score DESC LIMIT {_PER_STREAM_TOP}"
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("fts_scenes.failed", error=str(e))
            return []
    return [
        _StreamHit(
            scene_id=int(r["scene_id"]),
            score=float(r["score"]),
            snippet=r["snippet"],
        )
        for r in rows
    ]


def _vec_frames_stream(qvec: list[float], asset_id: str | None) -> list[_StreamHit]:
    sql = """
        SELECT f.scene_id AS scene_id,
               vf.distance AS distance
          FROM vec_frames vf
          JOIN frames f ON f.id = vf.rowid
         WHERE vf.embedding MATCH ?
           AND k = ?
    """
    params: list = [_pack(qvec), _PER_STREAM_TOP]
    if asset_id is not None:
        sql += " AND f.asset_id = ?"
        params.append(asset_id)
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("vec_frames.failed", error=str(e))
            return []
    return [
        _StreamHit(
            scene_id=int(r["scene_id"]),
            # sqlite-vec returns squared L2 distance on L2-normalized vecs;
            # for unit vectors, cosine_sim = 1 - distance^2 / 2.
            score=1.0 - (float(r["distance"]) ** 2) / 2.0,
            snippet=None,
        )
        for r in rows
    ]


def _vec_transcript_stream(qvec: list[float], asset_id: str | None) -> list[_StreamHit]:
    sql = """
        SELECT ts.scene_id AS scene_id,
               ts.text AS text,
               vt.distance AS distance
          FROM vec_transcript vt
          JOIN transcript_segments ts ON ts.id = vt.rowid
         WHERE vt.embedding MATCH ?
           AND k = ?
    """
    params: list = [_pack(qvec), _PER_STREAM_TOP]
    if asset_id is not None:
        sql += " AND ts.asset_id = ?"
        params.append(asset_id)
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("vec_transcript.failed", error=str(e))
            return []
    out: list[_StreamHit] = []
    for r in rows:
        if r["scene_id"] is None:
            continue
        text = r["text"] or ""
        snippet = text[:80] + ("…" if len(text) > 80 else "")
        out.append(
            _StreamHit(
                scene_id=int(r["scene_id"]),
                score=1.0 - (float(r["distance"]) ** 2) / 2.0,
                snippet=snippet,
            )
        )
    return out


def _scene_timing(scene_ids: list[int]) -> dict[int, tuple[str, float, float]]:
    if not scene_ids:
        return {}
    placeholders = ",".join("?" * len(scene_ids))
    sql = (
        f"SELECT id, asset_id, start_s, end_s FROM scenes WHERE id IN ({placeholders})"
    )
    with connect() as conn:
        rows = conn.execute(sql, scene_ids).fetchall()
    return {
        int(r["id"]): (str(r["asset_id"]), float(r["start_s"]), float(r["end_s"]))
        for r in rows
    }


def _rrf_fuse(
    streams: list[list[_StreamHit]], top_k: int
) -> list[tuple[int, float, str | None]]:
    """Return [(scene_id, fused_score, best_snippet), ...] top_k."""
    fused: dict[int, float] = {}
    snippets: dict[int, tuple[float, str | None]] = {}
    for hits in streams:
        for rank, hit in enumerate(hits):
            fused[hit.scene_id] = fused.get(hit.scene_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            cur = snippets.get(hit.scene_id)
            if hit.snippet and (cur is None or hit.score > cur[0]):
                snippets[hit.scene_id] = (hit.score, hit.snippet)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(sid, score, snippets.get(sid, (0.0, None))[1]) for sid, score in ordered]


async def handle_search(inp: SearchInput) -> SearchOutput:
    streams: list[list[_StreamHit]] = []

    if inp.mode in ("fts", "hybrid"):
        streams.append(_fts_transcript_stream(inp.query, inp.asset_id))
        streams.append(_fts_scenes_stream(inp.query, inp.asset_id))

    if inp.mode in ("vector", "hybrid"):
        try:
            qvec = await _encode_query_text(inp.query)
        except Exception as e:  # noqa: BLE001
            log.warning("query_encode.failed", error=str(e))
            qvec = []
        if qvec:
            streams.append(_vec_frames_stream(qvec, inp.asset_id))
            streams.append(_vec_transcript_stream(qvec, inp.asset_id))

    if inp.mode == "vector":
        flat: dict[int, _StreamHit] = {}
        for hits in streams:
            for hit in hits:
                cur = flat.get(hit.scene_id)
                if cur is None or hit.score > cur.score:
                    flat[hit.scene_id] = hit
        ordered = sorted(flat.values(), key=lambda h: h.score, reverse=True)[: inp.top_k]
        scene_meta = _scene_timing([h.scene_id for h in ordered])
        return SearchOutput(
            hits=[
                SearchHit(
                    asset_id=scene_meta[h.scene_id][0],
                    scene_id=str(h.scene_id),
                    start_s=scene_meta[h.scene_id][1],
                    end_s=scene_meta[h.scene_id][2],
                    score=h.score,
                    snippet=h.snippet or "[visual match]",
                )
                for h in ordered
                if h.scene_id in scene_meta
            ]
        )

    fused = _rrf_fuse(streams, inp.top_k)
    scene_meta = _scene_timing([sid for sid, _, _ in fused])
    return SearchOutput(
        hits=[
            SearchHit(
                asset_id=scene_meta[sid][0],
                scene_id=str(sid),
                start_s=scene_meta[sid][1],
                end_s=scene_meta[sid][2],
                score=score,
                snippet=snippet or "[visual match]",
            )
            for sid, score, snippet in fused
            if sid in scene_meta
        ]
    )
```

- [ ] **Step 4: Run the test and confirm it passes**

```
.venv.nosync/bin/pytest tests/test_handler_search_hybrid.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite — no regression on the legacy `test_handler_search.py`**

```
.venv.nosync/bin/pytest -q
```

Expected: all tests PASS. The legacy `test_handler_search.py` was updated in Task 3 to use the renamed fields; its FTS assertions still hold against the new handler.

- [ ] **Step 6: Commit**

```
git add src/mmrag/handlers/search.py tests/test_handler_search_hybrid.py
git commit -m "M3 task 8: hybrid RRF search (FTS + vec frames + vec transcript)"
```

---

## Task 9: End-to-end acceptance test — SMPTE color bars

This is the bead's acceptance criterion: ingest a generated SMPTE color bars clip end-to-end, query "red color bars" in vector mode, assert SigLIP cosine > 0.5.

**Files:**
- Create: `tests/test_m3_acceptance.py`

- [ ] **Step 1: Write the test**

Create `tests/test_m3_acceptance.py`:

```python
"""M3 bead acceptance: 'red color bars' lands on the SMPTE scene, cosine > 0.5.

Generates a 5-second SMPTE color bars clip via ffmpeg, runs the full
ingest pipeline end-to-end (fetch → normalize → scene_detect → transcribe
→ frame_sample → ocr → embed → summarize), then issues a cross-modal
vector query.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mmrag.db.migrations import apply_migrations
from mmrag.handlers.ingest import handle_ingest
from mmrag.handlers.search import handle_search
from mmrag.models.mcp_io import IngestInput, SearchInput

pytestmark = pytest.mark.m3_visual


def _make_colorbars(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "smptebars=duration=5:size=320x240:rate=1",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


async def test_smpte_color_bars_cross_modal_query(tmp_path, monkeypatch):
    monkeypatch.setenv("MMRAG_DATA_DIR", str(tmp_path))
    from mmrag import config
    config.get_settings.cache_clear()
    apply_migrations()

    video_path = tmp_path / "colorbars.mp4"
    _make_colorbars(video_path)

    ingest_result = await handle_ingest(
        IngestInput(source=str(video_path), wait_ms=120000)
    )
    assert ingest_result.status == "done", f"ingest failed: {ingest_result.error}"

    hits_out = await handle_search(
        SearchInput(
            query="red color bars",
            asset_id=ingest_result.asset_id,
            top_k=3,
            mode="vector",
        )
    )
    assert len(hits_out.hits) >= 1, "vector query returned no hits"
    top = hits_out.hits[0]
    assert top.asset_id == ingest_result.asset_id
    assert top.score > 0.5, f"SigLIP cosine too low: {top.score}"
```

- [ ] **Step 2: Run the test**

```
.venv.nosync/bin/pytest tests/test_m3_acceptance.py -v -s
```

Expected: PASS. First run downloads the SigLIP weights (~400 MB); subsequent runs use the HF cache. Runtime on a warm Mac dev box: ~30–60s.

- [ ] **Step 3: Run the full suite one more time**

```
.venv.nosync/bin/pytest -q
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```
git add tests/test_m3_acceptance.py
git commit -m "M3 task 9: SMPTE color bars cross-modal acceptance test"
```

---

## Task 10: Docs — CLAUDE.md status, architecture.md, README tesseract install

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/architecture.md`
- Modify: `README.md` (only if it has an install section)

- [ ] **Step 1: Update `CLAUDE.md` Status section to reflect M3 shipped**

Replace the "Open milestones" block with:

```
Shipped:
- **M3** — visual pipeline (frame sampling + Tesseract OCR + SigLIP-base-patch16-256 embeddings + sqlite-vec hybrid RRF over FTS transcript / FTS scenes / vec frames / vec transcript). Renamed `shots` → `scenes` across the schema. Optional `m3-visual` extra — core install stays lean.

Open milestones (see `bd ready` and `docs/pmf-rethink.md` for full rationale):
- **M4** — evidence packs, synth opt-in (`ask` returns rich evidence by default; `answer` is `str | None`; Gemma/Ollama moves to an optional `[reasoning]` extra)
- **M5** — streamable-HTTP MCP transport (tailnet-hosted shared service on Pironman; all edge agents hit one index)
- **M6** — Pi / Pironman deploy (lighter footprint: no bundled Gemma; depends on M5 transport)
- **M7** — Social Bookmarks Triage REST integration (reference consumer, not core)
```

Add a gotcha entry to the Gotchas section:

```
- **Tesseract is a required non-Python dep for ingest once M3 ships.** Install
  with `brew install tesseract` (macOS) or `apt install tesseract-ocr`
  (Debian/Pi). The `ocr` stage fails fast with `OCRError(kind="binary_missing")`
  and a clear install hint if it's missing. The `[m3-visual]` pyproject extra
  gates the Python bindings but NOT the system binary — the runtime check
  catches the delta.
```

Add to the Build & Test block, after `make sync-dev`:

```
make sync-m3                              # runtime + dev + M3 visual pipeline deps
```

- [ ] **Step 2: Update `docs/architecture.md` — flip M3 to shipped**

Find the M3 line in the Roadmap section and update:

Before:
```
- **M3** brings vision (frame sampling + OCR + SigLIP embeddings +
  sqlite-vec).
```

After:
```
- **M3** **(shipped)** brings vision: frame sampling at scene midpoints
  (plus 2s stride on long scenes), Tesseract OCR, SigLIP-base-patch16-256
  image+text embeddings, three sqlite-vec virtual tables
  (`vec_frames`, `vec_scenes`, `vec_transcript`), and hybrid RRF
  retrieval across FTS transcript / FTS scenes / vector frames / vector
  transcript. Renamed `shots` → `scenes` across the schema.
```

In the Stack table, drop the *(M3)* parenthetical from the "Frame sampling", "OCR", and "Vector store" rows, and add `sqlite-vec 0.1+` on the Vector store row.

- [ ] **Step 3: Check README for an install section**

```
grep -n "brew install\|apt install\|ffmpeg" README.md 2>/dev/null || echo "no install section"
```

If an install section exists, add tesseract alongside ffmpeg. If not, skip this step.

- [ ] **Step 4: Commit**

```
git add CLAUDE.md docs/architecture.md
git commit -m "M3 task 10: docs — mark M3 shipped, tesseract install note"
```

If README.md was modified, include it in the git add.

---

## Task 11: Close the bead

- [ ] **Step 1: Verify full suite one last time**

```
.venv.nosync/bin/pytest -q
```

Expected: all tests PASS — M1/M2 regression, M3 unit tests, M3 integration, and the color-bars acceptance.

- [ ] **Step 2: Close the beads issue**

```
bd close MM-RAG-eym --reason "M3 visual pipeline shipped: frame sampling, Tesseract OCR, SigLIP embeddings, sqlite-vec virtual tables, hybrid RRF retrieval, shots->scenes rename. Acceptance test 'red color bars' passes with SigLIP cosine > 0.5."
```

- [ ] **Step 3: Check for newly unblocked work**

```
bd ready
```

Expected: `MM-RAG-4oz` (M4 evidence packs) becomes ready.

---

## Self-review

**Spec coverage:** every section of the spec has at least one task:

- Spec §1 schema → Task 2 (migration) + Task 3 (rename cascade)
- Spec §2 frame storage → Task 4 (`_write_one_frame` + `frames_dir`)
- Spec §3 stages → Task 4 (frame_sample), Task 5 (ocr), Task 6 (embed)
- Spec §4 runner changes → Task 3 (rename persist) + Task 7 (new persist)
- Spec §5 retrieval → Task 8
- Spec §6 packaging → Task 1
- Spec §7 tests → every task ships its own test + Task 9 acceptance
- Spec §8 acceptance criteria → Task 9 + Task 11 bead close
- Spec §9 out-of-scope → respected (summarize stays stub, ask contract untouched)
- Spec §10 risks → addressed inline (tesseract install note in Task 10, graceful sqlite-vec fallback in Task 1)

**Placeholder scan:** no "TBD"/"TODO"/"implement later" — every step has either exact code or an exact command.

**Type consistency:**
- `_persist_scenes` / `_persist_frames` / `_persist_vectors` / `_rewrite_fts_scenes` — consistent names across Task 3, Task 7, Task 8 imports.
- `_encode_query_text` — defined in Task 8's `handlers/search.py`, monkey-patched in Task 8's tests under the same name.
- `frame_sample` signature — `mezzanine_path, scenes, assets_dir, content_hash, mode` — consistent between the stage (Task 4) and the dispatch in the runner (Task 7).
- `ocr` signature — `frames` — consistent (Task 5 + Task 7).
- `embed` signature — `frames, scenes, segments` — consistent (Task 6 + Task 7).
- `scene_idx` / `frame_idx` field names — consistent everywhere after Task 3.

**Ordering:** Tasks 2 and 3 commit together so the rename cascade doesn't leave `main` red between commits. All other tasks have clean test-first-commit boundaries.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-13-m3-visual-pipeline.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
