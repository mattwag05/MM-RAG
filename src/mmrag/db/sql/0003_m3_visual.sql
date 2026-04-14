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
