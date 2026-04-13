-- Milestone 1: walking skeleton schema.
-- Only assets + jobs (and the migration tracking table). Speech, visual,
-- and vector tables ship with their respective milestones (M2/M3).

CREATE TABLE IF NOT EXISTS schema_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS assets (
    id              TEXT PRIMARY KEY,
    content_hash    TEXT NOT NULL UNIQUE,
    source_url      TEXT,
    source_kind     TEXT NOT NULL,        -- 'url' | 'file'
    title           TEXT,
    duration_s      REAL,
    fps             REAL,
    width           INTEGER,
    height          INTEGER,
    mezzanine_path  TEXT,                 -- normalized mp4
    audio_path      TEXT,                 -- 16k mono wav
    ingested_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_assets_source_url ON assets(source_url);

CREATE TABLE IF NOT EXISTS jobs (
    id                    TEXT PRIMARY KEY,
    asset_id              TEXT REFERENCES assets(id) ON DELETE SET NULL,
    source                TEXT NOT NULL,
    mode                  TEXT NOT NULL DEFAULT 'standard',  -- 'standard' | 'shortform'
    push_to_sbt           INTEGER NOT NULL DEFAULT 0,
    status                TEXT NOT NULL,    -- 'queued' | 'running' | 'done' | 'error'
    stage                 TEXT NOT NULL,    -- last-completed or current stage name
    progress              REAL NOT NULL DEFAULT 0.0,
    retries               INTEGER NOT NULL DEFAULT 0,
    error_kind            TEXT,
    error_message         TEXT,
    wait_ms               INTEGER NOT NULL DEFAULT 30000,
    pipeline_state_json   TEXT NOT NULL DEFAULT '{}',
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_asset_id ON jobs(asset_id);
