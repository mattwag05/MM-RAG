-- Milestone 4 foundation: unified content item projection.
--
-- This table mirrors the current video/audio/image artifacts into a single
-- shape that future document ingestion and graph retrieval can share without
-- rewriting the existing staged pipeline.

CREATE TABLE IF NOT EXISTS content_items (
    id             TEXT PRIMARY KEY,
    asset_id       TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    item_type      TEXT NOT NULL,
    source_id      TEXT NOT NULL,
    chunk_idx      INTEGER NOT NULL,
    scene_id       INTEGER REFERENCES scenes(id) ON DELETE CASCADE,
    frame_id       INTEGER REFERENCES frames(id) ON DELETE CASCADE,
    segment_id     INTEGER REFERENCES transcript_segments(id) ON DELETE CASCADE,
    page_idx       INTEGER,
    start_s        REAL,
    end_s          REAL,
    text           TEXT,
    caption        TEXT,
    file_path      TEXT,
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_content_items_asset_id ON content_items(asset_id);
CREATE INDEX IF NOT EXISTS idx_content_items_type ON content_items(item_type);
CREATE INDEX IF NOT EXISTS idx_content_items_scene_id ON content_items(scene_id);
CREATE INDEX IF NOT EXISTS idx_content_items_frame_id ON content_items(frame_id);
CREATE INDEX IF NOT EXISTS idx_content_items_segment_id ON content_items(segment_id);
