-- Milestone 2: speech — shots, transcript_segments, fts_transcript.
--
-- `shots` holds scene boundaries from PySceneDetect. `transcript_segments`
-- holds faster-whisper output, each segment optionally tied to a shot.
-- `fts_transcript` is a FTS5 external-content virtual table mirroring the
-- `text` column of `transcript_segments` so search can do BM25 ranking
-- without duplicating storage. Triggers keep the FTS index in sync with
-- the content table on INSERT/UPDATE/DELETE.

CREATE TABLE IF NOT EXISTS shots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    shot_idx    INTEGER NOT NULL,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    UNIQUE(asset_id, shot_idx)
);

CREATE INDEX IF NOT EXISTS idx_shots_asset_id ON shots(asset_id);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    shot_id     INTEGER REFERENCES shots(id) ON DELETE SET NULL,
    seg_idx     INTEGER NOT NULL,
    start_s     REAL NOT NULL,
    end_s       REAL NOT NULL,
    text        TEXT NOT NULL,
    UNIQUE(asset_id, seg_idx)
);

CREATE INDEX IF NOT EXISTS idx_segments_asset_id ON transcript_segments(asset_id);
CREATE INDEX IF NOT EXISTS idx_segments_shot_id ON transcript_segments(shot_id);

-- FTS5 external-content table mirroring transcript_segments.text.
-- Using unicode61 tokenizer (default) with diacritic removal for robust
-- matching across accented speech transcriptions.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_transcript USING fts5(
    text,
    content='transcript_segments',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

-- Triggers to keep fts_transcript in lockstep with transcript_segments.
CREATE TRIGGER IF NOT EXISTS transcript_segments_ai
AFTER INSERT ON transcript_segments BEGIN
    INSERT INTO fts_transcript(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS transcript_segments_ad
AFTER DELETE ON transcript_segments BEGIN
    INSERT INTO fts_transcript(fts_transcript, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS transcript_segments_au
AFTER UPDATE ON transcript_segments BEGIN
    INSERT INTO fts_transcript(fts_transcript, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO fts_transcript(rowid, text) VALUES (new.id, new.text);
END;
