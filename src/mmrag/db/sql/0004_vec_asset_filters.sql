-- Milestone 4 prep: make sqlite-vec asset scoping a virtual-table filter.
--
-- sqlite-vec applies k= before ordinary JOIN/WHERE filters. If asset_id is
-- filtered only after MATCH, scoped search can under-deliver whenever another
-- asset owns the global nearest neighbors. sqlite-vec metadata columns let
-- the virtual table apply the filter inside the KNN query.

CREATE TABLE IF NOT EXISTS _vec_frames_backup (
    rowid INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    asset_id TEXT NOT NULL
);

INSERT OR REPLACE INTO _vec_frames_backup(rowid, embedding, asset_id)
SELECT vf.rowid, vf.embedding, f.asset_id
  FROM vec_frames vf
  JOIN frames f ON f.id = vf.rowid;

DROP TABLE IF EXISTS vec_frames;

CREATE VIRTUAL TABLE IF NOT EXISTS vec_frames USING vec0(
    embedding float[768],
    asset_id TEXT
);

INSERT INTO vec_frames(rowid, embedding, asset_id)
SELECT rowid, embedding, asset_id FROM _vec_frames_backup;

DROP TABLE IF EXISTS _vec_frames_backup;

CREATE TABLE IF NOT EXISTS _vec_scenes_backup (
    rowid INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    asset_id TEXT NOT NULL
);

INSERT OR REPLACE INTO _vec_scenes_backup(rowid, embedding, asset_id)
SELECT vs.rowid, vs.embedding, s.asset_id
  FROM vec_scenes vs
  JOIN scenes s ON s.id = vs.rowid;

DROP TABLE IF EXISTS vec_scenes;

CREATE VIRTUAL TABLE IF NOT EXISTS vec_scenes USING vec0(
    embedding float[768],
    asset_id TEXT
);

INSERT INTO vec_scenes(rowid, embedding, asset_id)
SELECT rowid, embedding, asset_id FROM _vec_scenes_backup;

DROP TABLE IF EXISTS _vec_scenes_backup;

CREATE TABLE IF NOT EXISTS _vec_transcript_backup (
    rowid INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    asset_id TEXT NOT NULL
);

INSERT OR REPLACE INTO _vec_transcript_backup(rowid, embedding, asset_id)
SELECT vt.rowid, vt.embedding, ts.asset_id
  FROM vec_transcript vt
  JOIN transcript_segments ts ON ts.id = vt.rowid;

DROP TABLE IF EXISTS vec_transcript;

CREATE VIRTUAL TABLE IF NOT EXISTS vec_transcript USING vec0(
    embedding float[768],
    asset_id TEXT
);

INSERT INTO vec_transcript(rowid, embedding, asset_id)
SELECT rowid, embedding, asset_id FROM _vec_transcript_backup;

DROP TABLE IF EXISTS _vec_transcript_backup;
