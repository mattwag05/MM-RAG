-- Milestone: ingest-time frame captions (MM-RAG-yzt).
--
-- A scene with no speech and no on-screen text previously produced the
-- summarize.py constant "No transcript or OCR text detected." — SigLIP
-- retrieval found the right moment and the evidence layer had nothing to
-- hand back. This column stores a VLM caption written once at ingest, so
-- it is an indexing artifact in the same category as ocr_text: no
-- request-time model call, and `synthesize=false` stays the default.
--
-- Only the frames that need it are captioned (scene midpoint, where both
-- transcript and OCR are empty), so this column is NULL for most rows.
--
-- content_items already has its own `caption` column (0005), and its FTS
-- projection already indexes text+caption, so no change is needed there.

ALTER TABLE frames ADD COLUMN caption TEXT;
