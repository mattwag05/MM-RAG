# M3 — Visual pipeline design

> **Scope:** `MM-RAG-eym` (M3). Implements stages 5–7 (frame sample, OCR,
> SigLIP embeddings), migration 0003, hybrid RRF retrieval, and the
> `m3-visual` optional extra. Stage 8 `summarize` stays a stub for M4.
> No change to M1/M2 contracts beyond the `shots → scenes` rename.

## Context

M2 shipped the speech pipeline (PySceneDetect shot boundaries,
faster-whisper transcription, FTS5 BM25 search). M3 adds the visual leg:
sample frames at scene boundaries, OCR them, embed images and text into
a unified SigLIP vector space, and fuse all of this with the existing
transcript FTS index via reciprocal rank fusion.

Two design decisions locked in during brainstorming:

1. **Rename `shots` → `scenes`** across the schema. The PMF rethink doc,
   the bead, and `SearchHit.scene_id` all use "scene"; M2's `shots` table
   drifts. Rename now while there's no production data.
2. **SigLIP text tower for everything.** One 768-d vector space,
   cross-modal retrieval ("red color bars" → matching frame) works
   directly. FTS5 keeps carrying keyword retrieval, so the sentence
   embedder we'd otherwise need isn't on the critical path.

Acceptance criterion from the bead: text query `"red color bars"` lands
on the corresponding scene with SigLIP cosine similarity > 0.5.

## Architecture

Four logical changes, all within the existing pipeline-stage contract:

```
┌────────────┐     ┌─────────────────┐     ┌──────────┐     ┌──────────┐
│ frames/    │     │ scenes (renamed │     │ vec_*    │     │ search   │
│ directory  │──▶──│ from shots)     │──▶──│ tables   │──▶──│ handler  │
│ (JPEGs)    │     │ + frames table  │     │ (sqlite- │     │ (RRF     │
│            │     │ + fts_scenes    │     │ vec)     │     │ fusion)  │
└────────────┘     └─────────────────┘     └──────────┘     └──────────┘
     ▲                    ▲                     ▲                ▲
     │                    │                     │                │
stage 5              stage 5 + 6             stage 7         new streams
frame_sample         _persist_frames          embed          in handlers/
                                                             search.py
```

All stages remain idempotent and resumable via the existing runner's
per-stage state persistence.

## 1. Schema — migration `0003_m3_visual.sql`

### Rename

- `ALTER TABLE shots RENAME TO scenes`
- `ALTER TABLE scenes RENAME COLUMN shot_idx TO scene_idx`
- `ALTER TABLE transcript_segments RENAME COLUMN shot_id TO scene_id`
- Rename indexes: `idx_shots_asset_id` → `idx_scenes_asset_id`,
  `idx_segments_shot_id` → `idx_segments_scene_id`.
- Drop and recreate FK references to `shots(id)` as `scenes(id)` (SQLite
  handles this via the rename; verify `PRAGMA foreign_key_list` on
  `transcript_segments` after migration).

### Add column

```sql
ALTER TABLE scenes ADD COLUMN summary TEXT;
```

M3 leaves this NULL. M4's stage 8 `summarize` populates it.

### New `frames` table

```sql
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
```

### sqlite-vec extension load

`db/connection.py` gains a one-time extension load on every new
connection:

```python
conn.enable_load_extension(True)
try:
    import sqlite_vec
    sqlite_vec.load(conn)
finally:
    conn.enable_load_extension(False)
```

Failure mode: if `sqlite_vec` isn't installed (core-only install), the
`ImportError` is caught and logged once per process; vec-dependent
queries in `handlers/search.py` degrade to FTS-only.

### Vector tables (768-d, SigLIP-base-patch16-256 output)

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS vec_frames USING vec0(
    embedding float[768]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_scenes USING vec0(
    embedding float[768]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_transcript USING vec0(
    embedding float[768]
);
```

Rowid convention: `vec_frames.rowid = frames.id`,
`vec_scenes.rowid = scenes.id`,
`vec_transcript.rowid = transcript_segments.id`. No explicit FK (sqlite-vec
virtual tables don't support it), enforced in application code via the
runner persist helpers.

### FTS scenes (OCR-aggregated)

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS fts_scenes USING fts5(
    text,
    tokenize='unicode61 remove_diacritics 2'
);
```

**Not** external-content. Maintained entirely by application code (the
new `_persist_frames` runner helper rewrites each affected scene's row
after OCR completes). Rowid = `scenes.id`. This is simpler than a trigger
that aggregates frame OCR text in pure SQL.

## 2. Frame storage on disk

`{settings.assets_dir}/{content_hash}/frames/{scene_idx:04d}_{frame_idx:02d}.jpg`

- JPEG quality 90.
- Written by stage 5 via ffmpeg shell-out through the existing subprocess
  wrapper (`pipeline/subprocess_util.py`).
- Overwritten in place on re-ingest (idempotent).
- Cleanup is the `delete_asset` admin move's responsibility (out of M3
  scope).

## 3. Pipeline stages

### Stage 5 — `frame_sample`

**Signature:**
```python
async def frame_sample(
    *,
    mezzanine_path: str,
    scenes: list[dict],
    assets_dir: Path,
    content_hash: str,
    mode: str,
) -> dict:
    # returns {"frames": list[dict]}
```

**Sampling rule:** for each scene, sample at `midpoint_s = (start_s + end_s) / 2`.
If `end_s - start_s > 10.0`, additionally sample every 2s starting at
`start_s + 1.0`. (1 fps is too many frames for Pi budgets on long scenes;
a 2s stride stays inside 30 frames for a 60s scene.)

**ffmpeg invocation** (per frame, via subprocess wrapper):
```
ffmpeg -y -ss <t_s> -i <mezzanine_path> -frames:v 1 -q:v 3 <out_path>
```

Per-frame timeout 15s, SIGTERM→SIGKILL escalation.

**Output per frame:**
```python
{
    "scene_idx": int,
    "frame_idx": int,
    "t_s": float,
    "path": str,            # absolute path
    "width": int,
    "height": int,
}
```

Width/height read via `PIL.Image.open().size` after writing. Patch
returned: `{"frames": [...]}`.

### Stage 6 — `ocr`

**Signature:**
```python
async def ocr(*, frames: list[dict]) -> dict:
    # returns {"frames": list[dict]} with ocr_text attached
```

**Implementation:** for each frame,
`pytesseract.image_to_string(Image.open(path), config="--psm 6").strip()`.
PSM 6 = "assume a single uniform block of text" — robust for burned-in
captions, on-screen UI, slides, title cards. Failure on any one frame
(pytesseract exception) sets `ocr_text = ""` and logs a warning, but
does not fail the stage.

**Hard error:** `OCRError(kind="binary_missing")` if
`pytesseract.get_tesseract_version()` raises at import time — clear
install hint ("brew install tesseract" / "apt install tesseract-ocr").

**Per-frame timeout:** 10s via `concurrent.futures.ThreadPoolExecutor`
with `future.result(timeout=10)`. OCR runs sequentially within the
thread pool (no parallelism) — scaling to multi-worker is post-v1.

### Stage 7 — `embed`

**Signature:**
```python
async def embed(
    *,
    frames: list[dict],
    scenes: list[dict],
    segments: list[dict],
) -> dict:
    # returns {"frame_vectors": [...], "scene_vectors": [...],
    #         "segment_vectors": [...], "vectors_written": int}
```

**Model loading:** `open_clip.create_model_and_transforms('hf-hub:timm/ViT-B-16-SigLIP-256')`
on first use, cached in a module-level global (`_MODEL_CACHE`). Force
`device="cpu"`, model in eval mode, `torch.set_grad_enabled(False)`.

**Frame embeddings:** open each frame's JPEG via Pillow, preprocess via
the transform returned by open_clip, batch-encode via the image tower.
Batch size = 8 frames. Output L2-normalized 768-d float32 vectors. Emit
`[(frame_id_placeholder, vector), ...]` — frame IDs come from the runner
persist helper after the DB insert.

**Scene embeddings:** group frame vectors by `scene_idx`, compute the
L2-normalized mean of each group. No second forward pass.

**Transcript embeddings:** tokenize each segment's text via
`open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-256')`, encode via
the text tower, L2-normalize. Batch size = 16.

**Memory budget:** SigLIP-base-patch16-256 is ~200 MB on CPU. Torch CPU
runtime overhead ~300 MB. Total ~500 MB peak during stage 7. Pi budget
is ~1.5 GB RAM target (per PMF rethink); fits.

## 4. Runner changes

### New persist helpers

- `_persist_frames(asset_id, scene_id_by_idx, frames)` — UPSERT rows into
  `frames` keyed on `(asset_id, scene_id, frame_idx)`. Returns a map
  `{(scene_idx, frame_idx): frames.id}` so stage 7 can key its vectors.
- `_persist_vectors(frame_id_map, scene_id_by_idx, segment_id_by_idx,
  frame_vectors, scene_vectors, segment_vectors)` — INSERT/REPLACE into
  `vec_frames`, `vec_scenes`, `vec_transcript` keyed on the real row
  IDs.
- `_rewrite_fts_scenes(asset_id)` — after OCR persists, DELETE all
  `fts_scenes` rows where rowid is in the asset's scenes, then INSERT a
  fresh row per scene with `text = " ".join(frames.ocr_text for frames in
  scene)`.

### Rename existing helpers

- `_persist_shots` → `_persist_scenes`
- `Stage.SCENE_DETECT` patch key `shots` → `scenes` (also in
  `pipeline_state_json` — migration 0003 does not touch jobs table;
  in-flight jobs at migration time are expected to be drained or
  discarded, acceptable pre-production).

### Dispatch (`_run_stage`)

```python
if stage is Stage.FRAME_SAMPLE:
    return await frame_sample(
        mezzanine_path=state["mezzanine_path"],
        scenes=state.get("scenes", []),
        assets_dir=get_settings().assets_dir,
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

After each stage, runner calls the matching persist helper (same pattern
as `_persist_scenes` / `_persist_segments` today).

## 5. Retrieval — hybrid RRF in `handlers/search.py`

Four streams, fused with reciprocal rank fusion (`k = 60`, standard):

1. **FTS transcript** — `bm25(fts_transcript)`, top 20. Map each
   transcript segment hit to its `scene_id` via the FK.
2. **FTS scenes** — `bm25(fts_scenes)`, top 20. Already keyed on `scene_id`.
3. **Vec frames** — encode query via SigLIP text tower, cosine via
   `vec_frames MATCH vec_frames.embedding <=> ? LIMIT 20`. Join
   `frames.scene_id` to key on scene.
4. **Vec transcript** — same encoded query, cosine via
   `vec_transcript LIMIT 20`. Map to `scene_id` via
   `transcript_segments.scene_id`.

Each stream produces `{scene_id: rank}` dicts. RRF:

```
fused[scene_id] += 1 / (60 + rank)  for each stream
```

Sort by fused score descending, return top `top_k` scenes with:

- `scene_id` — `scenes.id` as string
- `start_s`, `end_s` — from `scenes` row
- `score` — fused RRF score (small, typically 0.01–0.07)
- `snippet` — picked from highest-ranked contributing stream:
  - FTS transcript winner → transcript snippet via `snippet()` SQL fn
  - FTS scenes winner → OCR excerpt
  - Vec-only winner → `"[visual match]"` placeholder

### Modes

- `mode="fts"` — streams 1 + 2 only.
- `mode="vector"` — streams 3 + 4 only. **Returns raw SigLIP cosine
  similarity as the score** (not RRF), so `score > 0.5` thresholds are
  meaningful — this is the path the bead's acceptance test hits.
- `mode="hybrid"` — all four streams fused via RRF.

### Asset scoping

`asset_id` filter applied in each stream's SQL WHERE clause. FTS path
already has this today; vec path adds `JOIN frames ON frames.id =
vec_frames.rowid WHERE frames.asset_id = ?`.

## 6. Packaging — `m3-visual` optional extra

### `pyproject.toml`

```toml
[project.optional-dependencies]
m3-visual = [
    "open-clip-torch>=2.24",
    "torch>=2.2",
    "pytesseract>=0.3.10",
    "Pillow>=10.0",
    "sqlite-vec>=0.1.0",
    "numpy>=1.26",
]
```

### `Makefile`

```makefile
sync-m3:
 UV_PROJECT_ENVIRONMENT=.venv.nosync uv sync --extra m3-visual
```

### Non-Python prerequisites

- `ffmpeg` — already required by M1.
- `tesseract` — new. Document `brew install tesseract` /
  `apt install tesseract-ocr` in `CLAUDE.md § Build & Test` and in the
  README.

### Guarded imports

Per-stage try-import at module top, with a `_AVAILABLE = False` flag on
ImportError. Stages raise `M3ExtraMissingError(kind="extra_missing",
message="install with: uv pip install mmrag[m3-visual]")` if invoked
without the deps. `handlers/search.py` in `vector` or `hybrid` mode falls
back to FTS-only and logs a one-time warning.

**Hard stance:** a host running `mmrag worker` or `mmrag ingest` MUST
have `m3-visual` installed once M3 ships. This is documented as a
breaking change in the M3 commit. Search-only read-replicas may omit the
extra and stick to `fts` mode.

## 7. Tests

### Unit

- `tests/test_pipeline_frame_sample.py` — synthetic mezzanine via
  ffmpeg `testsrc` (`ffmpeg -f lavfi -i testsrc=duration=6:size=320x240:rate=24`);
  stub 3-scene list; assert frame count, midpoint `t_s` accuracy,
  existence of JPEG files, width/height populated.
- `tests/test_pipeline_ocr.py` — generate a JPEG with black "HELLO WORLD"
  text on white background via Pillow's `ImageDraw`. Assert
  `ocr_text.upper()` contains "HELLO WORLD".
- `tests/test_pipeline_embed.py` — 2 solid-color Pillow JPEGs (red, blue)
  and 2 text segments. Assert 768-d float32 L2-normalized vectors,
  same-color frames have cosine > 0.9, red vs. blue have cosine < 0.5.
- `tests/test_db_migrations_0003.py` — apply migrations from a fresh DB;
  assert `scenes` exists, `shots` does not, `frames` exists, `vec_*`
  load, `fts_scenes` queryable.

### Integration

- `tests/test_search_hybrid.py` — ingest the same short TTS-speech
  fixture M2 uses, extended with burned-in text via ffmpeg `drawtext`;
  run `search(mode="fts")`, `search(mode="vector")`, `search(mode="hybrid")`;
  assert top-1 in each mode is the expected scene.

### Acceptance (bead criterion)

- `tests/test_m3_acceptance.py`:
  ```python
  # Generate SMPTE color bars at 320x240 for 5s at 1 fps
  subprocess.run([
      "ffmpeg", "-y", "-f", "lavfi",
      "-i", "smptebars=duration=5:size=320x240:rate=1",
      "-pix_fmt", "yuv420p", str(tmp_path / "colorbars.mp4"),
  ], check=True)
  # Ingest end-to-end
  result = await handle_ingest(IngestInput(source=str(tmp_path / "colorbars.mp4")))
  assert result.status == "done"
  # Cross-modal query
  hits = await handle_search(SearchInput(
      query="red color bars",
      asset_id=result.asset_id,
      top_k=3,
      mode="vector",
  ))
  assert len(hits.hits) >= 1
  assert hits.hits[0].score > 0.5  # SigLIP cosine, not RRF
  ```

### Pytest marker

```python
# pyproject.toml [tool.pytest.ini_options]
markers = [
    "m3_visual: requires the m3-visual optional extra",
]
```

`conftest.py` auto-skips `m3_visual`-marked tests via
`pytest.importorskip("open_clip")` when the extra is absent. Existing
M1/M2 tests continue to pass in a core-only install.

## 8. Acceptance criteria (recap from bead)

- Stages 5–7 implemented and invoked by the runner.
- Migration 0003 applies cleanly on a fresh DB and on an M2-filled DB.
- Text query `"red color bars"` against the SMPTE fixture returns the
  color bars scene as top-1 in `vector` mode with SigLIP cosine > 0.5.
- `m3-visual` extra is optional for core install but required for
  ingest. Tests auto-skip when absent.
- All existing M1/M2 tests still pass (40 current tests → 40 + new M3
  tests, all green).

## 9. Out of scope (deferred to later milestones)

- Stage 8 `summarize` — M4.
- `scenes.summary` population — M4.
- `AskInput.synthesize` / evidence-pack rewrite — M4.
- Streamable-HTTP MCP transport — M5.
- Pi deploy manifest — M6.
- Multi-tenant auth — post-v1.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Torch CPU wheel download is ~200 MB on first `sync-m3` | Documented in README; one-time cost. |
| SigLIP HF Hub download is ~400 MB on first run | Documented in test docstring; subsequent runs cached. |
| Tesseract accuracy on compressed JPEGs can be poor | PSM 6 + q:v 3 JPEGs is a reasonable middle ground; users with strict OCR needs can open a follow-up issue. |
| sqlite-vec extension loading can fail on macOS code signing | Documented fallback: handler degrades to FTS-only and logs a warning. |
| ThreadPoolExecutor OCR timeout leaks Tesseract subprocesses on kill | Accepted for v1; pytesseract spawns its own subprocess and ThreadPoolExecutor cancellation only marks the future. Follow-up issue if it bites. |
| Migration 0003 rename on an in-flight M2 job breaks `pipeline_state_json` | Pre-production assumption: no in-flight jobs at migration time. Document in commit message. |

## References

- Bead: `MM-RAG-eym`
- PMF doc: `docs/pmf-rethink.md`
- Architecture: `docs/architecture.md`
- Stage 5 stub: `src/mmrag/pipeline/stages/frame_sample.py`
- Stage 6 stub: `src/mmrag/pipeline/stages/ocr.py`
- Stage 7 stub: `src/mmrag/pipeline/stages/embed.py`
- Runner: `src/mmrag/pipeline/runner.py`
- Search handler: `src/mmrag/handlers/search.py`
- M1 migration: `src/mmrag/db/sql/0001_m1_init.sql`
- M2 migration: `src/mmrag/db/sql/0002_m2_speech.sql`
- MCP I/O: `src/mmrag/models/mcp_io.py`
