# mmrag architecture

This is a slimmed-down in-repo copy of the design that lives in
`~/.claude/plans/staged-strolling-gray.md`. The plan file is the
canonical source for the M1–M2 brainstorm rationale; `docs/pmf-rethink.md`
is the canonical source for the M3–M7 scope; this file is what
contributors read.

## Goal

Edge-optimized multimodal ingestion tool exposed as an MCP server.
Ingests video (primary), audio, and image content from public URLs
(YouTube, Shorts, TikTok, Reels) and local files; produces transcripts,
scene maps, OCR, and embeddings into a local SQLite + sqlite-vec store;
and returns rich **evidence packs** over retrieved top-k results so the
calling agent can reason with its own LLM. Bundled reasoning (Gemma 4
via Ollama) is an optional `[reasoning]` extra — core install has no
Ollama dependency. See `docs/pmf-rethink.md` for the thesis.

A laptop is the dev home. A Raspberry Pi–class server is the deployment
floor — one shared tailnet-hosted instance, streamable-HTTP MCP transport,
all edge agents hit the same index. Stack choices are made so "deploy to Pi" is
a config change, not a rewrite.

## Stack (MIT/Apache/BSD-clean)

| Layer | Choice | License |
|---|---|---|
| Language | Python >=3.11,<3.14; tested/deployed on 3.13 | PSF |
| Dep manager | uv | Apache-2 |
| MCP server | FastMCP (`mcp.server.fastmcp`) | MIT |
| REST mirror | FastAPI | MIT |
| URL fetch | yt-dlp | Unlicense/PD |
| Media transform | ffmpeg (system binary, **not bundled**) | LGPL |
| ASR | onnx-asr *(M2)* | MIT (Parakeet TDT 0.6b v3 weights: CC-BY-4.0) |
| Scene detect | PySceneDetect *(M2)* | BSD-3 |
| Embeddings | open_clip SigLIP | MIT |
| OCR | Tesseract CLI (system binary, **not bundled**) | Apache-2 |
| Vector store | sqlite-vec | Apache-2 |
| Relational | SQLite (WAL) | Public domain |
| Reasoning | Ollama (`gemma4:e4b` / `:e2b`, **user-supplied**, optional `[reasoning]` extra) *(M4)* | Gemma terms / Apache-2 |
| Settings | pydantic-settings | MIT |
| Logging | structlog | Apache-2/MIT |

Everything in v0.1.0 is MIT/Apache/BSD/PD. One non-Python piece is
required and **not bundled**: `ffmpeg` (install via package manager).

Ollama + Gemma weights are **optional** and live behind the `[reasoning]`
pyproject extra. Core install has no Ollama hard dependency — `ask`
returns evidence packs by default, and only calls the model when
`synthesize=True` AND the `[reasoning]` extra is installed. Install
with `uv pip install -e .[reasoning]` and pull `gemma4:e4b`/`gemma4:e2b`
on your Ollama host yourself; mmrag then makes HTTP calls to the local
Ollama process.

## Components

```
┌──────────────────────────────────┐     ┌────────────────────┐
│ MCP server (FastMCP stdio/HTTP)  │     │ FastAPI REST       │
│ tools: ingest/ask/search/        │     │ /ingest /ask /...  │
│        densify/status            │     │                    │
└──────────────┬───────────────────┘     └─────────┬──────────┘
               │                                   │
               └─────────────┬─────────────────────┘
                             │
                  ┌──────────▼──────────┐    ┌─────────────────┐
                  │ Handlers            │    │ Worker          │
                  │ (in-process or via  │◄───┤ drains job queue│
                  │  the worker)        │    │ (mmrag worker)  │
                  └──────────┬──────────┘    └────────▲────────┘
                             │                        │
            ┌────────────────▼────────────────────────┴─────────┐
            │ Pipeline runner — async, idempotent per stage     │
            │   1. fetch    (yt-dlp / local file)               │
            │   2. normalize (ffmpeg → mezzanine + 16k mono wav)│
            │   3. scene_detect    (M2)                          │
            │   4. transcribe      (M2)                          │
            │   5. frame_sample    (M3)                          │
            │   6. ocr             (M3)                          │
            │   7. embed           (M3)                          │
            │   8. summarize       (M4)                          │
            └────────────────┬──────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │ SQLite WAL + sqlite-vec     │
              │ + filesystem asset blobs    │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │ Evidence packs              │
              │ ask/search over shared index│
              └─────────────────────────────┘
```

The runner records the active stage in `jobs.stage` at stage start, and
persists `pipeline_state_json` plus `last_completed_stage` after each stage
finishes. A worker crash mid-job is recoverable: on next startup the worker
scans for `status in ('queued','running')` and resumes from
`pipeline_state_json.last_completed_stage`, so the interrupted active stage is
retried rather than skipped.

`ingest` is **sync-if-fast, async-if-slow**: it blocks for up to
`wait_ms` milliseconds (default 30000) and returns either a finished
`{status:done, asset_id}` or `{status:in_progress, job_id}` for polling.
In local/default profiles the handler may run the pipeline inline. Pi/tailnet
Compose sets `MMRAG_INGEST_INLINE=false`, so the MCP service only enqueues and
polls while `mmrag-worker` owns pipeline execution.

"Inline" means the request owns the job, not that the pipeline runs in the
request's process. Both hosts — the MCP server and the worker — run every job
as `mmrag run-job <job_id>` in a child process that exits when the job ends
(`pipeline/spawn.py`). Pipeline models are not reclaimable in-process: across
8 VLMs, `del` + `gc.collect()` + `torch.mps.empty_cache()` left 1.5-23 GB
resident (`docs/vlm-selection.md`, Table 3). Measured on `speech.mp4`, the
host process goes 36 MB → 36 MB across an ingest and never imports torch,
against 36 MB → 1250 MB when the same pipeline ran in-process. The job id is
the entire hand-off, since source, stage, resume state, and lease all live in
SQLite; `MMRAG_DATA_DIR` is passed to the child explicitly so an in-process
config override cannot be silently dropped. The cost is per-job process
startup and one model set per concurrent job, so `MMRAG_WORKER_CONCURRENCY`
now bounds peak memory directly.

## Job lifecycle

```
queued → running → done
            │
            └─→ error  (error_kind + error_message recorded)
```

Stages, in order, are surfaced in `jobs.stage` as the current active stage.
The last completed stage is persisted separately in
`pipeline_state_json.last_completed_stage`. The runner is idempotent and keyed
on `(content_hash, stage)`: re-running advances only the stages that have not
completed, and re-ingesting the same source under a different URL is a no-op
(same SHA-256 → same asset).

## Data model (M1 schema only)

```sql
assets(id, content_hash UNIQUE, source_url, source_kind, title,
       duration_s, fps, width, height,
       mezzanine_path, audio_path, ingested_at, metadata_json)
jobs(id, asset_id?, source, status, stage, progress,
     retries, error_kind, error_message, wait_ms,
     pipeline_state_json, created_at, updated_at)
```

M2 adds `transcript_segments` + `fts_transcript`. M3 adds `scenes`,
`frames`, `vec_*` virtual tables, and `fts_scenes`. M4 adds
`content_items`; the 2.x foundation adds `fts_content_items`, `nodes`,
and `edges`. Each lives in its own numbered SQL file under
`src/mmrag/db/sql/`.

## MCP tool surface

```
ingest(source, wait_ms=30000, profile="full")
ask(question, asset_id=None, time_range=None, top_k=5,
    synthesize=False, model=None)
search(query, asset_id=None, time_range=None, top_k=10,
       mode="hybrid"|"vector"|"fts")
status(job_id)
```

**Evidence-first contract.** `ask` returns rich `Evidence` objects:
`{asset_id, content_item_id, scene_id, frame_id, start_s, end_s,
source_stream, snippet, score, summary, ocr_snippet, transcript_snippet}`.
`search` returns a compatible hit superset with `content_item_id`,
`source_stream`, `frame_id`, `snippet`, and `score`. A `hybrid_graph`
mode over the SQLite graph tables existed and was removed for costing
precision; the tables are still written when `MMRAG_GRAPH_ENABLED=true`
but nothing reads them. `ask` additionally returns an
optional `answer: str | None` and a `confidence` field — `answer` is only
populated when the caller passes `synthesize=True`. This keeps the core contract evidence-first and
matches the PMF thesis that edge agents
(Claude Code, Claude Desktop, self-hosted assistants) already have their own
LLMs and prefer
retrieved evidence packs to yet another inference layer.

REST-only (not exposed to MCP clients): `reindex`, `retry`,
`delete_asset`, `bulk_import`. Admin moves.

## Transport

- **Stdio MCP**: `mmrag serve-mcp`, for local subprocess
  Claude Code clients on the dev Mac.
- **Streamable-HTTP MCP**: `mmrag serve-mcp-http`, tailnet-hosted endpoint,
  shared bearer token in env (`MMRAG_MCP_TOKEN`), Tailscale-only bind.
  Defaults are `MMRAG_MCP_HOST=127.0.0.1`, `MMRAG_MCP_PORT=8766`, and
  `MMRAG_MCP_PATH=/mcp`; non-loopback binds fail without a token. Published
  `.well-known/mcp-resource` for client discovery. This is
  how multiple MCP clients (AI agents, Claude Code) all query the same
  self-hosted MM-RAG instance. Each is wired as an authenticated
  `mmrag` MCP client, with the bearer token supplied by env
  interpolation rather than checked into any repo file.
- **REST** (`serve-api`): admin + debug surface, not the agent path.
- **Pi deploy**: `docker-compose.pi.yml` runs `mmrag-init`,
  `mmrag-mcp`, and `mmrag-worker` against one `/data` volume. Only MCP HTTP
  is published; REST stays off the tailnet-facing Compose path. Ingest is
  queue-only in `mmrag-mcp` for this profile.

### Deployment reference

MM-RAG deploys from a checkout on the server via `docker-compose.pi.yml`.

| Surface | Value |
|---|---|
| Discovery | `http://<server-ip>:8766/.well-known/mcp-resource` |
| MCP endpoint | `http://<server-ip>:8766/mcp` |
| Auth | `Authorization: Bearer <MMRAG_MCP_TOKEN>` |
| Token location | `.env` in the server checkout |
| MCP clients | `mcp_servers.mmrag`, token from `MCP_MMRAG_API_KEY` |
| Published services | MCP HTTP only; REST is not exposed by the Pi stack |

The discovery document advertises `transport=streamable-http`, bearer
auth metadata, and exactly the five MCP tools: `ingest`, `ask`, `search`, `densify`, and
`status`. `mmrag-init` applies SQLite migrations once, then `mmrag-mcp` and
`mmrag-worker` stay up as the long-running services. A Docker stop/restart
probe validated that worker SIGTERM releases active job leases so interrupted
jobs can be reclaimed after container restart.

After any restart or redeploy, run `make check-mcp` from a shell that
has `MMRAG_MCP_TOKEN` or `MCP_MMRAG_API_KEY` set. The check verifies public
discovery metadata, authenticated MCP `list_tools`, `status`, scoped `search`,
and evidence-first `ask`.

Production burn-in against a live MCP endpoint passed with a real YouTube
ingest: one asset produced 145 scenes, 143 transcript segments, 354 frames,
642 content items, populated sqlite-vec rows, and graph rows. Post-restart
MCP `status`, `search`, and `ask(synthesize=false)` still worked.

**v1 is single-tenant.** No caller IDs, no per-caller quotas, no
asset-visibility scoping. Auth is one shared bearer token per host.
Multi-tenant isolation is post-v1 and only lands if a concrete caller
needs it.

## Roadmap

See `bd ready` for current open issues and `docs/pmf-rethink.md` for
the full rationale behind the current milestone ordering.

- **M1** ships fetch + normalize + the schemas, handlers, and surfaces
  around them.
- **M2** brings speech (scene detect + transcribe + FTS5).
- **M3** **(shipped)** brings vision: frame sampling at scene midpoints
  (plus 2s stride on scenes >10s, with midpoint/stride collision dedup),
  Tesseract OCR (PSM 6, 10s per-frame timeout), SigLIP-base-patch16-256
  image+text embeddings (768-d, L2-normalized, CPU inference via open_clip),
  three sqlite-vec virtual tables (`vec_frames`, `vec_scenes`, `vec_transcript`),
  a plain FTS5 `fts_scenes` index maintained by application code, and
  hybrid RRF retrieval across FTS transcript / FTS scenes / vector frames /
  vector transcript. Renamed `shots` → `scenes` across the schema.
- **M4** **(shipped)** brings evidence packs: rescoped from the original "Reasoning
  pipeline" to make `ask` evidence-only by default (`answer: str |
  None`, new `synthesize: bool = False` flag), enrich `search` hits
  with evidence metadata, fix sqlite-vec asset scoping with metadata
  prefilters, add the `content_items` projection over scenes, transcript
  segments, and frames, and make stage 8 `summarize` write deterministic
  per-scene short-text distillations to `scenes.summary`.
- **M5** **(shipped)** brings streamable-HTTP MCP transport for tailnet-hosted
  deployment, shared bearer-token auth, safe loopback defaults, and public
  discovery metadata. Promoted from P3 because the PMF thesis *is* shared
  index over MCP, and stdio-only silos contradict that.
- **M6** **(shipped)** brings the Raspberry Pi deploy path. The image
  includes M3 visual runtime deps and the Compose stack exposes MCP HTTP +
  worker without bundling Gemma 4 or Ollama.
- **M7 is dropped.** The reference-consumer push client was removed in
  MM-RAG-rrh: its receiver was never locatable, so the path was never
  validated end to end, and an app-specific integration cannot ship in a
  repo destined to be a generic plugin. Migration 0010 drops the column.
- **post-v1**: bundled reasoning `[reasoning]` extra (`MM-RAG-rif`).
