# mmrag

> **Edge-optimized multimodal RAG over video, audio, and images — exposed as an MCP server.**
> Ingest a YouTube Short, a TikTok, a Reels link, or a local screen recording.
> Get back a searchable transcript, a scene map, OCR, and evidence packs
> sitting on top of a single SQLite file. Optional synthesis can call Gemma 4
> via Ollama. Runs on your Mac. Designed to also run
> on a Raspberry Pi.

`mmrag` is a single self-hosted process that turns raw video into something an
AI agent can actually reason about. It's deliberately small, MIT-licensed, and
biased toward retrieval over brute-force frame inference — so the "smart"
multimodal model only sees the few seconds that actually matter.

> **Status: v0.1.0 (M6 Pi deploy path shipped).**
> Stages 1–7 wired end-to-end: fetch → normalize → scene_detect → transcribe
> → frame_sample → ocr → embed, with deterministic scene summaries at stage 8.
> `ask` is evidence-first by default, with
> request-time synthesis opt-in. Streamable HTTP MCP is the shared tailnet
> transport. See the [roadmap](#roadmap).

---

## Why does this exist?

When an agent sees a Reels link or a screen recording, it usually has nothing
to work with. It can read the post caption. It can maybe describe a thumbnail.
It cannot tell you what was actually said, what was on screen at 00:42, or
which clip in your library is the one where the onboarding modal appears.

`mmrag` fills that gap with a deliberately small set of moving parts:

- **One SQLite file** — assets, jobs, scenes, transcripts, OCR, vectors, FTS.
  No Qdrant, no Milvus, no Postgres. `sqlite-vec` lives inside the same DB.
- **Retrieval first, reasoning second** — `mmrag` doesn't shove a 10-minute
  video into a multimodal model. It splits the video into scenes, transcribes
  the audio with `faster-whisper`, OCRs sampled frames, and embeds everything
  into a hybrid (vector + BM25) index. `ask` returns the top-k retrieved
  evidence by default; only `synthesize=true` hands that evidence to Gemma 4
  for a final answer.
- **MCP-native** — four sharp tools: `ingest`, `ask`, `search`, `status`. Wire
  them into Claude Code, Claude Desktop, or any MCP client.
- **MIT-clean** — every Python dependency is MIT, Apache-2, BSD, or
  public-domain. No GPL/AGPL. The two non-Python pieces (ffmpeg, Ollama+Gemma
  weights) are required system installs, not bundled.

---

## What works today

- ✅ `ingest(local_file)` — sync, full pipeline through ffmpeg normalize, asset row populated with `content_hash`, `duration_s`, `fps`, `width`, `height`, `mezzanine_path`, `audio_path`
- ✅ `ingest(url)` via `yt-dlp` — best-effort URL fetch (any `yt-dlp`-supported source)
- ✅ `ingest` job model is **sync-if-fast, async-if-slow**: blocks for up to `wait_ms` (default 30000) and returns either a finished result or a `job_id` to poll
- ✅ Idempotent by content hash — re-ingesting the same file under a different URL is a no-op
- ✅ `status(job_id)` returns the live stage + progress fraction + error info
- ✅ Crash-resumable pipeline — `jobs.stage` reports the active stage, while
  `pipeline_state_json.last_completed_stage` drives safe resume/retry
- ✅ FastMCP stdio server with all 4 tools registered
- ✅ FastMCP Streamable HTTP server with shared bearer-token auth for tailnet use
- ✅ FastAPI REST mirror on `:8765` with the same surface
- ✅ Background worker (`mmrag worker`) that drains the job queue
- ✅ SQLite WAL store with migration runner
- ✅ Subprocess wrapper with `SIGTERM → SIGKILL` escalation for hung ffmpeg/whisper
- ✅ Pluggable `ModelProvider` slot for the eventual VLM swap
- ✅ Pydantic schema contract tests for every MCP tool's input/output
- ✅ Pytest end-to-end tests with auto-generated ffmpeg lavfi fixtures
- ✅ `ask(...)` returns rich evidence packs by default (`answer=null`) and can
  synthesize through Ollama/Gemma when `synthesize=true`
- ✅ Stage 8 writes deterministic per-scene summaries to `scenes.summary` and
  the `content_items` projection
- ✅ `ingest(local_document)` for Markdown, HTML, TXT, DOCX, and PDF text
  extraction into the same `content_items` retrieval path.
- ✅ `search(...)` supports `fts`, `vector`, `hybrid`, and `hybrid_graph`
  modes over transcript, OCR, document content, vectors, and graph neighbors.
- ✅ Lightweight SQLite graph tables (`nodes`, `edges`) over assets, content
  items, scenes, frames, segments, and topics.
- ✅ Optional vector backend protocol with SQLite default and a Qdrant
  selection hook for homelab experiments.
- ✅ MM-RAG-side Social Bookmarks Triage push client via `push_to_sbt=true`.
  The SBT receiver app was not available at the documented local path during
  the 2026-06-04 audit, so end-to-end SBT receiver validation is tracked
  separately.
- ✅ Pi/tailnet Docker Compose path for MCP HTTP + worker, with no bundled
  Ollama/Gemma dependency. The MCP service enqueues ingests and the worker
  owns pipeline execution in this profile.

---

## Quickstart (Mac)

```bash
brew install ffmpeg                   # required system binary (LGPL, not bundled)
brew install tesseract                # required for OCR stage (Apache-2, not bundled)

git clone <this repo>
cd MM-RAG
make sync-dev                         # uses Python 3.13 by default; installs runtime + dev deps
# For the M3 visual pipeline (frame sampling, OCR, SigLIP embeddings):
make sync-m3                          # adds torch, open-clip-torch, Pillow, etc.
make init-db                          # creates ~/.local/share/mmrag/mmrag.db
make serve-api &                      # FastAPI REST on http://127.0.0.1:8765
make serve-mcp-http &                 # Streamable HTTP MCP on http://127.0.0.1:8766/mcp
make worker &                         # drains the job queue

# Smoke test against the checked-in fixture
curl -s -X POST http://127.0.0.1:8765/ingest \
  -H 'content-type: application/json' \
  -d "{\"source\":\"$PWD/tests/fixtures/sample.mp4\",\"wait_ms\":15000}" | jq
```

> **Why `make` instead of `uv` directly?** The Makefile pins
> `UV_PROJECT_ENVIRONMENT=.venv.nosync` so the virtualenv lives in a
> directory iCloud Drive ignores. Without that, iCloud sets the macOS
> `UF_HIDDEN` flag on `.pth` files, which Python 3.13 silently skips, and
> the editable install becomes invisible. If your project root is *not*
> inside an iCloud-synced directory, `uv sync` directly works fine —
> see CLAUDE.md "Gotchas" for the full story.

You'll get back something like:

```json
{
  "status": "done",
  "asset_id": "999562c5-ba0f-4281-9d6e-e79e795999bd",
  "job_id":   "fceef714-54b6-441e-9ec9-ec74921fac97",
  "summary":  null,
  "error":    null
}
```

And then:

```bash
curl -s http://127.0.0.1:8765/asset/<asset_id> | jq
```

shows the populated asset row, including the `mezzanine_path` and `audio_path`
on disk under `~/.local/share/mmrag/assets/<content_hash>/`.

### Run the tests

```bash
make test
```

Test fixtures (a 3 s `mp4`, a 3 s `wav`, a 320×240 `png`) are generated on
first run via `ffmpeg lavfi` sources into `tests/fixtures/` and gitignored.

---

## MCP tool surface

```
ingest(source, mode="standard"|"shortform", wait_ms=30000, push_to_sbt=False)
  → { status, asset_id, job_id, summary, error }

ask(question, asset_id=None, time_range=None, top_k=5,
    model="gemma4:e4b", synthesize=False)
  → { answer, evidence: [{ asset_id, content_item_id, scene_id, start_s, end_s,
                           source_stream, snippet, score,
                           summary, ocr_snippet, transcript_snippet }],
      confidence }

search(query, asset_id=None, time_range=None, top_k=10,
       mode="hybrid"|"vector"|"fts"|"hybrid_graph")
  → { hits: [{ asset_id, content_item_id, scene_id, frame_id, start_s, end_s,
               score, snippet, source_stream }] }

status(job_id)
  → { status, stage, progress, asset_id, error }
```

**REST-only (intentionally not exposed to MCP):** `reindex`, `retry`,
`delete_asset`, `bulk_import`. These are admin moves; agents shouldn't have
those buttons.

### Wiring it into Claude Code / Claude Desktop

Add to your client's MCP config:

```json
{
  "mcpServers": {
    "mmrag": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/MM-RAG", "mmrag", "serve-mcp"]
    }
  }
}
```

Then in the chat: *"Ingest <some YouTube URL>, then ask what happens at the
30-second mark."*

### Shared tailnet MCP server

`serve-mcp-http` exposes the same four MCP tools over FastMCP's
Streamable HTTP transport. Loopback binds are allowed without a token for
local development; any non-loopback bind requires `MMRAG_MCP_TOKEN`.

```bash
export MMRAG_MCP_HOST=0.0.0.0
export MMRAG_MCP_PORT=8766
export MMRAG_MCP_PATH=/mcp
export MMRAG_MCP_PUBLIC_URL=http://mmrag.tailnet:8766
export MMRAG_MCP_TOKEN='shared-secret'
make serve-mcp-http
```

Discovery metadata is public at `/.well-known/mcp-resource`; the MCP
endpoint itself expects `Authorization: Bearer <MMRAG_MCP_TOKEN>`. The
FastAPI REST server remains an admin/debug mirror, not the shared agent
transport.

### Pi / homelab-host Docker deploy

The Pi deployment is MCP-first: one shared tailnet service on port `8766`
plus a worker draining the same SQLite volume. It includes the visual
retrieval stack (`m3-visual`, ffmpeg, Tesseract) but does not bundle Ollama,
Gemma weights, or the optional `reasoning` extra. The Pi Compose file sets
`MMRAG_INGEST_INLINE=false`, so `mmrag-mcp` stays a transport/queue service
and `mmrag-worker` runs the heavy ingest pipeline. It also sets
`MMRAG_QUERY_VECTOR_ENABLED=false`, so hybrid MCP queries use FTS/content
search without loading the SigLIP/open_clip model in the transport process.

Local loopback validation:

```bash
export MMRAG_MCP_TOKEN='dev-secret'
export MMRAG_MCP_PUBLIC_URL=http://127.0.0.1:8766
make docker-pi-config
make docker-build
make docker-pi-up
curl -s http://127.0.0.1:8766/.well-known/mcp-resource | jq
make docker-pi-down
```

homelab-host/tailnet example:

```bash
export MMRAG_MCP_TOKEN='shared-secret'
export MMRAG_PUBLISH_HOST='100.x.y.z'            # homelab-host Tailscale IP
export MMRAG_MCP_PUBLIC_URL='http://100.x.y.z:8766'
make docker-pi-up
```

`MMRAG_MCP_TOKEN` is required because the container binds MCP to
`0.0.0.0`. Do not publish the REST mirror in this stack; keep REST local for
admin/debug workflows.

Current homelab-host deployment, validated 2026-06-02 06:59 EDT / 2026-06-02 10:59 UTC:

- Host: `homelab-host` (`203.0.113.10` on Tailscale)
- Checkout: `~/Projects/MM-RAG`; last verified checkout is `b8963f2`
- Latest code-bearing deploy: `83604a7` (`b8963f2` only closes Beads tracking
  on top of the deployed code)
- Discovery: `http://203.0.113.10:8766/.well-known/mcp-resource`
- MCP endpoint: `http://203.0.113.10:8766/mcp`
- Token: `MMRAG_MCP_TOKEN` in `~/Projects/MM-RAG/.env` on homelab-host
- Services: `mmrag-init` applies migrations, `mmrag-mcp` exposes MCP only,
  and `mmrag-worker` runs ingest jobs from the shared `/data` volume
- [agent]/[agent-runtime] client: configured as the `mmrag` Streamable HTTP MCP server,
  with its bearer token read from local [agent-runtime] env as `MCP_MMRAG_API_KEY`

The deployed service was verified with the public discovery document, an
authenticated Streamable HTTP MCP `list_tools` probe, and a production burn-in
against a real YouTube video. Burn-in asset
`b30d0b6f-a449-4837-a9ad-a9f19b6fde38` produced 145 scenes, 143 transcript
segments, 354 frames, 642 content items, populated sqlite-vec rows, and graph
rows. After restarting `mmrag-mcp` and `mmrag-worker`, MCP `status`, `search`,
and `ask(synthesize=false)` still worked. The live service now includes the
`30225d7` active-stage status fix, the CPU-only Docker dependency fix, and the
atomic migration runner fix.

See [docs/homelab-host-burn-in.md](./docs/homelab-host-burn-in.md) for the exact
burn-in evidence, persisted counts, resource shape, and stabilization notes.

Post-restart health check:

```bash
export MMRAG_MCP_TOKEN='shared-secret'   # or export MCP_MMRAG_API_KEY
make check-homelab-host-mcp
```

The check verifies discovery metadata, the authenticated MCP tool surface,
`status`, scoped `search`, evidence-first `ask`, and `agent-runtime mcp test mmrag`.
Keep token values outside repo files and shell history.

---

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │ MCP server (FastMCP stdio or HTTP)          │
                │ tools: ingest / ask / search / status       │
                └───────────────┬─────────────────────────────┘
                                │ shared handler implementations
                ┌───────────────▼───────────────┐   ┌─────────────────┐
                │ FastAPI REST mirror           │   │ Worker process  │
                │ POST /ingest /ask /search     │◄──┤ drains job queue│
                │ GET  /asset/{id} /jobs/{id}   │   │ (mmrag worker)  │
                └───────────────┬───────────────┘   └────────▲────────┘
                                │                            │
                ┌───────────────▼────────────────────────────┴───────┐
                │ Pipeline (async, idempotent, phase-1 / phase-2)    │
                │ 1. fetch        (yt-dlp / local)                    │
                │ 2. normalize    (ffmpeg → mp4 mezzanine + 16k mono) │
                │ 3. scene_detect (PySceneDetect)         [M2]        │
                │ 4. transcribe   (faster-whisper int8)   [M2]        │
                │ 5. frame_sample (scene midpoint+1fps)   [M3]        │
                │ 6. ocr          (Tesseract)             [M3]        │
                │ 7. embed        (SigLIP image+text)     [M3]        │
                │ 8. summarize    (deterministic scene summaries)      │
                └────────────────┬───────────────────────────────────┘
                                 │
                        ┌────────▼─────────┐
                        │ SQLite WAL       │
                        │ + sqlite-vec     │
                        │ + assets/<hash>/ │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────────────────────────┐
                        │ Retrieval + optional reasoning       │
                        │ - sqlite-vec ANN on scenes/frames    │
                        │ - FTS5 BM25 on transcript/scenes     │
                        │ - RRF hybrid rerank                  │
                        │ - evidence pack by default           │
                        │ - synthesize=true → Ollama/Gemma     │
                        └─────────────────────────────────────┘
```

### Identity through `content_hash`

Every asset is keyed on the SHA-256 of its canonical mezzanine file. This
means re-ingesting the same video under a different URL is a no-op — the
content hash collapses to the same `assets` row.

### Pluggable VLM slot

`mmrag.providers.base.ModelProvider` is an abstract base class with a single
`generate(messages, config)` method. The default `OllamaProvider` talks to
`gemma4:e4b` / `gemma4:e2b`. Swapping in a future `LLaVAVideoProvider` or
`gemma4:video` is a constructor change, not a rewrite.

---

## License and dependencies

`mmrag` itself is **MIT**. Every Python dependency is MIT, Apache-2, BSD, or
public domain. No GPL/AGPL anywhere in the runtime tree.

| Dep             | License        | Role                            |
|-----------------|----------------|---------------------------------|
| `fastapi`       | MIT            | REST mirror                     |
| `mcp`           | MIT            | FastMCP server                  |
| `pydantic`      | MIT            | I/O contracts                   |
| `pydantic-settings` | MIT        | env-var config                  |
| `httpx`         | BSD-3          | async HTTP client               |
| `structlog`     | Apache-2/MIT   | structured logging              |
| `typer`         | MIT            | CLI                             |
| `yt-dlp`        | Unlicense (PD) | URL fetch                       |
| `uvicorn`       | BSD-3          | ASGI server                     |
| `setuptools`    | MIT            | build backend                   |

Three non-Python pieces are required and **not bundled**:

1. **`ffmpeg`** (LGPL) — install via `brew install ffmpeg` /
   `apt install ffmpeg`. LGPL is fine for an MIT Python project as long as
   we don't statically link or redistribute it, and we don't.
2. **`tesseract`** (Apache-2) — install via `brew install tesseract` /
   `apt install tesseract-ocr`. Required for the M3 OCR stage. `mmrag`
   shells out to this system binary via the pipeline subprocess wrapper
   and fails fast with a clear error if it's missing.
3. **Ollama + Gemma 4 weights.** Install Ollama from
   <https://ollama.com/download> and run `ollama pull gemma4:e4b` /
   `ollama pull gemma4:e2b`. Gemma weights are released under Google's
   Gemma terms (not MIT). `mmrag` does not bundle, redistribute, or
   fine-tune them — it merely makes HTTP calls to a local Ollama process
   you already trust.

If you're allergic to those terms, the `ModelProvider` abstraction lets you
plug in any other Ollama-served VLM (e.g. `qwen2-vl`, `minicpm-v`) by
changing one constructor argument.

---

## "Why not X?"

**Why not Qdrant or Milvus?** Because on a Raspberry Pi, a separate vector
daemon is ~half a gig of resident memory you don't have and a second process
you don't want to babysit. `sqlite-vec` lives inside the same SQLite file as
everything else, has no daemon, and graduates to Qdrant the day it actually
groans. (M3 makes this swap a one-liner via the `ModelProvider` pattern.)

**Why no speaker diarization?** `pyannote.audio` is MIT *code*, but the
pretrained diarization models are HuggingFace-gated under non-MIT terms,
which makes it dirty for an MIT project that wants to ship without surprises.
Diarization is deferred to v0.2 via `sherpa-onnx` (Apache-2). For
short-form social content (the primary use case) you rarely care *who* spoke.

**Why Gemma 4 and not a "real" video VLM?** Because it runs on the kind of
hardware you actually have. Gemma 4 has hard limits — 30 s of audio, ~60 s of
video frames in a single call — so it can't be the long-video engine on its
own. `mmrag` works around this by retrieving the top-k relevant 5–15 second
moments first and only handing those to Gemma. The architecture has a
pluggable slot for a dedicated video VLM (LLaVA-Video, VideoLLaMA, future
`gemma4:video`) when the temporal reasoning needs more.

**Why is `ingest` synchronous if I pass `wait_ms`?** Because most of what
people actually ingest interactively is short-form social content (Reels,
Shorts, TikToks). The 30-second default is enough for the bread-and-butter
case to feel synchronous; long videos correctly fall back to polling without
changing the agent's tool-call shape. Pi/tailnet Compose disables inline
execution, so `wait_ms` waits for the separate worker instead of doing heavy
work in the MCP server process.

**Why no UI?** Because the MCP tools and the REST mirror are the surface.
A UI would be a separate project layered on top.

---

## Roadmap

Tracked in `bd` (run `bd ready` to see open work). Each milestone is
independently testable; the project pauses for review between them.

| Milestone | Status | Scope |
|-----------|:------:|-------|
| **M1** | ✅ | Walking skeleton: project layout, `uv` + tested/deployed Python 3.13 default, FastMCP + 4 tool stubs, FastAPI mirror, SQLite + migrations, fetch + normalize stages, contract + pipeline tests |
| **M2** | ✅ | Scene detection (PySceneDetect) + transcription (faster-whisper int8 + word timestamps) + FTS5 transcript search |
| **M3** | ✅ | Frame sampling + Tesseract OCR + SigLIP-base-patch16-256 image+text embeddings (768-d) + sqlite-vec hybrid RRF retrieval (FTS transcript / FTS scenes / vec frames / vec transcript). Renamed `shots` → `scenes`. |
| **M4** | ✅ | Evidence packs + synth opt-in: `ask` returns evidence by default, `answer` is nullable, `synthesize=true` calls Ollama/Gemma, `content_items` projects scenes/segments/frames, and stage 8 writes deterministic scene summaries. |
| **M5** | ✅ | Streamable-HTTP MCP transport for a shared tailnet-hosted MM-RAG service, with shared bearer token and discovery metadata |
| **M6** | ✅ | Raspberry Pi / homelab-host deploy path: MCP HTTP + worker Compose stack, token-required tailnet bind, no bundled Ollama/Gemma |
| **M7** | Partial | MM-RAG-side Social Bookmarks Triage REST client is implemented; SBT-side receiver/schema validation is pending |
| **2.x foundation** | ✅ | Document ingestion via `content_items`, graph-aware `hybrid_graph` retrieval, and optional vector backend protocol |

**Deferred** (tracked, not forgotten): speaker diarization, PaddleOCR,
dedicated video VLM, UI/screen-recording mode with dense frame sampling and
scene diffing.

---

## Project layout

```
MM-RAG/
├── pyproject.toml             # uv project, MIT license metadata, setuptools backend
├── README.md
├── LICENSE                    # MIT
├── Dockerfile
├── docker-compose.yml         # Mac REST dev stack
├── docker-compose.pi.yml      # Pi/tailnet MCP + worker stack
├── docs/
│   └── architecture.md        # in-repo design (slim copy of the planning spec)
├── tests/
│   ├── conftest.py            # generates fixtures via ffmpeg lavfi
│   ├── test_contract.py       # pydantic schema validation for every MCP tool
│   ├── test_pipeline_fetch.py
│   └── test_pipeline_normalize.py
└── src/mmrag/
    ├── cli.py                 # typer: serve-mcp | serve-mcp-http | serve-api | worker | init-db
    ├── config.py              # pydantic-settings (data_dir, ollama_url, ...)
    ├── logging.py             # structlog setup
    ├── mcp_server.py          # FastMCP stdio + Streamable HTTP app factory
    ├── api.py                 # FastAPI REST mirror
    ├── worker.py              # job-queue drain
    ├── sbt_client.py          # MM-RAG-side Social Bookmarks Triage REST client
    ├── db/
    │   ├── connection.py      # WAL pragma, transaction helpers
    │   ├── migrations.py      # idempotent migration runner
    │   └── sql/
    │       ├── 0001_m1_init.sql
    │       ├── 0002_m2_speech.sql
    │       ├── 0003_m3_visual.sql
    │       ├── 0004_vec_asset_filters.sql
    │       └── 0005_content_items.sql
    ├── models/
    │   ├── asset.py
    │   ├── job.py             # JobStatus, Stage enums
    │   └── mcp_io.py          # IngestInput/Output, AskInput/Output, ...
    ├── handlers/
    │   ├── ingest.py          # sync-fast / async-slow branch
    │   ├── ask.py             # evidence packs + synth opt-in
    │   ├── search.py          # FTS/vector/hybrid retrieval
    │   └── status.py
    ├── pipeline/
    │   ├── runner.py          # orchestrates stages, persists state per stage
    │   ├── subprocess_util.py # SIGTERM → SIGKILL escalation
    │   └── stages/
    │       ├── fetch.py       # M1 — yt-dlp / local
    │       ├── normalize.py   # M1 — ffmpeg mezzanine + 16k mono wav
    │       ├── scene_detect.py    # M2 — PySceneDetect ContentDetector
    │       ├── transcribe.py      # M2 — faster-whisper int8 + word timestamps
    │       ├── frame_sample.py    # M3 — scene midpoints + stride sampling
    │       ├── ocr.py             # M3 — Tesseract PSM 6
    │       ├── embed.py           # M3 — SigLIP-base-patch16-256 (768-d)
    │       └── summarize.py       # deterministic per-scene summaries
    └── providers/
        ├── base.py            # ModelProvider ABC
        └── ollama.py          # request-time Ollama chat provider
```

---

## Configuration

All runtime config is via env vars (`MMRAG_*`) or `.env`. See
[.env.example](./.env.example) for the full list. The defaults are
sensible for Mac dev; Pi deployment overrides `MMRAG_OLLAMA_URL`,
`MMRAG_WORKER_CONCURRENCY`, and the data dir.

---

## Contributing

This is a personal project, but the design is documented end-to-end (see
`docs/architecture.md` and the original planning spec) and milestones are
tracked in `bd`. The non-negotiable rules:

1. **Stay MIT-clean.** Every new dependency must be MIT, Apache-2, BSD, or
   public domain. Flag anything else before adding it.
2. **Keep the MCP surface to four tools.** Add admin endpoints to REST, not
   to MCP.
3. **One sharp tool beats five mediocre ones.** Don't bloat the surface to
   wallpaper over a missing feature.
4. **Pipeline stages stay idempotent and resumable.** Crash recovery is a
   first-class requirement, not an afterthought. `jobs.stage` reports the
   active stage; `pipeline_state_json.last_completed_stage` controls resume so
   an interrupted active stage is retried.

---

## License

[MIT](./LICENSE) © 2026 Matthew Wagner
