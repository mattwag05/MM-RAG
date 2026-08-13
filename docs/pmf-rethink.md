# MM-RAG — PMF Rethink (v0.1.0 → v1.0)

> **Status:** Approved 2026-04-13. This doc supersedes the M3–M6 roadmap
> in `CLAUDE.md` and `docs/architecture.md` wherever they conflict.
> `~/.claude/plans/staged-strolling-gray.md` remains the canonical
> brainstorm rationale for M1–M2; this file is the canonical source for
> M3–M7 going forward.

## Why this exists

M1 (walking skeleton) and M2 (speech pipeline + FTS5) shipped on
2026-04-12. Before committing to M3–M6 as originally scoped, we audited
the roadmap against the actual product-market fit. The question we asked
was: *given that most agent harnesses (Claude Code, Claude API) already
handle images and PDFs natively, where does MM-RAG have unique leverage,
and does the committed roadmap put effort in the right places?*

Answer: yes to the core architecture, no to two specific scope choices
baked into M4, and yes to a reordering that promotes the multi-agent
transport story over the reference-consumer integration.

## Thesis

1. **The gap is long-form video and audio**, not images. Claude and
   Claude Code already ingest images and PDFs. Neither they nor the
   small local models powering self-hosted edge agents can take a
   YouTube video, podcast, lecture, or meeting recording as input — and
   even if they could, a 90-minute transcript plus frames would blow
   past the 32K context on `gemma4:e4b` running on the Pi.
2. **Retrieval-first matters because edge agents already have LLMs.**
   Edge agents run small local models with tight contexts. The
   value MM-RAG adds is a deterministic, persistent, queryable index —
   *not* another inference layer competing with the model the caller
   already has loaded.
3. **MCP is the plug-and-play surface.** Any MCP-capable caller (Claude
   Code on a laptop, edge agents on a tailnet, future harnesses) gets the
   index for free. This is the leverage point the four-tool contract
   (`ingest`/`ask`/`search`/`status`) was designed for.

## Findings from the audit

Audited: canonical plan, `docs/architecture.md`, the beads backlog,
`src/mmrag/models/mcp_io.py`, `src/mmrag/models/job.py`, and the open
M3/M4/M5/M6 issue bodies.

### What's aligned (keep)

- Retrieval-first pipeline with deterministic indexing (onnx-asr,
  PySceneDetect, FTS5). **Shipped.**
- Four-tool MCP surface (`ingest`/`ask`/`search`/`status`) with a shared
  handler layer used by both MCP and REST. **Shipped.**
- Content-hash identity and idempotent stage resume. Critical for a
  shared index across multiple agents. **Shipped.**
- Pi-ready by construction: SQLite WAL, SIGTERM→SIGKILL subprocess
  escalation, setuptools backend, venv gotchas handled. **Shipped.**

### What's misaligned or over-scoped

**1. `AskOutput.answer: str` is required** (`src/mmrag/models/mcp_io.py:61`).
The MCP contract *forces* MM-RAG to do reasoning. `AskInput.model` even
hardcodes `gemma4:e4b`/`gemma4:e2b` as the only legal values
(`mcp_io.py:43`). This is the biggest contradiction with the thesis:
edge agents already have their own LLMs. Bundling Gemma 4 inference
doubles the Pi footprint (roughly 3 GB memory plus 10 GB model weights)
and buys nothing the caller can't do itself.

**2. `SearchOutput` is too thin to serve as an evidence-pack path**
(`mcp_io.py:80–94`). Only `asset_id`, `start_s`, `end_s`, `score`, and
`snippet`. No summary, no OCR text, no transcript excerpt, no
`scene_id`. So an agent that wants "give me evidence and I'll reason"
has no good path: `ask` forces synthesis, `search` is too thin to feed
another LLM.

**3. M4 (`MM-RAG-4oz`) bakes Gemma 4 into `ask`** and adds per-scene
summaries as stage 8. Per-scene summaries are fine as a deterministic
indexing step (cheap local distillation, stored in the index). The
reasoning layer in `ask` is not — it ties the Pi footprint to a
4B-parameter model the caller is already running elsewhere.

**4. `MM-RAG-kb0` (streamable-HTTP MCP transport) is P3 but should be
P1.** The thesis explicitly says "shared index queried by multiple edge
agents via MCP." Stdio-only transport means each caller ships its own
MM-RAG silo, which is the opposite of that. Tailnet-hosted
streamable-HTTP is how several agents on one private network hit the same
index from one host. Currently ranked *below* the SBT integration,
which is inverted.

**5. `MM-RAG-456` (SBT integration) is scope-creep at its current
priority.** SBT is a reference consumer proving the tool works, not a
core thesis feature. Shipping it before a working Pi deploy and shared
MCP transport means proving the wrong thing first.

## Decisions

- **`ask` goes evidence-only by default; synthesis is opt-in.**
  `AskOutput.answer` becomes `str | None`. `AskInput` gains
  `synthesize: bool = False`. The `model: Literal[...]` lock-in is
  replaced with `model: str | None = None`, ignored unless
  `synthesize=True`. `OllamaProvider` and the answer-synthesis code move
  into an optional `[reasoning]` pyproject extra. Core install has no
  Ollama dependency.
- **M3 ships as committed.** SigLIP + Tesseract OCR + frame sampling +
  sqlite-vec + hybrid RRF retrieval all stay on the critical path. True
  visual-similarity queries are in scope for v1. We considered deferring
  SigLIP but the visual grounding is worth the ~200 MB base model on
  the Pi.
- **Deployment is a single shared tailnet service.** One MM-RAG
  instance on a self-hosted server, streamable-HTTP MCP transport, all edge
  agents hit the same index. `MM-RAG-kb0` promotes to P1 and is a hard
  prerequisite for the "Pi deploy is actually useful" milestone. v1 is
  explicitly *single-tenant*: no caller IDs, no per-caller quotas, no
  asset-visibility scoping. Auth on the streamable-HTTP endpoint is a
  shared token in env plus a private-network-only bind.
  This is now live as of 2026-06-01: a self-hosted server hosts MM-RAG behind
  the shared bearer token, with `.env` on the server holding the token.
  Production burn-in passed on 2026-06-02 UTC / 2026-06-01 EDT with real
  YouTube ingest, search, evidence-first ask, restart, and persistence
  checks, plus verified MCP client connectivity to the same four-tool
  surface.
- **`search` becomes a first-class evidence path.** `SearchOutput`
  enriches from thin hits to full `Evidence` objects (same shape as
  `ask` evidence: `summary`, `ocr_snippet`, `transcript_snippet`,
  `scene_id`). Agents that prefer ranked hits to a question-shaped query
  get the same rich data the `ask` path would return.

## New milestone ordering

| New | Beads ID      | Name                                                    | Priority | Change from prior |
|-----|---------------|---------------------------------------------------------|----------|-------------------|
| M3  | `MM-RAG-eym`  | Visual pipeline (SigLIP + Tesseract + sqlite-vec)        | P1       | no change         |
| M4  | `MM-RAG-4oz`  | Evidence packs (synth opt-in)                            | P1       | rescoped          |
| M5  | `MM-RAG-kb0`  | Streamable-HTTP MCP transport (tailnet-hosted service)   | P1       | P3 → P1           |
| M6  | `MM-RAG-xr0`  | Raspberry Pi deploy (lighter footprint)                 | P2       | scope trimmed     |
| M7  | `MM-RAG-456`  | SBT reference integration (**DROPPED**, see below)        | —        | removed 2026-08-13 |
| post-v1 | (new)     | Bundled reasoning `[reasoning]` pyproject extra           | P3       | new               |

Dependency chain: M3 → M4 → M5 → M6. (M7 is dropped, see below.)

### M3 — Visual pipeline (`MM-RAG-eym`, unchanged)

Ships as originally described: frame sampling at scene midpoints,
Tesseract OCR, SigLIP-base-patch16-256 embeddings via `open_clip`, new
`scenes`/`frames`/`vec_scenes`/`vec_frames`/`vec_transcript`/`fts_scenes`
tables, and hybrid retrieval across FTS + vectors with reciprocal rank
fusion. No changes from the committed scope.

### M4 — Evidence packs, synthesis opt-in (`MM-RAG-4oz`, rescoped)

Concrete changes from the original M4 scope:

- `AskOutput.answer: str` → `str | None`. `evidence` and `confidence`
  stay. The handler returns `answer=None` unless `synthesize=True`.
- `AskInput` gains `synthesize: bool = False`. Default is evidence-only.
- `AskInput.model: Literal["gemma4:e4b", "gemma4:e2b"]` →
  `model: str | None = None`, ignored unless `synthesize=True`.
- `SearchOutput.hits: list[SearchHit]` enriches to
  `hits: list[Evidence]` (or a new `SearchHit` superset with the full
  evidence fields). Decide at implementation time which name preserves
  the cleanest contract.
- Stage 8 (`summarize`) stays as a deterministic indexing artifact:
  per-scene short-text distillation run once at ingest from transcript
  and OCR text, stored in `scenes.summary`. This is *not* the same as
  `ask` answer synthesis — it's an index column, not request-time
  inference.
- `OllamaProvider` and the answer-synthesis code path move into an
  optional `[reasoning]` pyproject extra. Core install has no Ollama
  hard dependency.

### M5 — Streamable-HTTP MCP transport (`MM-RAG-kb0`, P3 → P1)

This is how several agents query one MM-RAG instance from one host. Without it, the v1 deployment is a stdio silo
per agent — which contradicts the whole thesis. The handler layer is
already transport-agnostic (FastAPI REST and FastMCP stdio share it);
adding streamable-HTTP is a second transport binding, not a rewrite.
Auth is a shared token in env, bind is Tailscale-only, and the endpoint
advertises a `.well-known/mcp-resource` for client discovery.

### M6 — Raspberry Pi deploy (`MM-RAG-xr0`, lighter footprint)

Footprint with the rescoped M4: ffmpeg + onnx-asr Parakeet TDT 0.6b v3
(int8, ~640 MB + 2 MB Silero VAD) + Tesseract +
SigLIP-base-patch16-256 (~200 MB) + sqlite-vec + SQLite. The ASR swap
(MM-RAG-k48) traded ~490 MB of disk for 12-14% → ~6.3% WER and 25
languages; a Pi deploy that cannot spare it can set
`MMRAG_TRANSCRIBE_MODEL` to a smaller onnx-asr model.
**No Ollama or Gemma 4 hard dependency** — that moves to the optional
`[reasoning]` extra. Estimated runtime: ~1.5 GB RAM, ~1.5 GB disk for
models. Comfortable on a Raspberry Pi 5–class server. Deploy
target is a single tailnet-hosted service, not per-agent sidecars, and
this milestone blocks on M5 shipping the streamable-HTTP transport.

### M7 — SBT reference integration (`MM-RAG-456`, P2 → P3)

Still valuable as a reference consumer proving the tool is useful to a
real application. Scope unchanged from the original issue, priority
drops to P3 behind the Pi deploy. Marks the transition from "tool
works" to "tool is integrated somewhere real."

**DECIDED 2026-08-13 (MM-RAG-rrh): M7 is dropped. Do not re-litigate.**
The client side shipped (`push_to_sbt=true`, `MMRAG_SBT_URL`,
`src/mmrag/sbt_client.py`, tests), but the receiver was never locatable — the
2026-06-04 audit could not run a single end-to-end smoke, and nothing changed
in the two months after. Two reasons to remove rather than keep waiting:
the path has no evidence it ever worked, and this repo is destined to ship as
a generic public plugin, which cannot carry an integration with one private
application. Removed: `sbt_client.py`, its tests, `MMRAG_SBT_URL`, the
`push_to_sbt` parameter on `ingest`, and the push call in the runner.
Migration `0010_drop_job_push_to_sbt.sql` drops the column.

If a reference consumer is wanted later, build it against the public MCP
surface as a separate repo. Nothing in MM-RAG needs to know about it.

## v1 single-tenant assumption (explicit)

v1 is single-tenant by design:

- No caller ID in the MCP or REST schemas.
- No per-caller quotas.
- No asset-visibility scoping.
- Auth is a shared bearer token in env; transport is Tailscale-only.

This is a deliberate scope choice, not an oversight. Multi-tenant
isolation belongs to post-v1 work only if a concrete caller actually
needs it. Until then, nobody should ship a half-multi-tenant feature.

## Critical files

- `src/mmrag/models/mcp_io.py` — M4 schema changes land here (`AskOutput`
  line 58, `AskInput` line 36, `SearchOutput` line 91, `SearchHit` line 80)
- `src/mmrag/models/job.py` — stage order unchanged; stage 8
  implementation changes in M4
- `src/mmrag/handlers/` — `ask.py` and `search.py` change under M4
- `src/mmrag/providers/` — `OllamaProvider` moves behind `[reasoning]`
  extra under M4
- `pyproject.toml` — `[project.optional-dependencies]` gains
  `reasoning = [...]` under M4
- `docs/architecture.md` — updated in this pass to reflect the new
  milestone scope
- `CLAUDE.md` — `§ Status` section updated in this pass

## What changes in this pass (and what doesn't)

This PMF rethink writes docs and updates beads issues. No production
code changes, no schema changes, no test changes. The actual M4 schema
refactor happens in a separate session against the rescoped
`MM-RAG-4oz` issue, once this plan is the source of truth.
