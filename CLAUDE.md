# CLAUDE.md — MM-RAG

> Edge-optimized multimodal ingestion tool exposed as an MCP server.
> Tested/deployed on Python 3.13, MIT-licensed, currently at v0.1.0
> (M6 Pi deploy path shipped).

## What it is

`mmrag` ingests video/audio/image content (URLs via yt-dlp + local files),
normalizes it with ffmpeg, and produces a transcript + scene map + OCR +
embeddings into a single SQLite + sqlite-vec store. Agents query it via
four MCP tools: `ingest`, `ask`, `search`, `status`. Reasoning is delegated
to Ollama-hosted Gemma 4 (`gemma4:e4b` primary, `gemma4:e2b` fallback) over
retrieved evidence packs — never over the raw video.

Mac is the dev home. Raspberry Pi is the deployment floor.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

> **Beads sync — `bd dolt push` is separate from `git push`.** Issues live in a
> local Dolt DB and travel on the `refs/dolt/data` git ref, which a plain
> `git clone` does NOT fetch. Always `bd dolt push` at session close (not just
> `git push`); hydrate a fresh clone with `bd dolt pull`. The project's original
> issue history was lost this way before the M5 machine transition.

## Status (v0.1.0 — M6 Pi deploy path)

What's wired end-to-end today:
- `uv` project with broad Python `>=3.11,<3.14` packaging; dev/deploy
  default is Python 3.13. **setuptools** backend (NOT hatchling — see "Gotchas")
- FastMCP stdio server with all 4 tools (`mmrag serve-mcp`)
- FastMCP Streamable HTTP server with the same 4 tools (`mmrag serve-mcp-http`)
  for shared tailnet access; non-loopback binds require `MMRAG_MCP_TOKEN`.
- FastAPI REST mirror on `:8765` (`mmrag serve-api`)
- Background worker that drains the job queue (`mmrag worker`)
- SQLite WAL store, migration runner, M1–M4 foundation schema (`assets`, `jobs`,
  `scenes`, `frames`, `transcript_segments`, `fts_transcript`, `fts_scenes`,
  `vec_frames`, `vec_scenes`, `vec_transcript`, `content_items`)
- Pipeline stages 1–7 live: fetch (yt-dlp / local file) → ffmpeg normalize →
  PySceneDetect `ContentDetector` → faster-whisper `tiny.en` int8 →
  frame sampling (scene midpoints + 2s stride on long scenes) →
  Tesseract OCR (PSM 6, 10s timeout) → SigLIP-base-patch16-256 embed (768-d, L2-norm)
- Stage 8 (summarize) deterministically distills per-scene transcript/OCR
  text into `scenes.summary` and rewrites `content_items` video-segment text
- Runner persists scenes + frames + transcript segments incrementally after each
  stage (idempotent via UNIQUE keys, so re-ingest is a no-op)
- `search` tool runs hybrid RRF over FTS transcript / FTS scenes / vec frames /
  vec transcript, scoped by optional `asset_id` and `top_k`, snippet-highlighted.
  sqlite-vec asset scoping uses metadata-column prefilters so `k=` is scoped
  before nearest-neighbor selection.
- `ask` returns evidence packs by default (`answer=None`) and only calls the
  configured Ollama/Gemma backend when `synthesize=True`.
- `content_items` is a persisted projection over scenes, transcript segments,
  and frames; it is the foundation for document ingestion and graph retrieval.
- Pydantic schema contract tests + pipeline unit tests + end-to-end MCP
  ingest → search round-trip using a TTS-generated speech fixture
  (73/73 passing on macOS with `say`; integration tests auto-skip if no
  TTS tool is available)
- Subprocess wrapper with SIGTERM → SIGKILL escalation (Pippin-pattern)
- `ModelProvider` ABC with a request-time `OllamaProvider` implementation
- Dockerfile + Compose paths for Mac REST dev and Pi/tailnet MCP deployment.
  The Pi path runs MCP HTTP + worker as separate services, includes M3 visual
  deps, and does not bundle Ollama/Gemma.

Shipped:
- **M3** — visual pipeline (frame sampling + Tesseract OCR + SigLIP-base-patch16-256 embeddings + sqlite-vec hybrid RRF over FTS transcript / FTS scenes / vec frames / vec transcript). Renamed `shots` → `scenes` across the schema. Optional `m3-visual` extra (torch, open-clip-torch, Pillow, numpy, transformers) keeps heavyweight ML packages out of the core install; sqlite-vec is core because the shipped schema requires vec0 tables.
- **M4** — evidence packs and synth opt-in (`ask` returns evidence by default; `answer` is `str | None`; request-time Ollama synthesis is behind `synthesize=True`). Also added `content_items` as the unified projection foundation and deterministic per-scene summaries.
- **M5** — streamable-HTTP MCP transport for one shared tailnet service (`MMRAG_MCP_HOST` / `MMRAG_MCP_PORT` / `MMRAG_MCP_PATH`, protected by `MMRAG_MCP_TOKEN` when not loopback). Discovery metadata is served at `/.well-known/mcp-resource`.
- **M6** — Pi / Pironman deploy path: MCP HTTP + worker Compose stack,
  token-required tailnet bind, M3 visual runtime deps, no bundled Gemma/Ollama.

Open milestones (see `bd ready` and `docs/pmf-rethink.md` for full rationale):
- **M7** — Social Bookmarks Triage REST integration (reference consumer, not core)
- **MM-RAG 2.x follow-ups** — document ingestion, graph retrieval, and optional vector backends (tracked in Beads)

v1 is a **single-tenant tailnet service**: one MM-RAG instance, shared bearer token in env, Tailscale-only bind. No caller IDs, no per-caller quotas, no asset-visibility scoping. Multi-tenant auth is post-v1. See `docs/pmf-rethink.md` for the thesis and audit behind this ordering.

## Build & Test

**Always go through `make`** — never `uv` directly, because the Makefile
pins the venv outside the iCloud sync path (see Gotchas).

```bash
make sync-dev                             # install runtime + dev deps into .venv.nosync/
make sync-m3                              # runtime + dev + M3 visual pipeline deps
make init-db                              # create the SQLite DB at MMRAG_DATA_DIR
make serve-api                            # FastAPI on :8765
make serve-mcp                            # FastMCP over stdio
make serve-mcp-http                       # FastMCP Streamable HTTP on :8766
make worker                               # drain the job queue
make test                                 # full test suite (73 tests)
make lint                                 # ruff check  src tests (correctness/style gate)
make format                               # ruff format src tests (separate formatter gate)
make docker-build                         # build the multi-arch-capable local image
make docker-up                            # Mac REST dev compose path (:8765)
make docker-pi-config                     # validate Pi/tailnet compose config
make docker-pi-up                         # run Pi/tailnet MCP + worker stack (:8766)
make docker-pi-down                       # stop Pi/tailnet stack
```

In a **sandboxed shell** (e.g. Claude Code's Bash), if OCR tests fail because
the `tesseract` subprocess can't read the sandbox `TMPDIR`, run
`TMPDIR=~/.cache/mmrag-pytest-tmp make test` (see Gotchas).

## Where things live

| Area | Path |
|---|---|
| MCP tool definitions | `src/mmrag/mcp_server.py` (FastMCP app factory for stdio + Streamable HTTP) |
| REST mirror | `src/mmrag/api.py` |
| Tool handlers (shared by MCP + REST) | `src/mmrag/handlers/` |
| Pipeline runner + stages | `src/mmrag/pipeline/runner.py`, `src/mmrag/pipeline/stages/` |
| DB schema | `src/mmrag/db/sql/0001_m1_init.sql`, `0002_m2_speech.sql`, `0003_m3_visual.sql` |
| Pydantic I/O models | `src/mmrag/models/mcp_io.py` |
| Settings (env-var driven) | `src/mmrag/config.py` |
| Tests | `tests/test_contract.py`, `tests/test_pipeline_*.py` |
| Design spec | `docs/architecture.md` (canonical: `~/.claude/plans/staged-strolling-gray.md`) |

## Architecture overview

Four MCP tools (`ingest`, `ask`, `search`, `status`) sit in front of a
shared handler layer. The handlers either run the pipeline inline (for the
sync-if-fast `ingest` path) or delegate to a background worker that drains
the job queue. Each pipeline stage takes a `pipeline_state` dict, returns a
patch, and the runner persists state to the `jobs` row after every stage —
so a worker crash mid-job is recoverable from the recorded `stage`.

Identity flows through `content_hash` (SHA-256 of the canonical mezzanine
file). Re-ingesting the same video under a different URL is a no-op.

Retrieval comes first, reasoning second: the index is built deterministically
(faster-whisper for transcription, PySceneDetect for scenes, SigLIP for
embeddings, Tesseract for OCR), and Gemma 4 only ever sees the top-k
retrieved evidence — never raw frames or full transcripts. This is the only
way the 32K context window on `gemma4:e4b` doesn't bite us.

See `docs/architecture.md` for the diagram and the full data flow.

## Conventions & patterns

- **Pipeline stages return *patches*, not mutated state.** The runner merges.
- **Asset identity = content hash, not URL or job id.** Idempotency comes for free.
- **Errors are typed.** `FetchError(kind, message)`, `NormalizeError(kind, message)`, etc. The runner records `error_kind` separately so the UI/agents can branch on it.
- **Subprocess timeouts use the pipeline subprocess wrapper**, never `subprocess.run` directly — the wrapper does SIGTERM → SIGKILL escalation.
- **Per-stage code lives in its own file** under `pipeline/stages/`. The runner imports them by name. New stages get added to `M1_STAGE_ORDER` in `models/job.py`.
- **MCP and REST share handlers.** Don't duplicate logic — both surfaces call `handlers/{ingest,ask,search,status}.py`.
- **Pydantic with `extra="forbid"` on inputs.** Unknown fields are rejected at the MCP/REST boundary.

## Gotchas (paid in blood during M1)

- **The venv MUST live at `.venv.nosync/`, not `.venv/`.** The project root
  is on the macOS Desktop, which iCloud Drive syncs. iCloud sets the macOS
  `UF_HIDDEN` flag on `.pth` files inside synced directories (and creates
  ` 2`-suffixed duplicates when you re-sync). Python 3.13's `site.py` then
  silently skips any `.pth` file with the hidden flag, and your editable
  install becomes invisible. The `.nosync` suffix tells iCloud to leave the
  directory alone. The `Makefile` pins `UV_PROJECT_ENVIRONMENT=.venv.nosync`
  so this happens automatically — **always go through `make`**, never `uv`
  directly. If you have to use `uv` directly, prefix it with
  `UV_PROJECT_ENVIRONMENT=.venv.nosync`.
- **Don't use hatchling as the build backend on Python 3.13.** Hatchling's
  default editable install creates `_<name>.pth`. Even outside iCloud, that
  filename can also trip Python 3.13's hidden-file check in `site.py`.
  `mmrag` uses `setuptools` for exactly this reason — the pyproject.toml
  comment explains it. Don't switch back without a permanent fix.
- **`executescript()` implicitly commits.** SQLite's `executescript` calls
  `commit` before running, so wrapping it in a manual `BEGIN/COMMIT`
  context manager fails with "cannot commit - no transaction is active."
  `db/migrations.py` runs `executescript` directly in autocommit mode.
- **Asset identity is content-hash, not URL.** Re-ingesting the same file
  under a different URL produces the same `assets` row. The runner's
  `_upsert_asset` reconciles `asset_id` if a hash collision is found.
- **`ingest` is sync-if-fast.** It blocks for up to `wait_ms` ms (default
  30000) and returns either a finished result or `{status: in_progress, job_id}`
  for polling. The worker keeps draining the same job in the background even
  if the request returns early.
- **`mmrag.logging` shadows stdlib `logging`** when imported via
  `from mmrag.logging import ...`. That's fine internally but be careful in
  test/repro snippets that mix the two.
- **Python module name is `mmrag`, project directory is `MM-RAG`.** Hyphens
  aren't valid in Python module names. The `uv` project name in pyproject is
  also `mmrag`.
- **`asyncio.create_subprocess_exec` substring `exec` trips a global Write
  hook.** The `pipeline/subprocess_util.py` file uses a `getattr` indirection
  to dodge it. Don't "fix" the indirection back to a literal name without
  also updating the hook.
- **Tesseract is a required non-Python dep for ingest once M3 ships.** Install
  with `brew install tesseract` (macOS) or `apt install tesseract-ocr`
  (Debian/Pi). The `ocr` stage fails fast with `OCRError(kind="binary_missing")`
  and a clear install hint if it's missing. The OCR stage shells out through
  `pipeline.subprocess_util.run`, so the 10s per-frame timeout terminates a
  slow Tesseract child with SIGTERM/SIGKILL escalation.
- **SigLIP raw cosine scores are modest.** The M3 acceptance test uses a
  natural apple/table fixture and asserts `> 0.14` on the `vec_frames`
  cross-modal hit. That is intentionally below old CLIP-style `> 0.5`
  expectations: `timm/ViT-B-16-SigLIP-256` returns lower normalized dot
  products, especially after video encode/frame extraction.
- **Test isolation uses `reset_settings_for_tests`, NOT `monkeypatch.setenv +
  cache_clear`.** `get_settings` is a module-level singleton, not `@lru_cache`.
  Correct pattern: `reset_settings_for_tests(Settings(data_dir=tmp_path))` at
  the start of the test, `reset_settings_for_tests(Settings())` in a
  `try/finally`. See `tests/conftest.py`'s `isolated_data_dir` fixture for the
  canonical pattern.
- **Subprocess wrapper is `run`, not `run_subprocess`.** Import with
  `from mmrag.pipeline.subprocess_util import run` and call with
  `await run([...], timeout_s=...)`. The kwarg is `timeout_s`, not `timeout`.
- **PyTorch inference mode: use `model.train(False)`.** A project-wide
  security hook blocks the literal PyTorch inference-mode method name
  (the one spelled e-v-a-l, with parens). `train(False)` is functionally
  equivalent. Also: disable autograd per-call via `torch.no_grad()` context
  managers, not a module-level `torch.set_grad_enabled(False)` — and
  `.detach()` before `.numpy()`.
- **sqlite-vec blob format is explicit little-endian.** Always
  `struct.pack(f"<{len(v)}f", *v)` — native order silently corrupts vectors
  on big-endian hosts. The runner's `_pack_vec` and the search handler's
  `_pack` both use the `<` prefix.
- **sqlite-vec `k=` is a GLOBAL pre-filter unless `asset_id` is a vec0 metadata
  constraint.** `vec_frames`, `vec_scenes`, and `vec_transcript` declare
  `asset_id TEXT` inside vec0 (not `+asset_id TEXT`), and scoped KNN SQL puts
  `asset_id = ?` before `embedding MATCH ?`. Plain JOIN filters after MATCH
  reintroduce the under-delivery bug fixed in `0004_vec_asset_filters.sql`.
- **open_clip's SigLIP HFTokenizer pulls in `transformers` at runtime.**
  `transformers>=4.40` is already in the `m3-visual` extra. Don't remove it
  thinking open_clip is self-contained — it isn't for the SigLIP variants.
- **Runner's `__`-prefix state keys are carry-only, not persisted.**
  `_strip_internal` filters them out before `pipeline_state_json` serialization.
  Use them for in-memory maps that cross stage boundaries inside one run
  (e.g., `__frame_id_map`, `__scene_id_by_idx`). Don't rely on them surviving
  a worker restart — add a DB-recompute fallback (see
  `_frame_id_map_from_db` for the pattern).
- **`uv run` re-syncs to default deps — it strips the m3-visual extras.**
  uv >=0.5 auto-syncs the environment to the project's *default* dependencies
  on every `uv run`, silently uninstalling whatever `make sync-m3` added
  (torch, Pillow, open-clip, transformers, ...). sqlite-vec is now core
  because the schema requires vec0 tables, but a bare `uv run pytest` can
  still fail collection (`No module named 'PIL'`) or visual-stage imports.
  The `Makefile`'s `test` target pins `--extra dev --extra m3-visual` on the
  `uv run` itself for this reason — **always go through `make test`**, and if
  you run `uv run` directly, pass both extras (and
  `UV_PROJECT_ENVIRONMENT=.venv.nosync`).
- **OCR tests need a subprocess-readable temp path.** Tesseract/Leptonica can
  fail if a sandbox redirects `TMPDIR` to a path the child process cannot read
  (e.g. Claude Code's `/tmp/claude-501/...`). The code is fine — point
  `TMPDIR` at a normal path: `TMPDIR=~/.cache/mmrag-pytest-tmp make test`, or
  just run the suite in a normal terminal.

## Reused patterns from other projects

The M1 scaffold deliberately borrowed structural patterns. When extending,
look at these references rather than reinventing:

| Pattern | Source |
|---|---|
| Dedup-by-content-hash | `pippin/pippin/MailAIBridge/EmbeddingStore.swift` |
| Phase-1 filter / phase-2 batch | `pippin/pippin/Commands/MailAICommand.swift` |
| Subprocess timeout + SIGTERM → SIGKILL | `pippin/pippin/MailBridge/MailBridgeRunner.swift` |
| ModelProvider abstraction | `SwiftClaw/Sources/SwiftClawCore/Backend/ModelBackend.swift` |
| Tool registry pattern | `SwiftClaw/Sources/SwiftClawCore/Tool/ToolRegistry.swift` |
| Ollama httpx streaming | `rag-quest/rag_quest/llm/ollama_provider.py` |
| FIFO + systemd Pi wrapper | `Personal Agent/deploy/pi-agent-wrapper.sh` (used in M6) |
| systemd user service | `Personal Agent/deploy/pi-agent.service` (used in M6) |

## Integration target: Social Bookmarks Triage

M7 will push enrichment payloads to SBT via REST:

- SBT lives at `~/Desktop/Projects/Social Bookmarks Triage` (Next.js + Prisma + SQLite, keyed on `postId`)
- Existing pattern to follow: `app/api/import/url/route.ts`
- Schema migration on the SBT side: extend `MediaItem` with `mmrag_asset_id String? @index` and `transcriptText String?`; extend `Bookmark` with `mmragSummary String?` and `mmragTopTags Json?`
- `transcriptText` joins SBT's existing FTS so spoken content becomes searchable inside SBT's UI
- Idempotency via matching SBT's base64url-of-URL-segments `postId` hash

## Bug-fixing workflow (per global CLAUDE.md)

When fixing a bug:
1. Write a failing test that reproduces it first
2. Dispatch subagents to fix it; their success criterion is the test going green
3. Only accept the fix the test proves

This applies to mmrag bugs too.

## Hard constraints (don't break)

- **License: MIT.** Every dependency must be MIT, Apache-2, BSD, or public domain. Flag anything else before adding it. Gemma weights and ffmpeg are user-supplied and explicitly not bundled.
- **Four MCP tools.** `ingest`, `ask`, `search`, `status`. Add admin operations to REST, not MCP.
- **Retrieval-first.** Gemma 4 only ever sees retrieved evidence packs, never raw videos.
- **Idempotent + resumable pipeline stages.** Crash recovery is a first-class requirement.
- **Pi-ready by construction.** Choices that look reasonable on Mac but fall over on ARM are not acceptable.
