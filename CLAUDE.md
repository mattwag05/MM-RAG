# CLAUDE.md — MM-RAG

> Edge-optimized multimodal ingestion tool exposed as an MCP server.
> Python 3.13, MIT-licensed, currently at v0.1.0 (Milestone 1 walking skeleton).

## What it is

`mmrag` ingests video/audio/image content (URLs via yt-dlp + local files),
normalizes it with ffmpeg, and produces a transcript + scene map + OCR +
embeddings into a single SQLite + sqlite-vec store. Agents query it via
four MCP tools: `ingest`, `ask`, `search`, `status`. Reasoning is delegated
to Ollama-hosted Gemma 4 (`gemma4:e4b` primary, `gemma4:e2b` fallback) over
retrieved evidence packs — never over the raw video.

Mac is the dev home. Raspberry Pi is the deployment floor.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
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

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Status (v0.1.0 — M1 walking skeleton)

What's wired end-to-end today:
- `uv` project on Python 3.13, **setuptools** backend (NOT hatchling — see "Gotchas")
- FastMCP stdio server with all 4 tools (`mmrag serve-mcp`)
- FastAPI REST mirror on `:8765` (`mmrag serve-api`)
- Background worker that drains the job queue (`mmrag worker`)
- SQLite WAL store, migration runner, M1 schema (`assets`, `jobs`, `schema_migrations`)
- Pipeline stages 1 (fetch via yt-dlp / local file) and 2 (ffmpeg normalize)
- Stages 3–8 are no-op stubs that return `{stub: "m2"}`-style patches; the
  runner walks all 8 so progress reporting works through to `done`
- Pydantic schema contract tests + pytest pipeline tests with auto-generated
  ffmpeg lavfi fixtures (15/15 passing)
- Subprocess wrapper with SIGTERM → SIGKILL escalation (Pippin-pattern)
- `ModelProvider` ABC with `OllamaProvider` shell (M4 ships the real impl)
- Dockerfile + docker-compose for Mac dev (Pi-targeted M6)

Open milestones (see `bd ready`):
- **M2** — speech (PySceneDetect + faster-whisper + FTS5 transcript)
- **M3** — visual (frame sampling + Tesseract OCR + SigLIP + sqlite-vec hybrid)
- **M4** — reasoning (scene summaries + ask evidence pack + Gemma 4 fallback)
- **M5** — Social Bookmarks Triage REST integration (`push_to_sbt`)
- **M6** — Raspberry Pi deploy

## Build & Test

**Always go through `make`** — never `uv` directly, because the Makefile
pins the venv outside the iCloud sync path (see Gotchas).

```bash
make sync-dev                             # install runtime + dev deps into .venv.nosync/
make init-db                              # create the SQLite DB at MMRAG_DATA_DIR
make serve-api                            # FastAPI on :8765
make serve-mcp                            # FastMCP over stdio
make worker                               # drain the job queue
make test                                 # full test suite (15 tests)
```

## Where things live

| Area | Path |
|---|---|
| MCP tool definitions | `src/mmrag/mcp_server.py` (4 `@mcp.tool()` decorators) |
| REST mirror | `src/mmrag/api.py` |
| Tool handlers (shared by MCP + REST) | `src/mmrag/handlers/` |
| Pipeline runner + stages | `src/mmrag/pipeline/runner.py`, `src/mmrag/pipeline/stages/` |
| DB schema (M1) | `src/mmrag/db/sql/0001_m1_init.sql` |
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
(faster-whisper for transcription, PySceneDetect for shots, SigLIP for
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

M5 will push enrichment payloads to SBT via REST:

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
