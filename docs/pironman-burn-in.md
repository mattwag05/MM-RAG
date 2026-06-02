# Pironman Burn-In Reference

Validated 2026-06-02 UTC / 2026-06-01 EDT.

## Live Surface

- Host: `pironman`
- Tailscale IP: `100.126.176.86`
- Discovery: `http://100.126.176.86:8766/.well-known/mcp-resource`
- MCP endpoint: `http://100.126.176.86:8766/mcp`
- Token location: `~/Projects/MM-RAG/.env` on Pironman
- Compose file: `~/Projects/MM-RAG/docker-compose.pi.yml`
- Long-running services: `mmrag-mcp`, `mmrag-worker`
- One-shot service: `mmrag-init`

Do not print `MMRAG_MCP_TOKEN` into docs, logs, or chat. Retrieve it over SSH
only into a local shell variable when a probe needs authenticated MCP.

## Burn-In Input

- Source: `https://youtu.be/8qQW4LTWgtc?si=QG2epuyKXFVjDBmk`
- Job: `da7c953e-a6db-45e9-bb1e-57237f144ebe`
- Asset: `b30d0b6f-a449-4837-a9ad-a9f19b6fde38`

## Passing Evidence

- MCP `list_tools` returned exactly `ask`, `ingest`, `search`, `status`.
- Ingest completed through fetch, normalize, scene detect, transcribe, frame
  sample, OCR, embed, and summarize.
- FTS search returned scoped hits for `Hestia`.
- Hybrid search returned scoped hits.
- `ask(question="What is this video about?", synthesize=false)` returned
  `answer=null` and non-empty evidence.
- Restarted `mmrag-mcp` and `mmrag-worker`; post-restart MCP `status`,
  `search`, and `ask` still passed.

## Persisted Counts

For asset `b30d0b6f-a449-4837-a9ad-a9f19b6fde38`:

- Scenes: 145
- Transcript segments: 143
- Frames: 354
- Content items: 642
- `vec_frames`: 354
- `vec_scenes`: 145
- `vec_transcript`: 143
- Graph totals after ingest: 1457 nodes, 7372 edges

## Resource Shape

During first-run model downloads and CPU inference:

- `mmrag-worker` peaked around 1.9 GiB RAM during SigLIP model load/embed.
- `mmrag-mcp` stayed small, roughly tens of MiB.
- Disk remained healthy: root filesystem about 19% used after burn-in.
- Host swap was already partly used by the broader homelab before the run, so
  watch total host pressure during future large ingests.

## Follow-Up Found And Fixed

During the burn-in, status appeared stale during long stages: logs showed
`stage.start stage=embed` while `status(job_id)` still reported `stage=ocr`.
This was tracked as `MM-RAG-x15` and fixed on `main` at `30225d7` by recording
the active stage at stage start while keeping
`pipeline_state_json.last_completed_stage` for safe resume.

Important deployment caveat: as of the doc update, the last verified running
Pironman checkout was `54b474f`; deploy `30225d7` or newer before expecting
the active-stage status behavior on the live service.
