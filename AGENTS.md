# Agent Instructions

**See [CLAUDE.md](./CLAUDE.md).** It is the single source of truth for this
project — architecture, build/test/lint commands, conventions, gotchas, and the
beads (`bd`) issue-tracking workflow (including `bd dolt push` at session close).

This file is intentionally a thin pointer and carries no content of its own, so
agents that read both never get duplicate or conflicting instructions.

> Note: `bd setup` / `bd init` may regenerate marker-wrapped `BEADS INTEGRATION`
> blocks here. If they reappear, this file was meant to stay a pointer — the
> beads guidance already lives in CLAUDE.md.
