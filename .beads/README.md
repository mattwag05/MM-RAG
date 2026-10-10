# Native Beads tracker

Project issues use Dolt `main` in `refs/dolt/data` on this project's canonical Git origin. Code branches do not select issue branches. Normal Git worktrees share the main checkout's embedded database; independent clones have independent replicas.

The October 10, 2026 adoption baseline is 42 issues, schema 66, project UUID `0ac3e4a0-96cc-4131-9f20-03d57aa109bc`. Later issue counts may change. Beads 1.3.1 and standalone Dolt 2.4.2 qualified this setup. Keep auto-push disabled.

For a new clone, check `git ls-remote origin refs/dolt/data`, then run `bd bootstrap --non-interactive`. Verify `bd where`, the project UUID, native remote and canonical records before claiming work. An absent ref or empty fallback is not a successful adoption. Preserve the project's existing Git hooks.

Obtain an acknowledged publishing turn across the actual writers, sync before selecting or claiming work, inspect conflicts, sync updates, verify the remote, then hand off. Keep the turn while recovering from failure. This is cooperative serialization; it does not provide a distributed lock or atomic cross-machine claims. Backups provide recovery separately.
