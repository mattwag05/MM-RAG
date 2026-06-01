-- Prevent multiple processes from executing the same queued job at once.
--
-- The MCP handler and background worker can both observe a newly queued job.
-- A runner lease lets exactly one process own execution while still allowing
-- stale running jobs to be resumed after a crash.

ALTER TABLE jobs ADD COLUMN runner_id TEXT;
ALTER TABLE jobs ADD COLUMN runner_heartbeat_at TEXT;

CREATE INDEX IF NOT EXISTS idx_jobs_runner_heartbeat
    ON jobs(status, runner_heartbeat_at);
