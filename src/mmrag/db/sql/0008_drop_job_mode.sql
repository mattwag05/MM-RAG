-- jobs.mode ('standard' | 'shortform') was threaded from the ingest MCP tool
-- down to the frame sampler and never read: sampling density was identical for
-- both values. Deleted rather than wired (MM-RAG-cwe).
ALTER TABLE jobs DROP COLUMN mode;
