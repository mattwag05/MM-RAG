-- jobs.push_to_sbt drove a REST push into Social Bookmarks Triage, a private
-- reference consumer whose receiver was never locatable, so the path was never
-- validated end to end. Removed rather than carried: this repo ships as a
-- generic plugin and cannot hold app-specific integrations (MM-RAG-rrh).
ALTER TABLE jobs DROP COLUMN push_to_sbt;
