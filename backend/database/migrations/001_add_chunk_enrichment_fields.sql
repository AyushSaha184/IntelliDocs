-- Migration 001: Add chunk enrichment fields
-- These columns support async metadata enrichment (summary, keywords, hypothetical questions)

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS summary TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS keywords TEXT;               -- JSON array
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS hypothetical_questions TEXT;  -- JSON array
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS enrichment_status TEXT DEFAULT 'pending';

-- Index for background enrichment worker to find pending chunks efficiently
CREATE INDEX IF NOT EXISTS idx_chunk_enrichment_status ON chunks(enrichment_status);
