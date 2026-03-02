-- Migration 003: Add evaluation and summary tables
-- Supports Phase 5 (Evaluation Framework) and the summarize_document tool

CREATE TABLE IF NOT EXISTS eval_metrics (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    query TEXT,
    latency_ms REAL,
    total_tokens INTEGER DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE,
    grounded BOOLEAN,
    escalated BOOLEAN DEFAULT FALSE,
    route TEXT,           -- trivial | single_agent | multi_agent
    eval_relevance REAL,
    eval_faithfulness REAL,
    eval_completeness REAL,
    eval_overall REAL,
    created_at TEXT DEFAULT (NOW()::TEXT)
);

CREATE TABLE IF NOT EXISTS document_summaries (
    document_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    created_at TEXT DEFAULT (NOW()::TEXT),
    updated_at TEXT DEFAULT (NOW()::TEXT)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_eval_metrics_session ON eval_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_metrics_created ON eval_metrics(created_at);
