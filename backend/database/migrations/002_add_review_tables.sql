-- Migration 002: Add review and audit tables
-- Supports the Human Validation Loop (Phase 4)

CREATE TABLE IF NOT EXISTS pending_reviews (
    review_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT,
    confidence REAL,
    reason TEXT,
    grounded BOOLEAN,
    sensitive BOOLEAN DEFAULT FALSE,
    created_at TEXT DEFAULT (NOW()::TEXT),
    status TEXT DEFAULT 'pending'  -- pending | approved | corrected | rejected
);

CREATE TABLE IF NOT EXISTS approved_answers (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    session_id TEXT NOT NULL,
    answer TEXT NOT NULL,
    review_id TEXT REFERENCES pending_reviews(review_id),
    created_at TEXT DEFAULT (NOW()::TEXT),
    UNIQUE (query, session_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,     -- escalation | approval | correction | rejection
    review_id TEXT,
    session_id TEXT,
    query TEXT,
    reason TEXT,
    created_at TEXT DEFAULT (NOW()::TEXT)
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_pending_reviews_status ON pending_reviews(status);
CREATE INDEX IF NOT EXISTS idx_approved_answers_query ON approved_answers(query, session_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
