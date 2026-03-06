-- Enable pgvector extension and create session-scoped embedding table.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_session
ON chunk_embeddings(session_id);

-- Optional ANN index for cosine distance search.
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_embedding_cosine
ON chunk_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
