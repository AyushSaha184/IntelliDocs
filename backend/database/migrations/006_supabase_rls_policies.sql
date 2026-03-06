-- Supabase RLS policies for tenant isolation (optional but recommended).
-- Run this only if your clients query these tables directly via Supabase APIs.

ALTER TABLE IF EXISTS sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS chunk_embeddings ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS sessions_owner_select ON sessions;
CREATE POLICY sessions_owner_select
ON sessions
FOR SELECT
USING (user_id = auth.uid()::text);

DROP POLICY IF EXISTS sessions_owner_insert ON sessions;
CREATE POLICY sessions_owner_insert
ON sessions
FOR INSERT
WITH CHECK (user_id = auth.uid()::text);

DROP POLICY IF EXISTS sessions_owner_update ON sessions;
CREATE POLICY sessions_owner_update
ON sessions
FOR UPDATE
USING (user_id = auth.uid()::text)
WITH CHECK (user_id = auth.uid()::text);

DROP POLICY IF EXISTS sessions_owner_delete ON sessions;
CREATE POLICY sessions_owner_delete
ON sessions
FOR DELETE
USING (user_id = auth.uid()::text);

DROP POLICY IF EXISTS chunk_embeddings_owner_select ON chunk_embeddings;
CREATE POLICY chunk_embeddings_owner_select
ON chunk_embeddings
FOR SELECT
USING (
  EXISTS (
    SELECT 1
    FROM sessions s
    WHERE s.session_id = chunk_embeddings.session_id
      AND s.user_id = auth.uid()::text
  )
);

