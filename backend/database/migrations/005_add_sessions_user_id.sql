ALTER TABLE sessions
ADD COLUMN IF NOT EXISTS user_id TEXT;

CREATE INDEX IF NOT EXISTS idx_sessions_user_id
ON sessions(user_id);
