CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW());
CREATE TABLE IF NOT EXISTS items (item_id TEXT PRIMARY KEY, title TEXT NOT NULL, category TEXT, image_url TEXT);
CREATE TABLE IF NOT EXISTS interactions (interaction_id BIGSERIAL PRIMARY KEY, user_id TEXT REFERENCES users(user_id), item_id TEXT REFERENCES items(item_id), event_type TEXT DEFAULT 'play', playtime_forever DOUBLE PRECISION, event_time TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT NOW());
CREATE INDEX IF NOT EXISTS idx_interactions_user_time ON interactions(user_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_item_time ON interactions(item_id, event_time DESC);
