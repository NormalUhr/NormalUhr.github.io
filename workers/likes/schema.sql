-- Running totals, one row per article.
CREATE TABLE IF NOT EXISTS likes (
  key   TEXT PRIMARY KEY,
  count INTEGER NOT NULL DEFAULT 0
);

-- One row per (article, reader, day), so a like counts once. The reader column is a
-- salted SHA-256 of the request IP truncated to 16 hex characters: enough to
-- deduplicate, not enough to identify anyone, and no address is ever stored.
CREATE TABLE IF NOT EXISTS voters (
  key    TEXT NOT NULL,
  reader TEXT NOT NULL,
  day    TEXT NOT NULL,
  PRIMARY KEY (key, reader, day)
);
