CREATE TABLE IF NOT EXISTS annotation_records (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_kind TEXT DEFAULT 'human',
    source_inference_id TEXT,
    method TEXT,
    status TEXT DEFAULT 'draft',
    is_human_edited INTEGER DEFAULT 1,
    annotation_count INTEGER DEFAULT 0,
    segment_count INTEGER DEFAULT 0,
    overall_attribute_json TEXT,
    annotations_json TEXT,
    meta TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS annotation_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotation_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    ann_index INTEGER DEFAULT 0,
    seg_index INTEGER DEFAULT 0,
    start INTEGER,
    end INTEGER,
    count INTEGER,
    label_id TEXT,
    label_text TEXT,
    score REAL,
    review_status TEXT DEFAULT 'pending',
    created_at TEXT,
    updated_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_annotation_records_user_source
ON annotation_records (user_id, source_id);

CREATE INDEX IF NOT EXISTS idx_annotation_records_user_updated
ON annotation_records (user_id, updated_at);

CREATE INDEX IF NOT EXISTS idx_annotation_records_source_kind
ON annotation_records (source_kind);

CREATE INDEX IF NOT EXISTS idx_annotation_segments_annotation
ON annotation_segments (annotation_id);

CREATE INDEX IF NOT EXISTS idx_annotation_segments_source
ON annotation_segments (user_id, source_id);
