-- Model Registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model_family TEXT DEFAULT 'chatts',
    model_type TEXT DEFAULT 'lora',
    version TEXT,
    model_path TEXT NOT NULL,
    base_model TEXT,
    config TEXT,
    metrics TEXT,
    train_loss REAL,
    status TEXT DEFAULT 'active',
    tags TEXT,
    description TEXT,
    source_task_id TEXT,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_registry_family ON model_registry(model_family);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status);
CREATE INDEX IF NOT EXISTS idx_model_registry_path ON model_registry(model_path);
