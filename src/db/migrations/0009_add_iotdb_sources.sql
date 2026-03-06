-- IoTDB 数据源配置表
CREATE TABLE IF NOT EXISTS iotdb_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    host TEXT DEFAULT '192.168.199.185',
    port TEXT DEFAULT '6667',
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    source_path TEXT NOT NULL,
    point_name TEXT DEFAULT '*',
    target_points INTEGER DEFAULT 5000,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_iotdb_sources_name ON iotdb_sources(name);
