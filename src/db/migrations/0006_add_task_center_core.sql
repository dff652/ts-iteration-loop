-- 0006_add_task_center_core.sql

CREATE TABLE IF NOT EXISTS task_definitions (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    task_type VARCHAR(50) NOT NULL DEFAULT 'acquire_inference',
    trigger_mode VARCHAR(20) NOT NULL DEFAULT 'manual',
    schedule_cron VARCHAR(100),
    enabled BOOLEAN DEFAULT 1,
    config TEXT,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),
    created_at DATETIME,
    updated_at DATETIME
);

CREATE TABLE IF NOT EXISTS task_runs (
    id VARCHAR(36) PRIMARY KEY,
    definition_id VARCHAR(36),
    task_type VARCHAR(50) NOT NULL DEFAULT 'acquire_inference',
    trigger_mode VARCHAR(20) NOT NULL DEFAULT 'manual',
    status VARCHAR(20) DEFAULT 'pending',
    input_payload TEXT,
    result TEXT,
    error TEXT,
    created_at DATETIME,
    started_at DATETIME,
    completed_at DATETIME
);

CREATE TABLE IF NOT EXISTS task_step_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id VARCHAR(36) NOT NULL,
    step_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    message TEXT,
    logs TEXT,
    result TEXT,
    created_at DATETIME,
    started_at DATETIME,
    completed_at DATETIME,
    updated_at DATETIME
);

CREATE TABLE IF NOT EXISTS task_run_result_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id VARCHAR(36) NOT NULL,
    point_id VARCHAR(200),
    model_version VARCHAR(200),
    result_path VARCHAR(500),
    status VARCHAR(20) DEFAULT 'completed',
    meta TEXT,
    created_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_task_runs_status ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_task_runs_definition_id ON task_runs(definition_id);
CREATE INDEX IF NOT EXISTS idx_task_step_runs_run_id ON task_step_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_task_run_result_index_run_id ON task_run_result_index(run_id);
