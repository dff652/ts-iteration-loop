-- 0010: 添加平台用户表
-- 角色: admin / annotator / reviewer

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    VARCHAR(64)  NOT NULL UNIQUE,
    password_hash VARCHAR(256) NOT NULL,
    display_name VARCHAR(128),
    role        VARCHAR(32)  NOT NULL DEFAULT 'annotator',
    is_active   BOOLEAN      NOT NULL DEFAULT 1,
    last_login_at DATETIME,
    created_at  DATETIME     NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username);

-- 默认管理员账户 (密码: admin123)
INSERT OR IGNORE INTO users (username, password_hash, display_name, role)
VALUES (
    'admin',
    'sha256:240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9',
    '管理员',
    'admin'
);
