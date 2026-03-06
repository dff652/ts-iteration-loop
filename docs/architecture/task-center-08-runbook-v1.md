# Task Center 运维 Runbook（v1）

## 1. 组件清单

- 主应用：FastAPI
- 执行器：Celery worker
- 调度器（v1.1）：dispatcher / scheduler
- 事件适配器（高级能力，按开关启用）：webhook / file_watch
- 存储：SQLite（当前），后续可切 Redis + DB

## 2. 启动顺序（建议）

1. 启动主应用
2. 启动 Celery worker
3. 启动 task center dispatcher（上线后）
4. 启动 task center scheduler（上线后）
5. 若 `TASK_CENTER_EVENTS_ENABLED=true`，再启动 file_watch 适配器（如使用）

## 3. 健康检查

- 主服务健康：`GET /health`
- 任务中心定义：`GET /api/v1/task-center/definitions`
- 事件指标（高级能力启用时）：`GET /api/v1/task-center/events/metrics`
- 死信列表（高级能力启用时）：`GET /api/v1/task-center/events/dead-letters`
- 运行状态：`GET /api/v1/task-center/runs/{run_id}/status`
- 运行日志：`GET /api/v1/task-center/runs/{run_id}/log`

## 4. 常见故障与处理

### 任务长期 pending

- 检查 dispatcher 是否运行
- 检查执行器可用性（worker 是否在线）
- 检查依赖状态是否满足（blocked 未转 runnable）

### file_watch 目录堆积

- 在定义 `config` 中设置 `post_action=move` 和 `archive_dir`
- 建议同时设置 `archive_by_date=true`，按天分层归档
- 或设置 `post_action=delete`（仅在可接受丢弃源文件时）

### 任务取消不生效

- 检查执行器取消接口是否返回成功
- 检查 step timeout 是否配置
- 必要时手动标记 run 终态并记录审计

### 日志无输出

- 检查 step logs 回写逻辑
- 检查 offset/max_bytes 参数
- 检查任务是否实际进入 running

### 死信积压

- 检查 `TASK_CENTER_DEAD_LETTER_PATH` 文件大小与增长速度
- 调用 `POST /api/v1/task-center/events/dead-letters/replay` 分批重放
- 若持续增长，先检查 webhook 签名/限流配置

## 5. 排查命令（示例）

```bash
pytest -q tests/test_task_center_api.py
```

```bash
python -m src.main
```

```bash
python -m celery -A src.core.tasks worker --loglevel=info
```

```bash
python -m src.task_center.event_adapters
```

## 6. 运行基线

- 调度周期：5~10 秒（按环境）
- 日志增量单次上限：200KB（接口支持参数）
- 超时阈值按任务类型配置（推理/训练分开）
- 事件限流：`TASK_CENTER_EVENT_RATE_LIMIT_PER_MIN`（默认 300）
