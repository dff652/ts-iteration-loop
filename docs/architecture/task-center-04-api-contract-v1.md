# Task Center API 合同（v1）

## 1. 路由前缀

- `/api/v1/task-center`
- 能力分层：
  - Core（默认启用）：定义/运行/状态/日志/结果/取消/重试
  - Advanced（默认关闭）：事件触发、Webhook、死信、事件指标
- 开关：
  - `TASK_CENTER_EVENTS_ENABLED=true` 启用 Advanced 事件链路
  - `VITE_TASK_CENTER_ADVANCED_UI_ENABLED=true` 显示前端高级页签（死信、DAG）

## 2. 任务定义

### 创建定义

- `POST /definitions`
- 请求字段：`name/task_type/trigger_mode/schedule_cron/enabled/config`
- 返回：`definition_id + 定义详情`

### 查询定义列表

- `GET /definitions`
- 过滤参数（v1.1）：`task_type/enabled/trigger_mode`

## 3. 运行实例

### 创建运行

- `POST /runs`
- 请求字段：`definition_id/task_type/trigger_mode/input_payload/steps/step_specs/max_retries/retry_policy/retry_delay_sec/retry_backoff_factor/retry_max_delay_sec/retry_on_errors/timeout_sec/auto_execute`
- `steps`：简化串行步骤定义（自动生成线性依赖）
- `step_specs`：DAG 依赖定义，格式：`[{name, depends_on[]}]`
- `max_retries`：失败/超时自动重试次数（run 级）
- `retry_policy`：`fixed/exponential`
- `retry_delay_sec`：基础重试延时（秒）
- `retry_backoff_factor`：指数退避倍率（仅 `exponential`）
- `retry_max_delay_sec`：重试延时上限（秒）
- `retry_on_errors`：错误关键字过滤（为空表示不过滤）
- `timeout_sec`：运行超时阈值（秒）
- 返回：`run_id/status/steps`

### 执行运行

- `POST /runs/{run_id}/execute`
- 请求字段：`simulate`
- `simulate=true`：状态机模拟执行
- `simulate=false`：真实分发到执行器（v1 支持 `acquire`、`inference`、`training`）
- 返回：`run_id/status/steps`

### 查询状态

- `GET /runs/{run_id}/status`
- 返回：run 状态 + step 状态 + 时间戳 + error
- step 状态集：`blocked/runnable/pending/running/completed/failed/cancelled/timeout`

### 查询日志（增量）

- `GET /runs/{run_id}/log?offset=0&max_bytes=200000`
- 返回：`log/offset/exists/eof`

### 查询结果

- `GET /runs/{run_id}/results`
- 返回：`result + indexed_results`

### 取消运行

- `POST /runs/{run_id}/cancel`
- 行为：best effort 取消运行中的 step，并向执行器下发取消信号
- 返回：`run_id/status/cancelled_steps/revoke_errors`

### 重试运行

- `POST /runs/{run_id}/retry`
- 行为：复制原 run 的输入与 step 模板，创建新的 pending run
- 限制：运行中/已完成任务不可重试（返回 `409`）
- 返回：`source_run_id/new_run_id/status/steps`

### 查询运行列表

- `GET /runs`
- 过滤字段：`run_id/status/task_type/trigger_mode/definition_id`
- 分页字段：`limit/offset`

### 事件触发

- `POST /events/trigger`
- 请求字段：`event_key/payload/definition_id/dedupe_key/execute_mode`
- `execute_mode`：`dispatch/simulate/none`
- 幂等规则：`definition_id + event_key + dedupe_key`
- 返回：`matched_definitions/created_count/dispatched_count/run_ids`
- 仅在 `TASK_CENTER_EVENTS_ENABLED=true` 时可用（否则返回 404）

### Webhook 事件入口

- `POST /events/webhook/{source}`
- Header：`X-TaskCenter-Signature: sha256=<hex>`
- 签名算法：`HMAC-SHA256(body, secret)`
- secret 来源：`TASK_CENTER_WEBHOOK_SECRETS`
- body 字段：`event_key/definition_id/dedupe_key/event_id/execute_mode/payload`
- 仅在 `TASK_CENTER_EVENTS_ENABLED=true` 时可用（否则返回 404）

### 死信与重放

- `GET /events/dead-letters?limit=&offset=`
- `POST /events/dead-letters/replay`
- 重放请求字段：`event_id/execute_mode`
- 仅在 `TASK_CENTER_EVENTS_ENABLED=true` 时可用（否则返回 404）

### 事件指标

- `GET /events/metrics`
- 返回：`counters/dead_letter_total/rate_limit_per_min/rate_limit_bucket_count`
- 仅在 `TASK_CENTER_EVENTS_ENABLED=true` 时可用（否则返回 404）

## 4. 后续预留接口

- `GET /runs?status=&task_type=&created_at_from=&created_at_to=`

## 5. 错误码约定

- `404`: run/definition 不存在
- `409`: 状态冲突（已完成任务重复执行）
- `422`: 参数校验失败
- `503`: 执行器不可用

## 6. 响应约定

- 统一使用 `ApiResponse { success, data, message }`
- `message` 为用户可读提示，`data` 放结构化字段
