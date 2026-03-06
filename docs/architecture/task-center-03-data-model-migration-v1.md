# Task Center 数据模型与迁移设计（v1）

## 1. 核心表

### `task_definitions`

- 任务定义（手动/自动/定时）
- 关键字段：`id/name/task_type/trigger_mode/schedule_cron/enabled/config`

### `task_runs`

- 任务运行实例（每次触发一条）
- 关键字段：`id/definition_id/task_type/trigger_mode/status/input_payload/result/error`

### `task_step_runs`

- 步骤实例（run 的子任务）
- 关键字段：`run_id/step_name/status/message/logs/result`
- v1.1 预留：`depends_on/retry_count/max_retries/timeout_seconds`

### `task_run_result_index`

- 运行结果索引
- 关键字段：`run_id/point_id/model_version/result_path/status/meta`

## 2. 索引建议

- `task_runs(status)`
- `task_runs(definition_id)`
- `task_step_runs(run_id)`
- `task_run_result_index(run_id)`

## 3. 与现有表关系

- 与现有 `tasks` 表并存（旧链路不动）。
- task center 可在 `meta` 中记录关联旧 `task_id`。
- 通过 `point_id/model_version` 与 `inference_results/model_evals` 建立追溯关系。

## 4. 迁移策略

1. 新增 SQL migration（已落地 `0006_add_task_center_core.sql`）。
2. 不改旧表字段，避免影响现有 API。
3. 新功能只写新表，旧功能继续写旧表。
4. 灰度期可双写关键状态用于核对一致性。

## 5. 数据一致性约束

- `task_type` 与步骤模板必须匹配。
- `run.status` 与 step 聚合结果一致。
- `result_index` 必须关联有效 `run_id`。
- 所有时间字段统一使用 UTC 存储。
