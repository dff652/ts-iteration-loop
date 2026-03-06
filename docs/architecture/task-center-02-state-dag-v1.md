# Task Center 状态机与 DAG 规则（v1）

## 1. 统一状态枚举

- `blocked`: 依赖未满足，不可执行
- `runnable`: 依赖满足，可执行
- `pending`: 已入队待执行
- `running`: 执行中
- `completed`: 执行成功
- `failed`: 执行失败
- `cancelled`: 人工/系统取消
- `timeout`: 超时终止

## 2. 实体状态

### run 状态

- 聚合 step 状态得到 run 状态。
- 任一 step `failed/timeout`，run 进入 `failed/timeout`。
- 所有 step `completed`，run 进入 `completed`。

### step 状态

- 初始：`blocked` 或 `runnable`。
- 调度：`runnable -> pending -> running`。
- 结束：`running -> completed/failed/cancelled/timeout`。

## 3. DAG 依赖规则

- 每个 step 可配置 `depends_on`（上游 step 列表）。
- 当 `depends_on` 全部 `completed`，step 变 `runnable`。
- 允许并行：多个 `runnable` step 可并发执行（受并发阈值控制）。

## 4. 失败与重试规则

- step `failed/timeout` 可重试为新 attempt。
- 达到 `max_retries` 后保持终态，不再自动重试。
- run 是否继续执行由策略决定：
- `fail_fast=true`: 任一步骤失败即终止 run。
- `fail_fast=false`: 可继续执行不依赖失败步骤的后续分支。

## 5. 取消规则

- 取消 run 时，将 `pending/runnable/blocked` step 标记 `cancelled`。
- `running` step 发取消信号，等待执行器确认。
- 执行器无响应时按超时策略转 `timeout`。

## 6. 推荐 DAG 模板（推理+训练）

- 模板 A（推理）：`acquire -> infer -> index`
- 模板 B（训练）：`select_dataset -> train -> eval -> register`
- 模板 C（并行）：`acquire_shard_* -> infer_shard_* -> merge_index`
