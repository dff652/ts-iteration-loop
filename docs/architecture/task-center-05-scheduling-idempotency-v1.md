# Task Center 调度与幂等策略（v1）

## 1. 触发模式

- `manual`: 用户显式触发
- `auto`: 事件触发（如新资产冻结）
- `schedule`: Cron 定时触发

## 2. 调度策略

- Dispatcher 周期扫描 `runnable` step。
- 按 `priority + created_at` 选取待执行队列。
- 每类任务设置并发上限（如 infer/train 独立阈值）。

## 3. 幂等键

- `idempotency_key = hash(task_type + trigger_mode + normalized_input + window_bucket)`
- 创建 run 时先查幂等键：
- 存在未终态 run 则复用已有 run_id。
- 已终态 run 根据策略决定是否新建（默认新建）。

## 4. 重入保护

- 定时任务在同一窗口（如 5min）仅允许一个有效 run。
- 同一 `definition_id` 的运行互斥策略可配置：
- `allow_parallel=true/false`

## 5. 超时与重试

- step 级别设置 `timeout_seconds`。
- 超时后状态转 `timeout`，并触发重试策略。
- 指数退避：`base_delay * 2^attempt`，并设置最大重试次数。

## 6. 取消策略

- run 取消优先调用执行器取消接口（Celery revoke / 子进程中断）。
- 取消结果回写 step 与 run 状态。
- 超过取消等待阈值，标记 `timeout` 并记录审计日志。

## 7. 审计字段建议

- `created_by/triggered_by/cancelled_by`
- `idempotency_key/dispatch_batch_id/retry_parent_run_id`
