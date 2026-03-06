# Task Center 降复杂度改造清单（v1）

## 1. 目标

- 保留统一任务中心方向，但将当前版本收敛到“推理/微调独立、手动+定时优先”的主场景。
- 高级能力不删除，改为默认关闭，避免对主路径引入额外复杂度。

## 2. 三线改造

### 2.1 接口线（Backend）

- 核心能力保持默认可用：`definitions/runs/status/log/results/cancel/retry`。
- 事件链路改为显式开关：`TASK_CENTER_EVENTS_ENABLED=true` 时才开放：
  - `/events/trigger`
  - `/events/webhook/{source}`
  - `/events/dead-letters`
  - `/events/dead-letters/replay`
  - `/events/metrics`
- 当事件开关关闭时：
  - `trigger_mode=event` 的定义创建返回 `409`。
  - 事件相关 API 返回 `404`。

### 2.2 前端线（Frontend）

- 默认只展示“运行看板”核心入口。
- 高级页签（死信补偿、DAG 编排）由 `VITE_TASK_CENTER_ADVANCED_UI_ENABLED=true` 控制显示。
- 高级页签关闭时，不主动请求事件指标/死信/定义列表，降低页面负担。

### 2.3 文档线（Docs）

- `docs/API.md`：明确 Core/Advanced 分层及开关。
- `docs/architecture/task-center-04-api-contract-v1.md`：补充事件 API 受开关控制。
- `docs/NEXT_PHASE_PLAN_2026Q1.md`：将 DAG 可视化从当前阶段目标降级为延后候选。

## 3. 验收标准

- 在默认配置（不开高级开关）下，手动/定时任务闭环可用且回归通过。
- 开启高级开关后，事件触发与死信能力可用，不影响核心链路。
- 文档、接口、前端行为一致，不出现“文档可用但默认不可用/可见”的口径冲突。
