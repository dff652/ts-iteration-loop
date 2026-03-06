# Task Center 架构设计（v1）

## 1. 目标

- 对推理与微调训练进行统一任务管理。
- 保留旧入口，采用并行架构，不做破坏性替换。
- 将现有线性 pipeline 逐步升级为 DAG 依赖调度。

## 2. 架构分层

### 控制平面（新增）

- `Task Center API`
- `Dispatcher`（任务分发）
- `Scheduler`（定时/自动触发）
- `Task Store`（任务定义、运行、步骤、结果索引）

### 执行平面（复用）

- 采集执行器：复用现有 data 任务能力
- 推理执行器：复用 `inference.batch`
- 训练执行器：复用 `training.run`

### 观测平面

- 统一状态查询
- 统一增量日志接口
- 统一结果索引查询

## 3. 逻辑流程

1. 客户端创建任务定义（manual/auto/schedule）。
2. 触发创建运行实例（run），生成步骤实例（step）。
3. Dispatcher 按 DAG 依赖挑选 `runnable` step。
4. 运行 step 时调用对应执行器（Celery task）。
5. 回写 step 状态与日志，推进下游 step。
6. 全部 step 完成后回写 run 完成并生成结果索引。

## 4. 边界原则

- 任务中心负责编排和状态，不重写模型算法逻辑。
- 执行器负责业务执行，不做跨模块编排。
- 标注模块暂不进入执行编排，仅消费/回写来源字段。

## 5. 非目标（v1 不做）

- 不替换现有 `/api/v1/inference`、`/api/v1/training` 入口。
- 不引入新调度中间件（保持 Celery 体系）。
- 不做多租户隔离模型（仅保留 owner/org 字段扩展点）。
