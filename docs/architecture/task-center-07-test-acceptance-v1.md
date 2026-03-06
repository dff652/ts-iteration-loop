# Task Center 测试与验收计划（v1）

## 1. 测试范围

- 定义管理：创建/查询
- 运行管理：创建/执行/状态/日志/结果
- 状态机：合法流转与冲突处理
- 幂等：重复请求不重复执行
- 调度：手动/自动/定时触发
- 回退：关闭开关后旧链路无回归

## 2. 测试分层

### 单元测试

- 状态流转函数
- DAG 依赖判断
- 幂等键计算

### 集成测试

- API 到 DB 持久化链路
- Dispatcher 到执行器投递链路
- 取消/重试/超时链路

### 回归测试

- 现有 `data/inference/training` API 行为不变
- 关键页面轮询日志与状态契约不变

## 3. 验收门槛（DoD）

- 关键 API 可用率 >= 99%
- 关键链路自动化测试通过率 100%
- 新旧结果一致性误差在可接受阈值内（按任务类型定义）
- 灰度期无 P0/P1 事故

## 4. 核心用例清单

1. 创建定义 + 手动执行 run。
2. 并行 step 调度与聚合完成。
3. step 失败重试并最终成功。
4. 运行中取消并正确终态。
5. 定时任务重入保护。
6. 执行器不可用时的错误返回与恢复。

## 5. 非功能测试

- 压测：并发 run 创建与状态查询
- 稳定性：长时间调度（>=24h）
- 数据一致性：run/step/result_index 关联校验

## 6. 执行流程（测试 -> 验证 -> 评估）

### 6.1 执行前准备

- 分支要求：在待验收分支执行，记录 `git rev-parse --abbrev-ref HEAD`。
- 环境要求：主应用和 Celery worker 可正常启动。
- 数据库要求：迁移已执行并包含 task center 相关表。

### 6.2 测试阶段（功能正确性）

1. 执行迁移检查

```bash
python scripts/db_migrate.py --status
python scripts/db_migrate.py --apply
```

2. 执行自动化回归

```bash
pytest -q tests/test_task_center_api.py tests/test_data_dispatch.py tests/test_inference_dispatch.py tests/test_training_dispatch.py
```

3. 执行接口冒烟（手工）

- `POST /api/v1/task-center/definitions`
- `POST /api/v1/task-center/runs`
- `POST /api/v1/task-center/runs/{run_id}/execute`
- `GET /api/v1/task-center/runs/{run_id}/status`
- `GET /api/v1/task-center/runs/{run_id}/log`
- `GET /api/v1/task-center/runs/{run_id}/results`

4. 测试判定

- 自动化测试全绿。
- 状态流转无非法跳转。
- 接口响应字段符合 API 合同。

### 6.3 验证阶段（新旧链路一致性）

1. Shadow 对比（同一输入跑两条链路）

- 旧入口：`/api/v1/inference`、`/api/v1/training`
- 新入口：`/api/v1/task-center/*`

2. 推理对比项

- `result_path`
- `segment_count`
- `score_avg/score_max`

3. 训练对比项

- 终态一致（completed/failed/cancelled）
- 产物目录可用
- 评估指标可读取

4. 异常链路验证

- 执行器不可用（503）
- 取消成功
- 超时转终态
- 重试行为符合配置

5. 验证判定

- 关键结果字段一致性通过。
- 异常处理路径行为符合设计。

### 6.4 评估阶段（是否进入灰度）

1. 评估指标

- 功能：核心用例通过率
- 稳定性：失败率、超时率、取消成功率
- 性能：排队时延、执行耗时、状态刷新延迟
- 运维：告警定位耗时、回退耗时

2. 建议门槛

- 核心用例通过率 = 100%
- 失败率 <= 5%
- 超时率 <= 3%
- 灰度周期内无 P0/P1 事故

3. 评估结论类型

- `Go`：满足门槛，进入下一阶段灰度
- `Hold`：局部风险，限流继续观察
- `No-Go`：不满足门槛，执行回退

## 7. 验收产出物

- 测试报告：测试命令、结果截图/日志、失败项与修复记录
- 验证报告：新旧链路对比表（输入、输出、差异、结论）
- 评估报告：指标汇总、门槛判定、Go/Hold/No-Go 决策
- 回退记录（如触发）：触发原因、回退动作、恢复时间
- 推荐模板：`docs/architecture/task-center-10-report-templates-v1.md`
