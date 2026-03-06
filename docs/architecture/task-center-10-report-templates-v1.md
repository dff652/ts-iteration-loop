# Task Center 报告模板（测试/验证/评估）

**版本**: v1  
**适用范围**: Task Center 新功能灰度与发布前评审

---

## A. 测试报告模板

### A1. 基本信息

- 分支：
- 提交：
- 执行人：
- 执行日期：
- 环境（dev/test/prod-shadow）：

### A2. 执行命令

```bash
python scripts/db_migrate.py --status
python scripts/db_migrate.py --apply
pytest -q tests/test_task_center_api.py tests/test_data_dispatch.py tests/test_inference_dispatch.py tests/test_training_dispatch.py
```

### A3. 结果摘要

- 自动化测试结果（通过/失败）：
- 失败用例列表：
- 已修复项：
- 未修复项：

### A4. 接口冒烟检查

- `POST /task-center/definitions`：
- `POST /task-center/runs`：
- `POST /task-center/runs/{run_id}/execute`：
- `GET /task-center/runs/{run_id}/status`：
- `GET /task-center/runs/{run_id}/log`：
- `GET /task-center/runs/{run_id}/results`：

### A5. 结论

- 测试结论（通过/不通过）：
- 备注：

---

## B. 验证报告模板（新旧链路一致性）

### B1. 验证范围

- 旧链路入口：
- 新链路入口：
- 输入数据范围：
- 任务类型（推理/训练）：

### B2. 对比结果表

| 用例ID | 输入摘要 | 旧链路输出 | 新链路输出 | 差异说明 | 结论（通过/失败） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CASE-001 |  |  |  |  |  |
| CASE-002 |  |  |  |  |  |
| CASE-003 |  |  |  |  |  |

### B3. 异常路径验证

- 执行器不可用（503）：
- 取消场景：
- 超时场景：
- 重试场景：

### B4. 结论

- 一致性结论（通过/不通过）：
- 关键风险：
- 建议动作：

---

## C. 评估报告模板（Go/Hold/No-Go）

### C1. 指标汇总

| 指标 | 目标阈值 | 实际值 | 是否达标 |
| :--- | :--- | :--- | :--- |
| 核心用例通过率 | 100% |  |  |
| 失败率 | <= 5% |  |  |
| 超时率 | <= 3% |  |  |
| 取消成功率 | >= 95%（建议） |  |  |
| 状态刷新延迟 | 自定义 |  |  |

### C2. 评估结论

- 结论：`Go / Hold / No-Go`
- 结论依据：
- 需修复项：
- 风险接受项：

### C3. 决策记录

- 评审人：
- 评审时间：
- 下一步计划：

---

## D. 回退记录模板（如触发）

### D1. 触发信息

- 触发时间：
- 触发条件（失败率/超时率/事故等级）：
- 影响范围：

### D2. 回退动作

- 执行动作：
- 回退耗时：
- 恢复状态：

### D3. 复盘结论

- 根因：
- 修复方案：
- 防再发措施：
