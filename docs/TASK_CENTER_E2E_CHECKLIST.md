# Task Center E2E 验证清单（推理/微调建单）

## 1. 验证目标

- 验证“新建推理/微调任务 -> 任务中心可定位并查看详情”的完整链路。
- 覆盖 API 冒烟和前端手工验收两类场景。

## 2. 前置条件

- 后端已启动：`python -m src.main`
- Worker 已启动：`celery -A src.core.tasks worker --loglevel=info`
- 前端可访问（开发或构建产物）

## 3. API 冒烟（自动）

```bash
cd /home/douff/ts/ts-iteration-loop
./scripts/e2e_task_center_smoke.sh
```

或使用统一入口：

```bash
cd /home/douff/ts/ts-iteration-loop
./scripts/start_local.sh smoke
```

可选：

```bash
API_BASE=http://127.0.0.1:8000/api/v1 ./scripts/e2e_task_center_smoke.sh
```

通过标准：

- inference run 创建成功，`/task-center/runs?run_id=` 返回 1 条。
- training run（有可用配置时）创建成功，`/task-center/runs?run_id=` 返回 1 条。

## 4. 前端手工 E2E

### 4.1 推理建单

1. 打开 `/inference` 页面。
2. 填写模型路径，选择至少一个输入文件（资产选择或手工输入）。
3. 点击“提交推理任务”。
4. 点击“前往任务中心”。
5. 预期：
   - 任务中心自动带上 `run_id` 过滤。
   - 运行列表只显示该 run。
   - 自动弹出 run 详情（若 URL 含 `auto_open=1`）。

### 4.2 微调建单

1. 打开 `/training` 页面。
2. 选择模型族、训练配置，按需选择数据集覆盖。
3. 点击“提交微调任务”。
4. 点击“前往任务中心”。
5. 预期与推理一致：可按 `run_id` 定位并打开详情。

## 5. 回归检查项

- 任务中心筛选新增 `run_id` 不影响原有 `status/task_type/trigger_mode/definition_id` 过滤。
- 任务提交失败时能看到错误提示，不出现空白页面。
- Advanced 页签关闭（默认）不影响上述链路。
