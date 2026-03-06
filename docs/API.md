# API 文档

## 概述
TS-Iteration-Loop 提供 RESTful API，覆盖数据获取、标注、推理、训练、数据资产管理。

- Base URL: `http://localhost:8000/api/v1`
- 通用响应: `ApiResponse { success, data, message }`

## 认证
标注服务复用 JWT：

```
Authorization: Bearer <token>
```

## 主要端点

### 数据服务 `/data`
- `GET /data/datasets`: 获取数据集列表
- `POST /data/datasets/upload`: 上传 CSV 创建数据集（可 `dataset_name`、`overwrite`）
- `POST /data/datasets/acquire`: 通过 IoTDB 配置创建数据集（语义化入口，等价于 `/data/acquire`）
- `POST /data/acquire`: 提交数据采集任务
- `GET /data/status/{task_id}`: 任务状态（统一状态口径）
- `GET /data/log/{task_id}`: 增量日志（`offset/max_bytes`）

### 标注服务 `/annotator` (复用原有工具)
- `GET /annotator/files`: 列出指定目录下的可标注文件（现已支持返回真实数据库的 `has_annotations` 与 `annotation_count` 以及兼容的 `size_bytes` 等模型字段）
- `GET /annotator/annotations/{filename}`: 读取标注
- `POST /annotator/annotations/{filename}`: 保存标注
- `GET /annotator/label_config`: 获取标签配置
- `GET /annotator/points`: 点位与标注聚合列表
- `GET /annotator/points/{point_id}`: 单个点位详情
- `POST /annotator/import-inference`: 导入推理结果为预标注
  - 支持 `inference_file`（文件路径）或 `rows`（直接传标注行）
  - 推荐使用 `rows`；`inference_file` 为兼容模式
  - `rows` 示例：
    ```json
    [
      {
        "filename": "P_1001.csv",
        "annotations": [{"id": "a1", "segments": [{"start": 1, "end": 2}]}]
      }
    ]
    ```

### 审核服务 `/review`
- `GET /review/queue`: 审核队列查询（支持 `status/method/annotation_kind/keyword/limit/offset`）
  - `status`: `pending/approved/needs_fix/unreviewed`
  - `annotation_kind`: `all/auto/human`
- `POST /review/queue/batch-update`: 批量更新审核状态
  - 请求体示例：
    ```json
    {
      "point_ids": ["P_1001", "P_1002"],
      "status": "approved",
      "reviewer": "douff"
    }
    ```

### 点位服务 `/points`
- `GET /points`: 点位列表（含标注/推理/审核聚合视图）
- `GET /points/{point_id}`: 点位摘要详情
- `GET /points/{point_id}/timeline`: 点位时间线事件

### 推理服务 `/inference`
- `POST /inference/batch`: 提交批量推理
- `POST /inference/cancel/{task_id}`: 取消任务
- `GET /inference/status/{task_id}`: 任务状态（统一状态口径）
- `GET /inference/log/{task_id}`: 增量日志（`offset/max_bytes`）
- `GET /inference/results/{task_id}`: 推理结果
- `POST /inference/export-to-annotation/{task_id}`: 导出预标注
  - 仅支持已完成任务（`status=completed`）
  - 默认返回 `rows`（内存标注行）与 `row_count`
  - 可选 `persist_file=true` 生成兼容 `annotation_file`

### 任务中心 `/task-center`（新增，旧模块保留）
- 默认运行在“核心模式”：聚焦手动/定时任务管理。
- 高级能力（事件触发、死信、DAG 编排）默认关闭，按开关启用：
  - 后端事件能力：`TASK_CENTER_EVENTS_ENABLED=true`
  - 前端高级页面：`VITE_TASK_CENTER_ADVANCED_UI_ENABLED=true`
- `POST /task-center/definitions`: 创建任务定义（默认支持手动/自动/定时）
- `GET /task-center/definitions`: 查询任务定义列表
- `POST /task-center/runs`: 创建任务运行实例（支持 `auto_execute`）
  - 支持 `step_specs`（DAG 依赖定义）：`[{\"name\":\"inference\",\"depends_on\":[\"acquire\"]}]`
  - 支持运行策略：
    - `max_retries`：失败/超时自动重试次数
    - `retry_policy`：`fixed` 或 `exponential`
    - `retry_delay_sec`：基础重试延时（秒）
    - `retry_backoff_factor`：指数退避倍率（仅 `exponential`）
    - `retry_max_delay_sec`：重试延时上限（秒）
    - `retry_on_errors`：按错误关键字触发自动重试（为空表示不过滤）
    - `timeout_sec`：运行超时阈值（秒）
- `GET /task-center/runs`: 运行实例列表（支持 `run_id/status/task_type/trigger_mode/definition_id` 过滤）
- `POST /task-center/runs/{run_id}/execute`: 触发执行
  - `simulate=true`：仅跑状态机模拟
  - `simulate=false`：真实分发（当前支持 `acquire`、`inference`、`training` step）
- `POST /task-center/runs/{run_id}/cancel`: 取消运行（best effort，向执行器发送 revoke）
- `POST /task-center/runs/{run_id}/retry`: 基于原 run 创建新的 pending run
- `GET /task-center/runs/{run_id}/status`: 运行状态与步骤状态
  - step 状态支持：`blocked/runnable/pending/running/completed/failed/cancelled/timeout`
- `GET /task-center/runs/{run_id}/log`: 增量日志查询
- `GET /task-center/runs/{run_id}/results`: 运行结果与索引结果

### 任务中心高级能力（按需启用）
- `POST /task-center/events/trigger`: 事件触发创建运行
  - 关键字段：`event_key/payload/definition_id/dedupe_key/execute_mode`
  - `execute_mode`：`dispatch/simulate/none`
  - `dedupe_key`：同一 `definition + event_key + dedupe_key` 幂等去重
- `POST /task-center/events/webhook/{source}`: Webhook 事件入口（可选 HMAC-SHA256 签名）
  - Header 默认：`X-TaskCenter-Signature: sha256=<hex>`
  - 环境变量：`TASK_CENTER_WEBHOOK_SECRETS`，格式 `source1=secret1,source2=secret2` 或 `default=<secret>`
- `GET /task-center/events/metrics`: 事件链路指标（触发总量、限流、死信等计数）
- `GET /task-center/events/dead-letters`: 死信列表（`limit/offset`）
- `POST /task-center/events/dead-letters/replay`: 死信重放（`event_id/execute_mode`）

### 事件源适配器（高级能力）
- 文件到达轮询适配器：`python -m src.task_center.event_adapters`
- 生产建议环境变量：
  - `TASK_CENTER_EVENT_RATE_LIMIT_PER_MIN`：事件触发每分钟限流（`0` 表示关闭）
  - `TASK_CENTER_DEAD_LETTER_PATH`：死信 JSONL 文件路径
- 任务定义 `config` 示例（`trigger_mode=event`）：
```json
{
  "adapter": "file_watch",
  "watch_dir": "/data/inbox",
  "file_glob": "*.json",
  "post_action": "move",
  "archive_dir": "/data/inbox_archive",
  "archive_by_date": true,
  "event_key": "file.arrived",
  "task_type": "inference",
  "steps": ["inference"],
  "input_payload": {
    "model": "/tmp/model",
    "algorithm": "chatts",
    "input_files": ["/tmp/a.csv"]
  }
}
```
- `post_action` 可选：`keep`（默认）/`move`/`delete`
- `archive_by_date=true` 时按 `archive_dir/YYYY-MM-DD/` 分层归档

### 训练服务 `/training`
- `GET /training/configs`: 训练配置列表
- `POST /training/start`: 启动训练
- `POST /training/stop/{task_id}`: 停止训练
- `GET /training/status/{task_id}`: 任务状态（统一状态口径）
- `GET /training/log/{task_id}`: 增量日志（`offset/max_bytes`）

### 数据资产服务 `/assets`
- `GET /assets/datasets`: 数据集列表（支持 `dataset_type/owner_id/org_id`）
- `GET /assets/datasets/{dataset_id}`: 数据集详情（支持 `owner_id/org_id` 校验）
- `POST /assets/datasets/save`: 保存数据集
  - 请求字段支持：`owner_id/org_id/freeze/overwrite`
- `DELETE /assets/datasets/{dataset_id}`: 删除数据集（支持 `owner_id/org_id` 校验）
- `GET /assets/sources`: 来源筛选（`annotations/inference/training`，DB-First）
- `POST /assets/export/training`: 从 DB 快照导出训练集
  - 请求字段支持：`dataset_id/model_family/approved_only/output_name/owner_id/org_id`
  - 返回包含：`output_path/selected_count/export_id/exported_at`

## 通用响应示例

```json
{
  "success": true,
  "data": {},
  "message": "ok"
}
```

## 任务日志增量接口示例

`GET /{module}/log/{task_id}?offset=0&max_bytes=200000`

```json
{
  "success": true,
  "data": {
    "task_id": "xxx",
    "status": "running",
    "log": "....",
    "offset": 1024,
    "exists": true,
    "eof": false
  },
  "message": "任务执行中"
}
```
