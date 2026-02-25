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
- `POST /data/acquire`: 提交数据采集任务
- `GET /data/status/{task_id}`: 任务状态（统一状态口径）
- `GET /data/log/{task_id}`: 增量日志（`offset/max_bytes`）

### 标注服务 `/annotation`
- `GET /annotation/files`: 可标注文件列表
- `GET /annotation/{filename}`: 获取标注
- `POST /annotation/{filename}`: 保存标注
- `POST /annotation/import-inference`: 导入推理结果为预标注
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
