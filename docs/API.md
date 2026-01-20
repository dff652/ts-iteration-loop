# API 文档

## 概述

TS-Iteration-Loop 提供 RESTful API，支持数据获取、标注、微调、推理四大模块的管理。

**Base URL**: `http://localhost:8000/api/v1`

---

## 认证

所有 API 需要 JWT Token 认证（复用标注工具）。

```
Authorization: Bearer <token>
```

---

## API 端点

### 1. 数据服务 `/data`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/data/datasets` | 获取数据集列表 |
| POST | `/data/acquire` | 触发数据采集任务 |
| GET | `/data/status/{task_id}` | 获取采集任务状态 |

### 2. 标注服务 `/annotation`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/annotation/files` | 获取可标注文件列表 |
| GET | `/annotation/{filename}` | 获取文件标注 |
| POST | `/annotation/{filename}` | 保存标注 |
| POST | `/annotation/import-inference` | 导入推理结果为预标注 |

### 3. 微调服务 `/training`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/training/configs` | 获取训练配置列表 |
| POST | `/training/start` | 启动训练任务 |
| GET | `/training/status/{task_id}` | 获取训练状态 |
| POST | `/training/stop/{task_id}` | 停止训练 |

### 4. 推理服务 `/inference`

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/inference/batch` | 批量推理任务 |
| GET | `/inference/status/{task_id}` | 获取推理状态 |
| GET | `/inference/results/{task_id}` | 获取推理结果 |

### 5. 版本管理 `/version`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/version/iterations` | 获取迭代版本列表 |
| POST | `/version/export/{version_id}` | 导出版本为文件 |

---

## 数据格式

### 通用响应格式

```json
{
  "success": true,
  "data": { ... },
  "message": "操作成功"
}
```

### 错误响应

```json
{
  "success": false,
  "error": {
    "code": "TASK_NOT_FOUND",
    "message": "任务不存在"
  }
}
```

---

## 待实现

> 以上 API 为设计文档，实际开发中会逐步实现。
