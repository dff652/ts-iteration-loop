# 时序数据标注工具 - API 文档

## 基础信息

- **Base URL**: `http://localhost:5000/api`
- **Content-Type**: `application/json`

---

## 一、路径管理

### 1.1 获取当前路径

```http
GET /api/current-path
```

**响应示例**:
```json
{
  "success": true,
  "path": "/home/user/data"
}
```

### 1.2 设置数据路径

```http
POST /api/set-path
```

**请求体**:
```json
{
  "path": "/home/user/data"
}
```

**响应示例**:
```json
{
  "success": true,
  "path": "/home/user/data"
}
```

**错误响应**:
```json
{
  "success": false,
  "error": "Invalid directory path"
}
```

### 1.3 浏览服务器目录

```http
GET /api/browse-dir?path=/home/user
```

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| path | string | 否 | 目录路径，默认用户主目录 |

**响应示例**:
```json
{
  "success": true,
  "current_path": "/home/user",
  "parent_path": "/home",
  "directories": [
    {
      "name": "data",
      "path": "/home/user/data",
      "is_dir": true,
      "has_data_files": true
    }
  ],
  "has_data_files": false
}
```

---

## 二、文件管理

### 2.1 获取文件列表

```http
GET /api/files
```

**响应示例**:
```json
{
  "success": true,
  "files": [
    {
      "name": "data_2501FI001.csv",
      "has_annotations": true,
      "annotation_count": 4
    },
    {
      "name": "data_2501FI002.csv",
      "has_annotations": false,
      "annotation_count": 0
    }
  ],
  "path": "/home/user/data"
}
```

### 2.2 获取文件数据

```http
GET /api/data/{filename}
```

**响应示例**:
```json
{
  "success": true,
  "filename": "data_2501FI001.csv",
  "columns": ["timestamp", "value"],
  "data": [
    {"time": "2025-01-01T00:00:00", "val": 12.5, "series": "value", "label": ""},
    {"time": "2025-01-01T00:01:00", "val": 13.2, "series": "value", "label": ""}
  ],
  "seriesList": ["value"],
  "labelList": []
}
```

---

## 三、标注管理

### 3.1 获取标注列表

```http
GET /api/annotations/{filename}
```

**响应示例**:
```json
{
  "success": true,
  "filename": "data_2501FI001.csv",
  "annotations": [
    {
      "id": "ann_1703123456789",
      "start_index": 1820,
      "end_index": 2561,
      "label": "上行尖峰",
      "color": "#ef4444",
      "created_at": "2025-12-21T10:30:00"
    }
  ]
}
```

### 3.2 保存标注

```http
POST /api/annotations
```

**请求体**:
```json
{
  "filename": "data_2501FI001.csv",
  "annotations": [
    {
      "id": "ann_1703123456789",
      "start_index": 1820,
      "end_index": 2561,
      "overall_attributes": {
        "trend": "decrease",
        "frequency": "high_freq"
      },
      "local_changes": [
        {"id": "upward_spike", "text": "上行尖峰", "color": "#ef4444"}
      ],
      "input": "Supposing that a time series...",
      "output": "Yes, the observed series..."
    }
  ]
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "Annotations saved successfully"
}
```

### 3.3 删除标注

```http
DELETE /api/annotations/{filename}
```

**请求体**:
```json
{
  "annotation_id": "ann_1703123456789"
}
```

### 3.4 导出标注

```http
GET /api/download-annotations/{filename}
```

**响应示例**:
```json
{
  "annotations": [{
    "categories": {
      "trend": {"name": "趋势", "labels": [{"id": "decrease", "text": "下降"}]}
    },
    "local_change": {
      "name": "局部变化",
      "categories": {
        "spike": {
          "name": "尖峰类",
          "labels": [{
            "id": "upward_spike",
            "text": "上行尖峰",
            "color": "#ef4444",
            "index": [1820, 2561],
            "input": "Supposing that...",
            "output": "Yes, the observed series..."
          }]
        }
      }
    }
  }],
  "export_time": "2025-12-21T00:55:29.817221",
  "filename": "data_2501FI001.csv"
}
```

---

## 四、标签管理

### 4.1 获取标签配置

```http
GET /api/labels
```

**响应示例**:
```json
{
  "success": true,
  "labels": {
    "version": "2.0",
    "overall_attribute": {
      "name": "整体属性",
      "categories": {
        "trend": {
          "name": "趋势",
          "labels": [
            {"id": "decrease", "text": "下降"},
            {"id": "increase", "text": "上升"}
          ]
        }
      }
    },
    "local_change": {
      "name": "局部变化",
      "categories": {
        "spike": {
          "name": "尖峰类",
          "labels": [
            {"id": "upward_spike", "text": "上行尖峰", "color": "#ef4444"}
          ]
        }
      }
    },
    "custom_labels": []
  }
}
```

### 4.2 添加自定义标签

```http
POST /api/labels/custom
```

**请求体**:
```json
{
  "label": "异常波动",
  "color": "#3b82f6"
}
```

**响应示例**:
```json
{
  "success": true,
  "custom_labels": [
    {"id": "custom_1703123456789", "text": "异常波动", "color": "#3b82f6"}
  ]
}
```

### 4.3 保存标签配置

```http
POST /api/labels
```

**请求体**: 完整的标签配置JSON

---

## 五、错误码

| HTTP状态码 | 说明 |
|:----------:|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 403 | 权限不足 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

**错误响应格式**:
```json
{
  "success": false,
  "error": "错误描述信息"
}
```
