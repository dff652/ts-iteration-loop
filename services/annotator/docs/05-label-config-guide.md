# 时序数据标注工具 - 标签配置说明

## 概述

标签配置文件 `labels.json` 定义了标注工具中使用的预设标签体系。

## 标签层级结构

```
labels.json
├── overall_attribute (整体属性)
│   ├── trend (趋势)
│   │   ├── decrease (下降)
│   │   ├── increase (上升)
│   │   ├── steady (平稳)
│   │   └── multiple (多段式)
│   ├── seasonal (周期性)
│   │   ├── no_periodic (无周期)
│   │   ├── sin_periodic (正弦周期)
│   │   ├── square_periodic (方波周期)
│   │   └── triangle_periodic (三角波周期)
│   ├── frequency (频率)
│   │   ├── high_freq (高频)
│   │   └── low_freq (低频)
│   └── noise (噪声)
│       ├── noisy (有噪声)
│       └── no_noise (几乎无噪声)
│
├── local_change (局部变化) - 带颜色
│   ├── spike (尖峰类)
│   │   ├── upward_spike (上行尖峰) #ef4444
│   │   ├── downward_spike (下行尖峰) #f97316
│   │   ├── continuous_up_spike (连续上行尖峰) #dc2626
│   │   └── continuous_down_spike (连续下行尖峰) #ea580c
│   ├── sudden (突变类)
│   │   ├── sudden_increase (突然上升) #22c55e
│   │   └── sudden_decrease (突然下降) #16a34a
│   ├── convex (凸起类)
│   │   ├── upward_convex (上凸) #3b82f6
│   │   └── downward_convex (下凸) #6366f1
│   └── pattern (复合模式)
│       ├── rapid_rise_slow_decline (快升慢降) #8b5cf6
│       ├── slow_rise_rapid_decline (慢升快降) #a855f7
│       ├── rapid_decline_slow_rise (快降慢升) #d946ef
│       └── slow_decline_rapid_rise (慢降快升) #ec4899
│
└── custom_labels (自定义标签)
    └── [...] 用户添加的标签
```

## 配置文件位置

```
backend/config/labels.json
```

## 配置文件格式

```json
{
  "version": "2.0",
  "overall_attribute": {
    "name": "整体属性",
    "categories": {
      "category_id": {
        "name": "分类显示名",
        "labels": [
          { "id": "label_id", "text": "标签显示文本" }
        ]
      }
    }
  },
  "local_change": {
    "name": "局部变化",
    "categories": {
      "category_id": {
        "name": "分类显示名",
        "labels": [
          { "id": "label_id", "text": "标签显示文本", "color": "#颜色值" }
        ]
      }
    }
  },
  "custom_labels": []
}
```

## 标签使用说明

### 整体属性
- **用途**: 描述整个时间序列的全局特征
- **选择方式**: 每个分类单选一个
- **颜色**: 不带颜色

### 局部变化
- **用途**: 描述选中区域的局部特征
- **选择方式**: 可多选
- **颜色**: 每个标签带有颜色，用于图表高亮

### 自定义标签
- **用途**: 用户临时添加的标签
- **持久化**: 保存到配置文件
- **颜色**: 用户自选

## 如何添加新标签

### 方法1: 直接编辑配置文件

```json
{
  "local_change": {
    "categories": {
      "new_category": {
        "name": "新分类",
        "labels": [
          { "id": "new_label", "text": "新标签", "color": "#颜色值" }
        ]
      }
    }
  }
}
```

### 方法2: 通过API添加自定义标签

```bash
curl -X POST http://localhost:5000/api/labels/custom \
  -H "Content-Type: application/json" \
  -d '{"label": "新标签", "color": "#3b82f6"}'
```

## 颜色规范

建议使用以下色系：

| 分类 | 色系 | 示例 |
|------|------|------|
| 尖峰 | 红/橙 | #ef4444, #f97316 |
| 突变 | 绿色 | #22c55e, #16a34a |
| 凸起 | 蓝色 | #3b82f6, #6366f1 |
| 模式 | 紫色 | #8b5cf6, #a855f7 |
| 自定义 | 任意 | 用户选择 |
