# JSONL 数据格式问题与修复

## 问题描述

在训练 ChatTS 模型时，使用 `gdsh_hbsn_data.jsonl` 数据集遇到以下错误：

### 错误 1: output 类型不一致

```
pyarrow.lib.ArrowInvalid: JSON parse error: Column(/output) changed from array to string in row 27
```

**原因**: `output` 字段部分是 Python 列表类型，部分是字符串类型。

### 错误 2: 缺少时序占位符

```
ValueError: Mismatch between <ts><ts/> placeholders (0) and time series (0).
```

**原因**: `input` 字段缺少 `<ts><ts/>` 占位符。

---

## ChatTS 训练数据格式要求

### 正确格式

```json
{
    "input": "<ts><ts/>\n你是一位时间序列异常检测专家...",
    "output": "{\"status\": \"success\", \"detected_anomalies\": [...]}",
    "timeseries": [[1.0, 2.0, 3.0, ...]]
}
```

### 格式要点

| 字段 | 类型 | 要求 |
|-----|------|-----|
| input | string | **必须**包含 `<ts><ts/>` 占位符 |
| output | string | 必须是 JSON **字符串**（不是对象） |
| timeseries | list | 嵌套列表格式 `[[...]]` |

---

## 解决方案

### 修复脚本

`scripts/fix_jsonl_format.py`

**功能**:
1. 在 `input` 开头添加 `<ts><ts/>\n` 占位符
2. 将 `output` 字段转换为 JSON 字符串
3. 确保 `timeseries` 是嵌套列表格式

**使用方法**:

```bash
python scripts/fix_jsonl_format.py <输入文件> <输出文件>

# 示例
python scripts/fix_jsonl_format.py \
    data/chatts_tune/gdsh_hbsn_data.jsonl \
    data/chatts_tune/gdsh_hbsn_data_chatts.jsonl
```

### 修复结果

| 项目 | 数值 |
|-----|------|
| 总记录数 | 2238 |
| 添加 `<ts><ts/>` | 2238 (100%) |
| 修复 output 类型 | 1038 |
| 跳过无效记录 | 0 |

---

## 相关文件

| 文件 | 说明 |
|-----|------|
| `data/chatts_tune/gdsh_hbsn_data.jsonl` | 原始数据（格式错误）|
| `data/chatts_tune/gdsh_hbsn_data_chatts.jsonl` | 修复后数据 ✅ |
| `data/chatts_tune/train.jsonl` | 另一份正确格式数据 |
| `scripts/fix_jsonl_format.py` | 修复脚本 |

---

## 使用修复后的数据训练

修改 `/home/share/data/training_chatts/dataset_info.json`:

```json
"chatts_tune": {
    "file_name": "chatts_tune/gdsh_hbsn_data_chatts.jsonl",
    ...
}
```

或使用已有的正确格式文件:

```json
"chatts_tune": {
    "file_name": "chatts_tune/train.jsonl",
    ...
}
```
