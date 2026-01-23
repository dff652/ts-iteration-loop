# 数据准备指南 (Data Preparation Guide)

本文档介绍如何为 ChatTS 微调准备数据集。

## 1. 数据格式要求

ChatTS 采用 `jsonl` 格式存储文本和时序数据。每行应为一个完整的 JSON 对象。

### 字段说明
- `input`: 包含文本提示词（Prompt）。必须包含占位符 `<ts><ts/>`，模型会在此处插入时序特征。
- `output`: 期望的文本输出（Response）。
- `timeseries`: 一个嵌套列表（List of Lists）。
    - 列表中的每个子列表代表一个长度为 **256** 的一维时序数据。
    - 子列表的数量必须与 `input` 中 `<ts><ts/>` 占位符的数量一致。

### 示例数据
```json
{
  "input": "你是一个时序分析专家。现有指标 \"CPU 使用率\": <ts><ts/>，请分析其特征。",
  "output": "该指标在点 120 附近出现明显向上尖峰，随后恢复正常。",
  "timeseries": [[0.1, 0.2, 0.15, ...]] 
}
```

## 2. 数据集注册

微调前，必须在项目根目录的 `data/dataset_info.json` 中注册您的数据集。

### 配置示例
```json
"chatts_tune_data": {
  "file_name": "/home/data1/dataset/chatts_tune/json/train.jsonl",
  "columns": {
    "prompt": "input",
    "response": "output",
    "timeseries": "timeseries"
  }
}
```
*注意：`file_name` 可以使用绝对路径。*

## 3. 存放建议
- **JSON 文件**：建议放在 `/home/data1/dataset/chatts_tune/json/`。
- **原始时序数据**：如果 JSON 中存的是路径而非数值，请确保路径解析逻辑正确（详见项目 README 的 Data Preprocessing 部分）。
