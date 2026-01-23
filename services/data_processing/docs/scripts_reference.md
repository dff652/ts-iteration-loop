# 脚本功能参考

## 数据采集 (scripts/acquisition/)

### get_downsampled.py

从 IoTDB 数据库读取时序数据，使用 M4 算法降采样并保存。

**功能**：
- 连接 IoTDB 数据库查询原始时序数据
- 使用 M4 降采样算法将数据压缩至 5000 点
- 保存 CSV 文件和可视化图片

**输入**：IoTDB 数据库连接参数、数据路径  
**输出**：`data_downsampled/*.csv`, `picture_data/*.jpg`

---

## 标注格式转换 (scripts/transformation/)

### convert_annotations.py

将标注工具导出的 JSON 转换为对话格式。

**功能**：
- 提取点位名称匹配图片文件
- 执行异常类型映射（如"上行尖峰"→"竖直尖峰"）
- 生成原因说明文本
- 将 overall_attribute 转换为中文描述

**输入**：标注 JSON 文件目录  
**输出**：对话格式的 JSON 文件

---

### convert_and_merge_annotations.py

合并多个 JSON 标注文件。

**功能**：
- 解析 overall_attribute 生成文字描述
- 创建包含全局属性的用户提示
- 合并所有文件到单个 JSON

**输入**：多个 JSON 标注文件  
**输出**：合并后的 JSON 文件

---

### merge_point_records.py

合并同一点位的多条异常记录。

**功能**：
- 按点位名称分组记录
- 合并多条异常，去重并排序
- 过滤"无异常"类型条目

**输入**：`gdsh.json`, `hbsn.json`, `whlj_ljsj.json`  
**输出**：`*_merged.json`

---

### modify_annotations.py

批量修改标注数据。

**功能**：
- 更新图片路径格式
- 扩展单点异常区间（±5 索引）
- 添加"无异常"条目

**输入**：`merged_annotations.json`  
**输出**：`merged_annotations_modified.json`

---

### update_anomaly_labels.py

更新异常标签名称。

**功能**：将"点异常"替换为"尖峰"

**输入/输出**：同一 JSON 文件

---

## 微调数据预处理 (scripts/preprocessing/)

### preprocess_tune_data.py

生成 ChatTS 训练用的 JSONL 数据集。

**功能**：
- 匹配 JSON 标注与 CSV 时序数据
- 处理多个数据集（gdsh、hbsn、whlj、zhlh）
- 替换 `<image>` 为 `<ts><ts/>`
- 生成 `{input, output, timeseries}` 格式

**输入**：`data/chatts_tune/json/*.json`, `data/chatts_tune/timeseries/`  
**输出**：`data/chatts_tune/train.jsonl`

---

### preprocess_gdsh.py

针对 gdsh 数据集的预处理脚本（功能同上，单数据集版本）。

---

### split_anomalies.py

将多异常样本拆分为单异常样本。

**使用示例**：
```bash
python split_anomalies.py train.jsonl -o train_split.jsonl
python split_anomalies.py train.jsonl --dry-run  # 仅统计
```

**输入**：包含多异常的 JSONL  
**输出**：每条记录仅含一个异常的 JSONL

---

### fix_jsonl_format.py

修复 JSONL 格式以符合 ChatTS 训练要求。

**修复内容**：
- 添加 `<ts><ts/>` 占位符
- 将 output 转换为 JSON 字符串
- 确保 timeseries 为嵌套列表格式

**使用示例**：
```bash
python fix_jsonl_format.py input.jsonl output.jsonl
```

---

## 数据验证 (scripts/validation/)

### check_data_quality.py

检查 JSONL 数据集质量。

**检查项**：
- 必要字段存在性（input、output、timeseries）
- `<ts><ts/>` 占位符
- timeseries 格式和长度统计
- 打印样本示例

---

### verify_conversion.py

验证转换后数据与源数据的一致性。

**验证项**：
- 异常数量匹配
- overall_attribute 转换完整性
- 异常类型一致性

---

## 辅助工具 (scripts/utils/)

### insert_excel_images.py

将检测结果图片插入 Excel 报告。

**功能**：
- 匹配点位名称与图片文件
- 自动调整图片尺寸
- 插入到指定列

**配置**：修改脚本顶部的路径变量
