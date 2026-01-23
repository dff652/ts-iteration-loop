# 数据预处理问题记录

## 概述

本文档记录了 ChatTS 微调数据预处理过程中发现的问题及解决方案。

## 问题一：多记录点位需要合并

### 问题描述

原始 JSON 标注数据存在两种格式：

| 格式类型 | 示例文件 | 说明 |
|---------|---------|------|
| 单记录格式 | zhlh.json | 每个点位一条记录，包含所有异常 |
| 多记录格式 | gdsh.json, hbsn.json, whlj_ljsj.json | 每个点位多条记录，每条记录只有部分异常 |

### 解决方案

创建合并脚本 `scripts/merge_point_records.py`：
- 按点位名称分组
- 合并同一点位的所有异常区间（去重）
- 过滤 "无异常" 类型记录
- 按起始索引排序

**输出文件**：`*_merged.json`

---

## 问题二：gdsh.json 混入了其他数据集的数据

### 问题描述

原始 `gdsh.json` 文件中混入了来自其他数据集的数据：

| 混入数据 | 数量 | 特征 |
|---------|------|------|
| whlj 数据 | 138 条 | 图片路径包含 `NB.LJSJ` |
| hbsn 数据 | 4 条 | 图片路径为 `gdsh_second_AT_xxx` 格式 |

### 原因分析

数据标注时可能将不同来源的数据放在同一个 JSON 文件中，导致数据来源与文件名不匹配。

### 解决方案

修改 `scripts/preprocess_tune_data.py`，添加智能识别逻辑：

```python
# 智能识别: 如果在 gdsh 目录找不到,且是 NB.LJSJ 点位,则尝试在 whlj 目录查找
if not target_path and ts_subdir == "gdsh" and "NB.LJSJ" in csv_name_from_json:
    whlj_dir = os.path.join(ts_root_dir, "whlj")
    whlj_filename = "数据集whlj_ljsj_" + csv_name_from_json
    whlj_path = os.path.join(whlj_dir, whlj_filename)
    if os.path.exists(whlj_path):
        target_path = whlj_path

# 智能识别: 如果在 gdsh 目录找不到,且是 hbsn 格式,则尝试在 hbsn 目录查找
if not target_path and ts_subdir == "gdsh" and csv_name_from_json.startswith("gdsh_second_"):
    hbsn_dir = os.path.join(ts_root_dir, "hbsn")
    hbsn_filename = csv_name_from_json.replace("gdsh_second_", "")
    hbsn_path = os.path.join(hbsn_dir, hbsn_filename)
    if os.path.exists(hbsn_path):
        target_path = hbsn_path
```

---

## 问题三：CSV 文件命名规则不一致

### 问题描述

不同数据集的 CSV 文件命名规则不同：

| 数据集 | JSON 图片名 | CSV 文件名 |
|-------|------------|-----------|
| gdsh | `gdsh_second_xxx.PV.jpg` | `数据集gdsh_second_xxx.PV.csv` |
| hbsn | `gdsh_second_AT_xxx.PV.jpg` | `AT_xxx.PV.csv` |
| whlj | `NB.LJSJ.xxx.PV.jpg` | `数据集whlj_ljsj_NB.LJSJ.xxx.PV.csv` |
| zhlh | `zhlh_xxx.PV.jpg` | 直接匹配 |

### 解决方案

在 `preprocess_tune_data.py` 中针对每个数据集定义转换规则，尝试多种可能的文件名匹配。

---

## 最终处理结果

| 数据集 | 原始记录数 | 合并后记录数 | 处理成功 |
|-------|-----------|-------------|---------|
| gdsh | 321 | 179 | 179 ✅ |
| hbsn | 142 | 109 | 109 ✅ |
| whlj_ljsj | 138 | 29 | 29 ✅ |
| zhlh | 100 | 100 | 100 ✅ |
| **总计** | **701** | **417** | **417 (100%)** |

## 相关脚本

- `scripts/merge_point_records.py` - 合并多记录点位
- `scripts/preprocess_tune_data.py` - 生成训练数据 train.jsonl
