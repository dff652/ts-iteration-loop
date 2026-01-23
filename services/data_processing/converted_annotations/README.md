# 标注数据转换说明

## 转换脚本
`/home/douff/convert_and_merge_annotations.py`

## 转换内容

### 输入
- 源目录：`/home/douff/ts/timeseries-annotator-v2/backend/annotations/douff`
- 文件数量：99个JSON文件

### 输出
- 合并文件：`/home/douff/converted_annotations/merged_annotations.json`
- 文件大小：449.47 KB
- 样本数量：99个

## 格式调整

### 原始格式
```json
{
  "filename": "数据集zhlh_100_XXX.csv",
  "overall_attribute": {
    "frequency": "low_freq",
    "noise": "clean",
    "seasonal": "no_periodic",
    "trend": "multiple"
  },
  "annotations": [...]
}
```

### 转换后格式
```json
{
  "image": "/home/douff/数据标注/data/picture_data/数据集zhlh_100_XXX.jpg",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n你是一位时间序列异常检测专家。请分析图中的低频、低噪声、无周期性的时间序列数据，并识别异常发生的区间。\n\n..."
    },
    {
      "from": "assistant",
      "value": "{\"status\": \"success\", \"detected_anomalies\": [...]}..."
    }
  ]
}
```

## 核心调整
1. **全局属性嵌入user提示**：将 `overall_attribute` 中的频率、噪声、周期性特征嵌入到user的value中
   - 原：`请分析图中的时间序列数据,识别异常区间`
   - 新：`请分析图中的低频、低噪声、无周期性的时间序列数据，并识别异常发生的区间`

2. **格式统一**：所有标注转换为标准的 conversations 格式，适用于对话模型训练

3. **自动推理**：根据标注类型和区间自动生成合理的reason描述

## 属性映射

### 频率 (frequency)
- `low_freq` → 低频
- `mid_freq` → 中频
- `high_freq` → 高频

### 噪声 (noise)
- `clean` / `no_noise` → 低噪声/无噪声
- `moderate_noise` → 中等噪声
- `noisy` → 高噪声

### 周期性 (seasonal)
- `periodic` → 有周期性
- `local_periodic` → 局部有周期性
- `no_periodic` → 无周期性

## 使用方法

重新运行转换：
```bash
python3 /home/douff/convert_and_merge_annotations.py
```

查看结果统计：
```bash
python3 -c "
import json
data = json.load(open('/home/douff/converted_annotations/merged_annotations.json'))
print(f'总样本数: {len(data)}')
"
```
