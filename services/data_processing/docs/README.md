# Data-Processing 数据处理工具集

本项目包含用于时序异常检测数据处理的脚本集合，支持从数据采集、标注转换到微调数据生成的完整流程。

## 快速开始

### 环境依赖

```bash
pip install pandas numpy matplotlib iotdb-session tsdownsample openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 典型工作流程

```bash
# 1. 从 IoTDB 获取降采样数据
python scripts/acquisition/get_downsampled.py

# 2. 使用 timeseries-annotator-v2 进行人工标注（外部工具）

# 3. 转换标注格式
python scripts/transformation/convert_annotations.py

# 4. 生成微调数据集
python scripts/preprocessing/preprocess_tune_data.py

# 5. 拆分多异常样本（可选）
python scripts/preprocessing/split_anomalies.py data/train.jsonl

# 6. 验证数据质量
python scripts/validation/check_data_quality.py
```

## 目录结构

```
Data-Processing/
├── scripts/                    # 脚本按功能分类
│   ├── acquisition/            # 数据采集 (1)
│   │   └── get_downsampled.py
│   ├── transformation/         # 格式转换 (5)
│   │   ├── convert_annotations.py
│   │   ├── convert_and_merge_annotations.py
│   │   ├── merge_point_records.py
│   │   ├── modify_annotations.py
│   │   └── update_anomaly_labels.py
│   ├── preprocessing/          # 微调数据预处理 (4)
│   │   ├── preprocess_tune_data.py
│   │   ├── preprocess_gdsh.py
│   │   ├── split_anomalies.py
│   │   └── fix_jsonl_format.py
│   ├── validation/             # 数据验证 (2)
│   │   ├── check_data_quality.py
│   │   └── verify_conversion.py
│   └── utils/                  # 辅助工具 (1)
│       └── insert_excel_images.py
├── data_downsampled/           # 降采样CSV数据
├── converted_annotations/      # 转换后的标注
├── res_annotations/            # 标注工具输出
└── docs/                       # 文档
    ├── README.md               # 本文档
    ├── scripts_reference.md    # 脚本功能参考
    └── data_pipeline.md        # 数据流程说明
```

## 相关项目

| 项目 | 路径 | 用途 |
|------|------|------|
| timeseries-annotator-v2 | `/home/douff/ts/timeseries-annotator-v2` | 时序数据标注工具 |
| ChatTS-Training | `/home/douff/ts/ChatTS-Training` | ChatTS 模型微调 |

## 详细文档

- [脚本功能参考](scripts_reference.md) - 各脚本的详细使用说明
- [数据处理流程](data_pipeline.md) - 完整的数据管道说明

