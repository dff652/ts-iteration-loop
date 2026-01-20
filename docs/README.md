# TS-Iteration-Loop 时序异常检测迭代循环系统

> 整合数据获取、标注、微调、推理四大模块的统一平台

## 项目状态

🟡 **开发中** - Phase 1: 基础集成

## 功能概览

```
数据获取 → 标注 → 微调 → 推理 → (反馈) → 标注 → ...
```

| 模块 | 状态 | 说明 |
|------|------|------|
| 数据获取 | 🔲 待集成 | 封装 Data-Processing 脚本 |
| 标注 | 🔲 待集成 | 集成 timeseries-annotator-v2 |
| 微调 | 🔲 待集成 | 封装 ChatTS-Training |
| 推理 | 🔲 待集成 | 封装 check_outlier |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m src.main
```

## 目录结构

```
ts-iteration-loop/
├── src/                      # 源代码
│   ├── api/                  # FastAPI 路由
│   │   ├── data.py           # 数据服务 API
│   │   ├── annotation.py     # 标注服务 API
│   │   ├── training.py       # 微调服务 API
│   │   └── inference.py      # 推理服务 API
│   ├── adapters/             # 外部项目适配器
│   │   ├── data_processing.py
│   │   ├── annotator.py
│   │   ├── chatts_training.py
│   │   └── check_outlier.py
│   ├── core/                 # 核心业务逻辑
│   │   ├── version.py        # 版本管理
│   │   ├── task_queue.py     # 任务队列
│   │   └── auth.py           # 认证 (复用JWT)
│   ├── models/               # 数据模型
│   │   └── schemas.py
│   ├── db/                   # 数据库
│   │   └── database.py
│   └── main.py               # 应用入口
├── configs/                  # 配置文件
│   └── settings.py
├── scripts/                  # 工具脚本
├── docs/                     # 文档
│   ├── README.md             # 本文档
│   ├── DEVELOPMENT.md        # 开发文档
│   ├── API.md                # API 文档
│   └── CHANGELOG.md          # 更新日志
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```

## 相关项目

| 项目 | 路径 | 用途 |
|------|------|------|
| Data-Processing | `/home/douff/ts/Data-Processing` | 数据采集与处理 |
| timeseries-annotator-v2 | `/home/douff/ts/timeseries-annotator-v2` | 标注工具 |
| ChatTS-Training | `/home/douff/ts/ChatTS-Training` | 模型微调 |
| check_outlier | `/home/douff/ilabel/check_outlier` | 推理检测 |

## 文档

- [开发文档](DEVELOPMENT.md) - 开发进度与技术细节
- [API 文档](API.md) - 接口说明
- [更新日志](CHANGELOG.md) - 版本历史
