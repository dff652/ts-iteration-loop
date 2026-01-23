# TS-Iteration-Loop

时序异常检测迭代循环系统 - 集成数据采集、标注、模型微调和推理的 Monorepo 项目。

## 功能特性

- 📁 **数据获取**: IoTDB 数据采集、降采样
- 🏷️ **数据标注**: Web 标注界面，支持时序异常区间标注
- 🎯 **模型训练**: LlamaFactory 微调框架，支持 LoRA/QLoRA
- 🔍 **推理检测**: ChatTS 大模型异常检测
- 📊 **版本管理**: 模型版本追踪、Loss 对比

## 快速开始

### 本地开发

```bash
# 一键搭建开发环境
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh

# 激活环境并启动
conda activate ts-iteration-loop
python -m src.main
```

### Docker 部署

```bash
# 开发模式（热重载）
docker-compose -f docker-compose.dev.yml up --build

# 生产模式
docker-compose up --build -d
```

### 访问地址

- **API 文档**: http://localhost:8000/docs
- **管理界面**: http://localhost:8000/train-ui

## 项目结构

```
ts-iteration-loop/
├── src/                    # 核心应用代码
│   ├── api/                # FastAPI 路由
│   ├── adapters/           # 模块适配器
│   └── webui/              # Gradio 界面
├── services/               # 整合的子模块
│   ├── inference/          # 推理检测
│   ├── training/           # 模型训练 (LlamaFactory)
│   ├── data_processing/    # 数据处理
│   └── annotator/          # 标注工具
├── envs/                   # 环境配置
│   ├── environment.yml     # Conda 环境
│   └── requirements.txt    # pip 依赖
├── docker/                 # Docker 配置
└── scripts/                # 脚本工具
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | FastAPI + Python 3.10+ |
| 前端 | Gradio |
| 任务队列 | Celery |
| 数据库 | SQLite |
| 容器化 | Docker Compose |

## 配置

编辑 `configs/settings.py` 或通过环境变量配置：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `USE_LOCAL_MODULES` | 使用本地 services/ 模块 | `True` |
| `DEBUG` | 调试模式 | `True` |
| `API_PORT` | 服务端口 | `8000` |

## 文档

- [开发文档](docs/DEVELOPMENT.md)
- [API 文档](docs/API.md)
- [更新日志](docs/CHANGELOG.md)

## 版本

- **v0.2.0** - Monorepo 整合重构
- **v0.1.0** - MVP 完成
