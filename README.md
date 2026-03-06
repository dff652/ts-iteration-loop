# TS-Iteration-Loop

时序异常检测迭代循环系统 - 集成数据采集、标注、模型微调和推理的 Monorepo 项目。

## 功能特性

- 📁 **数据获取**: IoTDB 数据采集、降采样
- 🏷️ **数据标注**: Web 标注界面，支持时序异常区间标注
- 🎯 **模型训练**: LlamaFactory 微调框架，支持 LoRA/QLoRA
- 🔍 **推理检测**: ChatTS 大模型异常检测
- 🤖 **多模态检测**: Qwen-VL 视觉语言模型异常检测
- 📊 **版本与资产管理**: 数据资产化存储与统一汇聚管理
- 🛠️ **全局任务调度**: 包含完整生命周期的异步 Task Center (任务流转、状态监控)
- 🖥️ **现代化架构**: 新旧前后端平滑过渡，全新 Vue 3 核心工作流

## 快速开始

### 本地开发

```bash
# 一键搭建开发环境
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh

# 新旧页面一键启动（推荐）
./scripts/start_local.sh legacy   # 仅旧版: 8000/train-ui
./scripts/start_local.sh dual     # 新旧同启: 8000 + 5173

# 兼容命令（dual 别名）
./scripts/start_local.sh all
```

说明：本地脚本默认设置 `ANNOTATOR_AUTH_BYPASS=true`（避免标注中心 401）。如需严格鉴权，可在启动前设置 `ANNOTATOR_AUTH_BYPASS=false`。

### Docker 部署

```bash
# 开发模式（热重载）
docker-compose -f docker-compose.dev.yml up --build

# 生产模式
docker-compose up --build -d
```

### 访问地址

- **API 文档**: http://localhost:8000/docs
- **旧版管理界面 (Gradio)**: http://localhost:8000/train-ui
- **新版任务中心 (Vue)**: http://localhost:5173/task-center

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
│   ├── frontend/           # Vue 3 现代化前端源码
│   ├── db/                 # 数据库模型与升级迁移脚本 (Alembic/SQL)
│   ├── task_center/        # 平台调度引擎
│   └── webui/              # Gradio 退役兼容界面
├── services/               # 整合的子应用模块
│   ├── inference/          # 推理检测服务
│   ├── training/           # 模型训练服务 (LlamaFactory)
│   │   └── data/           # 训练数据根目录
│   │       ├── chatts/     # ChatTS 格式数据
│   │       └── qwen/       # Qwen 格式数据
│   ├── data_processing/    # 数据处理
│   └── annotator/          # 原生标注工具节点
├── configs/                # 全局配置 (settings.py, iotdb_config.json)
├── envs/                   # 环境配置
│   ├── environment.yml     # Conda 环境
│   └── requirements.txt    # pip 依赖
├── docker/                 # Docker 配置
└── scripts/                # 脚本工具
```

## 技术栈与环境架构 (Architecture)

为解决大模型训练(Training)与推理(Inference)对 `transformers` 等库的版本冲突问题，本项目采用 **微服务化环境隔离 (Environment Isolation)** 策略：

| 模块 | 推荐环境 | 关键依赖 | 说明 |
|------|---------|---------|------|
| **WebUI / Backend** | `ts-iteration-loop` | `fastapi`, `gradio`, `celery` | 负责任务调度、界面展示、进程管理 |
| **Inference (推理)** | `chatts` (或复用主环境) | `transformers>=4.40`, `torch` | 需要支持最新模型架构 |
| **Training (训练)** | `llama_factory_env` | `transformers<=4.34/4.54`, `trl` | 严格依赖 LLaMA-Factory 的版本要求 |

**注意**：在开发模式下，如果版本兼容，允许使用统一环境 (`ts-iteration-loop`)，但需通过 `DISABLE_VERSION_CHECK=1` 等手段解决冲突。

## 配置

编辑 `configs/settings.py` 或通过环境变量配置：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `USE_LOCAL_MODULES` | 使用本地 services/ 模块 | `True` |
| `DEBUG` | 调试模式 | `True` |
| `API_PORT` | 服务端口 | `8000` |

## 📊 UI 界面

### 现代化新版 (Vue 3, `http://localhost:5173`)
| Tab | 功能 |
|-----|------|
| 📈 任务中心 | 查看所有推理/采集任务调度及底层日志、节点监控 |
| 🗂️ 数据资产 | 管理经过标注的高质量数据集与数据迁移 |
| 📍 标注工作台 | 内嵌 Annotator 标注视图、点位/时间线管理、批量状态审核 |
| ⚙️ 更多工作台 | `模块陆续迁移中` |

### 传统兼容版 (Gradio, `http://localhost:8000/train-ui`)
| Tab | 功能 |
|-----|------|
| 📁 数据获取 | 数据集列表、曲线预览、采集配置 |
| 🔍 推理监控 | 任务创建、状态监控 |
| 🏷️ 标注工具 | 跳转到本地标注工具 |
| 🎯 开始训练 | 微调参数配置 |
| 📊 已训练模型 | 模型详情、Loss 曲线 |
| ⚖️ 模型对比 | 多模型对比图 |

## 文档

- [开发文档](docs/DEVELOPMENT.md)
- [API 文档](docs/API.md)
- [更新日志](docs/CHANGELOG.md)
- [Task Center E2E 清单](docs/TASK_CENTER_E2E_CHECKLIST.md)

## 版本

- **v0.3.5** - 支持数据库迁移、引入统一数据资产、前端 Phase 3 Vue 重构与任务中心支持 (Unreleased 2026Q1)
- **v0.3.3** - 建立训练筛选与审核评估的索引闭环 (2026-02-03)
- **v0.3.0** - Qwen-VL 多模态检测集成 & 架构重构 (2026-01-28)
- **v0.2.3** - 数据链路优化 (2026-01-27)
- **v0.2.1** - 推理 UI 交互体验优化 (2026-01-23)
- **v0.2.0** - Monorepo 整合重构
- **v0.1.0** - MVP 完成
