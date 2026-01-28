# 开发文档

## 项目信息

- **项目名称**: TS-Iteration-Loop
- **开发开始日期**: 2026-01-20
- **当前版本**: v0.2.0
- **当前阶段**: Phase 4 完成 ✅ (Monorepo 整合)

---

## 开发进度

### Phase 1: 基础集成 ✅

| 任务 | 状态 | 说明 |
|------|------|------|
| 项目结构创建 | ✅ 完成 | 创建目录和基础文件 |
| FastAPI 后端框架 | ✅ 完成 | main.py + 5个 API 路由 |
| 数据服务 API | ✅ 完成 | 封装 Data-Processing |
| 标注服务集成 | ✅ 完成 | 代理到现有标注工具 |
| 推理服务 API | ✅ 完成 | 封装 check_outlier |

### Phase 2: 微调集成 ✅

| 任务 | 状态 | 说明 |
|------|------|------|
| 微调任务配置界面 | ✅ 完成 | Gradio 嵌入 FastAPI `/train-ui` |
| 训练任务队列 | ✅ 完成 | Celery + SQLite 异步执行 |
| 训练进度监控 | ✅ 完成 | 实时解析日志获取进度与 Loss |

### Phase 3: 迭代循环 ✅

| 任务 | 状态 | 说明 |
|------|------|------|
| 推理→预标注转换 | ✅ 完成 | 实现 API `/import-inference` |
| 标注→微调数据转换 | ✅ 完成 | 实现 API `/export/training-data` |
| 版本追踪 | ✅ 完成 | Gradio 界面实现 Loss 曲线对比 |
| **迭代版本管理 API** | ✅ 完成 | CRUD `/api/v1/iteration` |
| **统一 UI 界面** | ✅ 完成 | 数据/推理/标注/微调 4 大模块 Tab |

### Phase 4: Monorepo 整合 ✅

| 任务 | 状态 | 说明 |
|------|------|------|
| 代码整合 | ✅ 完成 | `services/` 目录已包含所有模块 |
| 路径配置更新 | ✅ 完成 | 添加 `USE_LOCAL_MODULES` 开关 |
| 统一环境配置 | ✅ 完成 | 创建 `envs/environment.yml` |
| 开发脚本 | ✅ 完成 | 创建 `scripts/setup_dev.sh` |
| Docker 配置 | ✅ 完成 | 移除外部目录挂载 |

---

## 快速开始

### 本地开发环境

```bash
# 一键搭建
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh

# 激活环境
conda activate ts-iteration-loop

# 启动应用
python -m src.main
```

### Docker 开发环境

```bash
# 开发模式（热重载）
docker-compose -f docker-compose.dev.yml up --build

# 生产模式
docker-compose up --build -d
```

---

## UI 功能

访问地址: `http://localhost:8000/train-ui`

| Tab | 功能 |
|-----|------|
| 📁 数据获取 | 数据集列表、预览、IoTDB 采集配置 |
| 🔍 推理监控 | 任务创建、状态监控、结果预览 |
| 🏷️ 标注工具 | 跳转链接、使用说明 |
| 🎯 开始训练 | 微调参数配置、任务提交 |
| 📊 已训练模型 | 模型详情、Loss 曲线 |
| ⚖️ 模型对比 | 多模型 Loss 对比图 |
| ⚙️ 配置说明 | 参数文档 |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | FastAPI + Python 3.10+ |
| 前端 | Gradio (统一管理界面) |
| 任务队列 | Celery + SQLite |
| 数据库 | SQLite |
| 容器化 | Docker Compose |

---

## 项目结构 (Monorepo)

```
ts-iteration-loop/
├── src/                    # 核心应用代码
│   ├── api/                # FastAPI 路由
│   ├── adapters/           # 模块适配器
│   ├── core/               # 核心业务逻辑
│   ├── db/                 # 数据库模块
│   └── webui/              # Gradio 界面
│
├── services/               # 整合的子模块
│   ├── inference/          # 推理检测 (原 check_outlier)
│   ├── training/           # 模型训练 (LlamaFactory)
│   ├── data_processing/    # 数据处理
│   └── annotator/          # 标注工具
│
├── envs/                   # 环境配置
│   ├── environment.yml     # Conda 环境
│   └── requirements.txt    # pip 依赖
│
├── docker/                 # Docker 配置
│   ├── Dockerfile.dev      # 开发环境
│   └── Dockerfile.prod     # 生产环境
│
├── scripts/                # 脚本工具
│   └── setup_dev.sh        # 开发环境搭建
│
├── configs/                # 应用配置
│   └── settings.py         # 路径和环境配置
│
├── docker-compose.yml      # 生产部署
└── docker-compose.dev.yml  # 开发部署
```

---

## 配置说明

### settings.py 关键配置

| 配置项 | 说明 |
|--------|------|
| `USE_LOCAL_MODULES` | `True`: 使用项目内 services/ 目录<br>`False`: 使用外部路径 |
| `PYTHON_UNIFIED` | 统一模式 Python 解释器 |
| `LOCAL_*_PATH` | 本地模块路径 |
| `EXTERNAL_*_PATH` | 外部模块路径（兼容模式） |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEBUG` | `true` | 调试模式 |
| `USE_LOCAL_MODULES` | `true` | 使用本地模块 |
| `PYTHON_UNIFIED` | 当前解释器 | Python 路径 |

---

## 开发日志

### 2026-01-22

- ✅ 完成 Monorepo 整合重构
- ✅ 更新配置支持本地/外部模块切换
- ✅ 创建统一环境配置
- ✅ 创建开发环境搭建脚本
- ✅ 更新 Docker 配置移除外部依赖

### 2026-01-20

- ✅ 完成技术可行性评估和开发规划
- ✅ 创建项目结构和 API 路由
- ✅ 实现 Gradio 微调界面
- ✅ 实现推理结果自动反馈闭环
- ✅ 添加数据获取、推理监控、标注工具 Tab
- ✅ 实现迭代版本管理 API
- ✅ **MVP 完成**

---

## TODO (Phase 2 产品化)

> ⚠️ 当前 Gradio 仅适用于 MVP/Demo，生产级产品需要技术栈升级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P1 | 前端技术栈升级 | Vue 3 + Element Plus 或 React + Ant Design |
| P1 | 数据资产管理 (DB) | 引入数据库取代文件系统管理，解决链接与一致性问题 |
| P1 | 左侧统一文件管理器 | 所有模块共享文件浏览/选择 |
| P2 | 多区域联动交互 | 选中文件自动同步到其他模块 |
| P2 | 实时数据流 | WebSocket 推送任务状态 |
| P3 | 用户权限管理 | 多用户隔离和权限控制 |

### 架构演进：数据资产管理 (Data Asset Management)

当前基于文件系统管理的弊端：逻辑与物理存储强耦合、多模块共享困难（依赖符号链接）、状态一致性维护成本高。

**建议方案：引入数据库管理**

构建 `Dataset Assets` 表：
- `id`: UUID
- `uri`: 物理存储路径 (e.g. `s3://...` or `local://...`)
- `type`: `raw` | `inference_result` | `annotation`
- `meta`: JSON (关联的任务、时间范围、用户等)
- `tags`: user tags

**收益**：
1. **统一视图**：所有模块查询 DB 获取文件列表，无需 `glob` 扫描。
2. **逻辑隔离**：Annotator 可以只看 "assigned to me" 的数据，无需物理复制。
3. **一致性**：删除操作变为数据库逻辑删除或事务性物理删除。

