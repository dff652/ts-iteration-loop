# 开发文档

## 项目信息

- **项目名称**: TS-Iteration-Loop
- **开发开始日期**: 2026-01-20
- **当前版本**: v0.3.5
- **当前阶段**: Phase 4 完成 ✅；MVP+ 索引/审核迭代进行中

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

### 环境部署模式

项目支持两种部署模式，可通过 `ENV_MODE` 环境变量控制。

#### 1. 统一环境模式 (Unified) - **推荐**
所有模块（WebUI、推理、训练、标注）运行在同一个 Conda 环境 `ts-iteration-loop` 中。
- **优点**：一键搭建，维护简单，无需管理多个环境。
- **操作**：直接运行 `./scripts/setup_dev.sh` 即可。

#### 2. 半统一模式 (Legacy)
WebUI 运行在主环境，但训练、标注等模块调用外部独立的 Conda 环境（如 `chatts_tune`）。
- **用途**：当统一环境遇到无法解决的兼容性问题时的**回退方案**。
- **操作**：
  1. 运行 `./scripts/setup_dev.sh` 搭建主环境。
  2. 确保服务器上已存在旧版独立环境。
  3. 启动时指定变量：`ENV_MODE=legacy python -m src.main`
  4. 或在 `.env` 中设置 `ENV_MODE=legacy`。

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

### 2026-02-04 (v0.3.5) - 环境统一与部署优化

- ✅ **统一环境支持** (Unified Environment):
  - 核心依赖更新：补充 `tiktoken`, `tyro`, `openpyxl` 等缺失库
  - 版本兼容优化：为 `transformers`, `peft` 等添加版本上限，完美支持 LlamaFactory
  - 全流程验证：统一环境已完全支持 Qwen/ChatTS 的推理与微调
- ✅ **部署双模式支持** (ENV_MODE):
  - 默认模式 (`unified`): 推荐使用，一键部署维护简单
  - 兼容模式 (`legacy`): 支持回退到旧版独立环境配置
- ✅ **一键部署增强**: `setup_dev.sh` 优化提示与引导
- 📝 文档更新：新增[环境部署模式]章节说明

### 2026-02-04 (v0.3.4) - Qwen 推理优化

- ✅ Qwen 推理参数配置：LoRA 可选、模型路径自动切换
- ✅ 算法-模型路径映射：切换算法时自动更新默认模型路径
- ✅ 模型路径验证：选择 Qwen 但使用 ChatTS 路径时给出警告
- ✅ LoRA 适配器下拉框添加"无 (使用原始模型)"选项
- ✅ 隐藏冗余的 LoRA 模型类型选择框（自动跟随算法选择）
- ✅ 推理失败处理优化：OOM 等错误不再生成误导性全零结果
- ✅ 添加 `QwenInferenceError` 异常类，明确失败状态
- ✅ 调试输出改进：推理日志打印模型输出、保存到 `_qwen_output.txt`
- 🛠️ CSV 列名处理优化：非数值列自动回退到首个数值列

### 2026-02-03 (v0.3.3)

- ✅ 推理结果产物接入：自动写入 `metrics.json` / `segments.json` 与 DB 索引
- ✅ 候选队列：置信度筛选、排序与 TopK/低置信度/随机策略
- ✅ 审核队列：抽样、状态流转、批量审核与进度统计
- ✅ 训练导出仅保留 `approved`（单用户 MVP）
- ✅ 推理索引回填脚本（历史 CSV 批量入库）

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
| P1 | 数据资产管理 (DB) | 引入数据库取代文件系统管理，解决链接与一致性问题（v0.3.3 已完成基础索引） |
| P1 | 左侧统一文件管理器 | 所有模块共享文件浏览/选择 |
| P2 | 多区域联动交互 | 选中文件自动同步到其他模块 |
| P2 | 实时数据流 | WebSocket 推送任务状态 |

---

## MVP+ 产品化评估与新增需求

### 需求评估结论（摘要）
- **推理结果置信度指标**: 可行且高价值，建议统一为标准化指标产物，支持多指标扩展。
- **标注/推理筛选与排序**: 必要，需引入“候选队列”概念，避免全量标注。
- **微调结果评估**: 强烈建议训练后自动评估（黄金集），结果进入模型资产。
- **标注人工审核**: 建议独立为审核队列与流程，仅审核通过数据进入训练。

### 关键设计原则
- **指标统一**: 置信度/评分需要统一尺度与版本化定义。
- **资产化**: 推理结果/模型/标注有统一索引与元数据。
- **队列化**: 人工标注与审核进入队列管理。
- **可追溯**: 训练、评估、推理结果可关联回数据与模型版本。

### 推荐落地路径（3 阶段）
1. **阶段 1（最小闭环）**: 推理输出指标产物 + UI 置信度筛选 + 候选队列。
2. **阶段 2（训练评估）**: 训练后自动评估黄金集 + 指标展示/对比。
3. **阶段 3（审核模块）**: 审核队列 + 抽样策略 + 审核流程。

### 具体实现任务拆分（MVP+）
**A. 数据模型与索引层（DB 优先）✅**
- 新增表：`inference_results`、`segment_scores`、`metrics`、`review_queue`、`model_evals`
- 基于 `src/db/database.py` 扩展 SQLAlchemy 模型并初始化/迁移（`init_db()` 可创建）
- 新增一次性回填脚本（扫描 CSV 生成索引）

**B. 推理置信度产物 ✅**
- 推理结束后触发 `lb_eval` 计算置信度
- 输出 `metrics.json` / `segments.json`（与 CSV 同目录）
- 将摘要与段级指标写入 DB

**C. 推理结果筛选 & 标注候选队列 ✅**
- 标注/推理列表新增排序与阈值筛选（置信度）
- 增加“候选队列”视图（TopK/低置信度/随机）

**D. 训练后黄金集评估 ⏳**
- 训练结束自动跑 `eval_metrics.py`
- 结果写入 `model_evals` + 模型目录 `eval_results.json`
- UI 展示评估指标与模型对比

**E. 审核模块 ✅（MVP）**
- 审核状态流转：`pending → approved/rejected/needs_fix`
- 抽样策略：置信度优先/随机/覆盖率均衡
- 训练导出仅使用 `approved`
- 后续（P3）：用户权限管理（多用户隔离和权限控制）

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
