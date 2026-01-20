# 开发文档

## 项目信息

- **项目名称**: TS-Iteration-Loop
- **开发开始日期**: 2026-01-20
- **当前阶段**: Phase 3 完成 ✅ (MVP 达成)

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

## 外部依赖

| 项目 | 路径 | 用途 |
|------|------|------|
| Data-Processing | `/home/douff/ts/Data-Processing` | 数据采集、降采样 |
| timeseries-annotator-v2 | `/home/douff/ts/timeseries-annotator-v2` | Web 标注界面 |
| ChatTS-Training | `/home/douff/ts/ChatTS-Training` | 模型微调 |
| check_outlier | `/home/douff/ilabel/check_outlier` | 批量推理 |

---

## 开发日志

### 2026-01-20

- ✅ 完成技术可行性评估和开发规划
- ✅ 创建项目结构和 API 路由
- ✅ 实现 Gradio 微调界面
- ✅ 实现推理结果自动反馈闭环
- ✅ 添加数据获取、推理监控、标注工具 Tab
- ✅ 实现迭代版本管理 API
- ✅ **MVP 完成**
