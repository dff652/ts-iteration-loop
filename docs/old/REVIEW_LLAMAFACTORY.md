# Phase 1 代码 Review 与 LlamaFactory 调研报告

## 一、Phase 1 代码 Review

### 1.1 整体评估 ✅

| 评分项 | 评分 | 说明 |
|--------|------|------|
| 代码结构 | 9/10 | API 与适配器分离清晰，职责明确 |
| 可维护性 | 8/10 | 模块化设计，易于扩展 |
| 错误处理 | 7/10 | 有基础异常处理，可增强 |
| 文档完整性 | 8/10 | 函数有 docstring，可增加类型提示 |

### 1.2 优点

- **分层架构合理**：`api/` → `adapters/` → 外部项目
- **统一响应格式**：使用 `ApiResponse` 和 `TaskResponse` 标准化
- **后台任务支持**：使用 FastAPI `BackgroundTasks` 处理长时间运行任务
- **数据库集成**：SQLite + SQLAlchemy 用于任务状态持久化

### 1.3 改进建议

| 问题 | 文件 | 建议 |
|------|------|------|
| 任务状态未更新 | `api/data.py` | 后台任务完成后应更新数据库任务状态 |
| 适配器是全局单例 | 所有 API | 考虑使用依赖注入，便于测试 |
| 缺少日志 | 所有文件 | 添加 logging 记录关键操作 |
| 进程管理 | `adapters/chatts_training.py` | 进程 dict 应持久化，服务重启后丢失 |

### 1.4 待完善功能

- [ ] 任务状态回调机制（后台任务完成后更新DB）
- [ ] 日志记录系统
- [ ] 配置热更新
- [ ] 健康检查增强（检查外部服务可用性）

---

## 二、LlamaFactory WebUI 调研

### 2.1 技术栈

| 组件 | 技术 |
|------|------|
| UI 框架 | **Gradio** |
| 入口 | `src/webui.py` → `llamafactory.webui.interface.create_ui()` |
| 核心组件 | `components/train.py` (448行，16KB) |

### 2.2 界面架构

```
LlamaFactory WebUI
├── Top Bar (模型选择、微调类型)
├── Train Tab          ← 核心功能
│   ├── 基础配置 (learning_rate, epochs, batch_size)
│   ├── LoRA 配置 (rank, alpha, dropout)
│   ├── Freeze 配置
│   ├── RLHF 配置
│   ├── DeepSpeed 配置
│   ├── 训练控制 (Start/Stop/Save/Load)
│   └── 输出区域 (进度条 + loss 图表)
├── Evaluate & Predict Tab
├── Chat Tab
└── Export Tab
```

### 2.3 可借鉴的设计

| 特性 | 实现方式 | 借鉴价值 |
|------|----------|----------|
| **参数分组** | 使用 `gr.Accordion` 折叠高级选项 | ⭐⭐⭐ 减少复杂度 |
| **实时监控** | `loss_viewer = gr.Plot()` + 定时刷新 | ⭐⭐⭐ 训练可视化 |
| **进度条** | `progress_bar = gr.Slider()` 隐藏交互 | ⭐⭐ 进度反馈 |
| **配置保存/加载** | `arg_save_btn` / `arg_load_btn` | ⭐⭐⭐ 配置复用 |
| **命令预览** | `cmd_preview_btn` 显示实际执行命令 | ⭐⭐ 调试友好 |
| **多语言** | `locales.py` (108KB) | ⭐ 国际化 |

### 2.4 Phase 2 实现建议

基于 LlamaFactory 调研，建议 Phase 2 采用以下方案：

| 组件 | 建议 |
|------|------|
| UI 框架 | **Gradio** (与 LlamaFactory 保持一致，快速开发) |
| 训练界面 | 模仿 LlamaFactory 的分组折叠设计 |
| 实时监控 | 使用 Gradio Plot + WebSocket 推送 loss 曲线 |
| 配置管理 | 支持保存/加载训练配置 JSON |

### 2.5 简化版 Train Tab 设计

```python
# 建议的 Gradio 组件结构
with gr.Tab("微调训练"):
    # 基础配置
    with gr.Row():
        config_name = gr.Dropdown(label="训练配置")
        dataset = gr.Dropdown(label="训练数据集")
    
    with gr.Row():
        learning_rate = gr.Textbox(label="学习率", value="2e-5")
        epochs = gr.Number(label="训练轮数", value=3)
        batch_size = gr.Number(label="批次大小", value=2)
    
    # 高级选项 (折叠)
    with gr.Accordion("LoRA 配置", open=False):
        lora_rank = gr.Slider(label="LoRA Rank", value=8)
        lora_alpha = gr.Slider(label="LoRA Alpha", value=16)
    
    # 控制按钮
    with gr.Row():
        start_btn = gr.Button("开始训练", variant="primary")
        stop_btn = gr.Button("停止", variant="stop")
    
    # 输出区域
    with gr.Row():
        output_log = gr.Textbox(label="训练日志", lines=10)
        loss_plot = gr.Plot(label="Loss 曲线")
```

---

## 三、下一步行动

1. **修复 Phase 1 问题**：添加任务状态更新、日志系统
2. **创建 Gradio 微调界面**：基于上述设计
3. **集成训练监控**：实时 loss 曲线、进度显示
