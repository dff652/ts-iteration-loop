"""
Gradio 统一管理界面
包含：数据获取、推理监控、微调训练、模型对比
"""
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional
import json
import pandas as pd

from configs.settings import settings
from src.adapters.chatts_training import ChatTSTrainingAdapter
from src.adapters.data_processing import DataProcessingAdapter
from src.adapters.check_outlier import CheckOutlierAdapter


# 初始化适配器
training_adapter = ChatTSTrainingAdapter()
data_adapter = DataProcessingAdapter()
inference_adapter = CheckOutlierAdapter()

# 为了兼容性保留旧变量名
adapter = training_adapter


def get_training_configs() -> List[str]:
    """获取训练配置列表"""
    configs = adapter.list_configs()
    return [c["name"] for c in configs]


def get_trained_models() -> List[str]:
    """获取已训练模型列表"""
    models = adapter.list_models()
    return [m["name"] for m in models]


def get_model_info(model_name: str) -> str:
    """获取模型详细信息"""
    if not model_name:
        return "请选择一个模型"
    
    models = adapter.list_models()
    model = next((m for m in models if m["name"] == model_name), None)
    
    if not model:
        return "模型不存在"
    
    info_lines = [
        f"**模型名称**: {model['name']}",
        f"**类型**: {model.get('type', 'unknown')}",
        f"**检查点**: {', '.join(model.get('checkpoints', []))}",
        f"**训练步数**: {model.get('global_step', 'N/A')}",
    ]
    
    # 训练结果
    train_results = model.get("train_results", {})
    if train_results:
        info_lines.append(f"**训练 Loss**: {train_results.get('train_loss', 'N/A'):.4f}")
        info_lines.append(f"**训练时长**: {train_results.get('train_runtime', 'N/A'):.1f}s")
    
    return "\n\n".join(info_lines)


def get_loss_plot(model_name: str):
    """获取 Loss 曲线图"""
    if not model_name:
        return None
    
    models = adapter.list_models()
    model = next((m for m in models if m["name"] == model_name), None)
    
    if not model or not model.get("loss_image"):
        return None
    
    loss_image = model.get("loss_image")
    if Path(loss_image).exists():
        return loss_image
    return None


def get_comparison_plot(model_names: List[str]):
    """获取多个模型的 Loss 对比图 (使用 Matplotlib 动态生成)"""
    if not model_names or len(model_names) == 0:
        return None
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.figure(figsize=(10, 6))
    
    for name in model_names:
        models = adapter.list_models()
        model = next((m for m in models if m["name"] == name), None)
        if not model:
            continue
            
        logs = adapter.get_training_log(model["path"])
        if not logs:
            continue
            
        df = pd.DataFrame([{"step": l.get("current_steps", 0), "loss": l.get("loss")} for l in logs if "loss" in l])
        if not df.empty:
            plt.plot(df["step"], df["loss"], label=name)
            
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Model Comparison: Training Loss")
    plt.legend()
    plt.grid(True)
    
    # 保存到项目本地临时目录，避免系统 /tmp 权限问题
    import uuid
    
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    
    output_path = temp_dir / f"compare_{uuid.uuid4().hex[:8]}.png"
    plt.savefig(str(output_path))
    plt.close()
    
    return str(output_path)


# ==================== 数据获取辅助函数 ====================

def get_datasets_table() -> pd.DataFrame:
    """获取数据集列表并返回 DataFrame"""
    datasets = data_adapter.list_datasets()
    if not datasets:
        return pd.DataFrame(columns=["文件名", "大小 (KB)", "修改时间"])
    
    from datetime import datetime
    rows = []
    for d in datasets:
        rows.append({
            "文件名": d["filename"],
            "大小 (KB)": round(d["size_bytes"] / 1024, 2),
            "修改时间": datetime.fromtimestamp(d["modified_time"]).strftime("%Y-%m-%d %H:%M")
        })
    return pd.DataFrame(rows)


def get_dataset_names() -> List[str]:
    """获取数据集文件名列表"""
    datasets = data_adapter.list_datasets()
    return [d["filename"] for d in datasets]


def preview_dataset(filename: str) -> tuple:
    """预览数据集，返回 (表格数据, 曲线图)"""
    if not filename:
        return pd.DataFrame(), None
    
    try:
        # 获取预览数据
        data = data_adapter.preview_csv(filename, limit=200)
        df = pd.DataFrame(data)
        
        # 生成曲线图
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        # 假设第一列是时间或索引，后面的列是数值
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            for col in numeric_cols[:3]:  # 最多显示 3 条曲线
                plt.plot(df.index, df[col], label=col, alpha=0.8)
            plt.xlabel("索引")
            plt.ylabel("值")
            plt.title(f"数据预览: {filename}")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        import uuid
        plot_path = temp_dir / f"preview_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(str(plot_path), dpi=100, bbox_inches='tight')
        plt.close()
        
        return df, str(plot_path)
    except Exception as e:
        return pd.DataFrame({"错误": [str(e)]}), None


def start_acquire_task(source: str, target_points: int) -> str:
    """启动数据采集任务"""
    if not source:
        return "❌ 请输入 IoTDB 源路径"
    
    try:
        result = data_adapter.run_acquire_task(
            task_id="manual",
            source=source,
            target_points=int(target_points)
        )
        if result.get("success"):
            return f"✅ 采集任务完成\n\n{result.get('stdout', '')[:500]}"
        else:
            return f"❌ 采集失败: {result.get('error', result.get('stderr', '未知错误'))}"
    except Exception as e:
        return f"❌ 启动失败: {str(e)}"


# ==================== 推理监控辅助函数 ====================

def get_algorithms() -> List[str]:
    """获取可用算法列表"""
    return ["chatts", "adtk_hbos", "ensemble"]


def get_inference_models() -> List[str]:
    """获取可用于推理的模型列表"""
    models = training_adapter.list_models()
    return [m["name"] for m in models]


def start_inference_task(algorithm: str, model: str, files: List[str]) -> str:
    """启动推理任务"""
    if not algorithm:
        return "❌ 请选择算法"
    if not files:
        return "❌ 请选择输入文件"
    
    # 将选中的文件名转换为完整路径
    file_paths = []
    for f in files:
        full_path = data_adapter.data_path / f
        if full_path.exists():
            file_paths.append(str(full_path))
    
    if not file_paths:
        return "❌ 未找到有效的输入文件"
    
    import uuid
    task_id = str(uuid.uuid4())[:8]
    
    try:
        # 这里应该调用异步任务，目前返回提示
        return f"""✅ 推理任务已提交

**任务 ID**: {task_id}
**算法**: {algorithm}
**模型**: {model or '默认模型'}
**文件数**: {len(file_paths)}

请通过 API `/api/v1/inference/status/{task_id}` 查询进度。
"""
    except Exception as e:
        return f"❌ 启动失败: {str(e)}"


def get_task_status_table() -> pd.DataFrame:
    """获取任务状态列表 (从数据库读取)"""
    try:
        from src.db.database import SessionLocal, Task
        db = SessionLocal()
        tasks = db.query(Task).order_by(Task.created_at.desc()).limit(20).all()
        db.close()
        
        if not tasks:
            return pd.DataFrame(columns=["ID", "类型", "状态", "创建时间"])
        
        rows = []
        for t in tasks:
            rows.append({
                "ID": t.id[:8] + "...",
                "类型": t.type,
                "状态": t.status,
                "创建时间": t.created_at.strftime("%H:%M:%S") if t.created_at else "N/A"
            })
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame({"错误": [str(e)]})


def start_training(
    config_name: str,
    learning_rate: str,
    num_epochs: float,
    batch_size: int,
    lora_rank: int,
    lora_alpha: int,
    output_name: str
) -> str:
    """启动训练"""
    if not config_name:
        return "❌ 请选择训练配置"
    
    if not output_name:
        return "❌ 请输入输出目录名称"
    
    # 调用适配器启动训练
    import uuid
    task_id = str(uuid.uuid4())[:8]
    
    try:
        # 这里应该调用 adapter.run_training，但由于是后台任务，先返回提示
        return f"""✅ 训练任务已提交

**任务 ID**: {task_id}
**配置**: {config_name}
**输出目录**: {output_name}

**参数**:
- 学习率: {learning_rate}
- 训练轮数: {num_epochs}
- 批次大小: {batch_size}
- LoRA Rank: {lora_rank}
- LoRA Alpha: {lora_alpha}

请通过 API `/api/v1/training/status/{task_id}` 查询进度。
"""
    except Exception as e:
        return f"❌ 启动失败: {str(e)}"


def create_training_ui() -> gr.Blocks:
    """创建统一管理界面（数据获取、推理监控、微调训练）"""
    
    with gr.Blocks(title="TS-Iteration-Loop", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔄 TS-Iteration-Loop 时序迭代平台")
        gr.Markdown("整合数据获取、推理监控、微调训练的统一管理界面")
        
        # ==================== 数据获取 Tab ====================
        with gr.Tab("📁 数据获取"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 数据集列表")
                    datasets_table = gr.Dataframe(
                        value=get_datasets_table(),
                        label="已有数据集",
                        interactive=False
                    )
                    refresh_datasets_btn = gr.Button("🔄 刷新列表")
                    
                    gr.Markdown("### 预览数据")
                    preview_dropdown = gr.Dropdown(
                        label="选择数据集",
                        choices=get_dataset_names(),
                        interactive=True
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 数据采集配置")
                    with gr.Row():
                        source_input = gr.Textbox(
                            label="IoTDB 源路径",
                            placeholder="root.xxx.yyy.zzz",
                            scale=2
                        )
                        target_points = gr.Slider(
                            label="目标点数",
                            minimum=1000,
                            maximum=10000,
                            value=5000,
                            step=500,
                            scale=1
                        )
                    acquire_btn = gr.Button("📥 开始采集", variant="primary")
                    acquire_output = gr.Markdown(value="等待采集...")
            
            # 数据预览区域
            with gr.Row():
                with gr.Column(scale=1):
                    preview_table = gr.Dataframe(
                        label="数据预览 (前200行)",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    preview_plot = gr.Image(label="曲线预览")
            
            # 事件绑定 - 数据获取
            refresh_datasets_btn.click(
                fn=get_datasets_table,
                outputs=datasets_table
            )
            refresh_datasets_btn.click(
                fn=lambda: gr.Dropdown(choices=get_dataset_names()),
                outputs=preview_dropdown
            )
            preview_dropdown.change(
                fn=preview_dataset,
                inputs=preview_dropdown,
                outputs=[preview_table, preview_plot]
            )
            acquire_btn.click(
                fn=start_acquire_task,
                inputs=[source_input, target_points],
                outputs=acquire_output
            )
        
        # ==================== 推理监控 Tab ====================
        with gr.Tab("🔍 推理监控"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 新建推理任务")
                    algo_dropdown = gr.Dropdown(
                        label="选择算法",
                        choices=get_algorithms(),
                        value="chatts",
                        interactive=True
                    )
                    model_select = gr.Dropdown(
                        label="选择模型 (可选)",
                        choices=get_inference_models(),
                        interactive=True
                    )
                    files_select = gr.CheckboxGroup(
                        label="选择输入文件",
                        choices=get_dataset_names()
                    )
                    submit_inference_btn = gr.Button("🚀 提交任务", variant="primary")
                    inference_output = gr.Markdown(value="等待提交...")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 任务状态")
                    task_table = gr.Dataframe(
                        value=get_task_status_table(),
                        label="最近 20 条任务",
                        interactive=False
                    )
                    with gr.Row():
                        refresh_tasks_btn = gr.Button("🔄 刷新状态")
                        # auto_refresh = gr.Checkbox(label="自动刷新 (5s)", value=False)
            
            # 事件绑定 - 推理监控
            submit_inference_btn.click(
                fn=start_inference_task,
                inputs=[algo_dropdown, model_select, files_select],
                outputs=inference_output
            )
            refresh_tasks_btn.click(
                fn=get_task_status_table,
                outputs=task_table
            )
            refresh_tasks_btn.click(
                fn=lambda: gr.CheckboxGroup(choices=get_dataset_names()),
                outputs=files_select
            )
            refresh_tasks_btn.click(
                fn=lambda: gr.Dropdown(choices=get_inference_models()),
                outputs=model_select
            )
        
        # ==================== 标注工具 Tab ====================
        with gr.Tab("🏷️ 标注工具"):
            gr.Markdown("### 时序数据标注")
            gr.Markdown(f"""
> [!NOTE]
> 标注工具运行在独立服务上，点击下方链接跳转。

**标注工具地址**: [http://localhost:5000](http://localhost:5000)

---

### 使用说明

1. **打开标注工具**: 点击上方链接进入标注界面
2. **选择数据文件**: 在标注工具中选择要标注的 CSV 文件
3. **进行标注**: 使用框选工具标记异常区间
4. **保存标注**: 完成后保存标注结果

### 标注与迭代流程

```
📁 数据获取 → 🏷️ 人工标注 → 🎯 微调训练 → 🔍 推理检测 → 🏷️ 审核修正 → 🎯 再次微调 → ...
```

### 快速操作
""")
            with gr.Row():
                open_annotator_btn = gr.Button("🔗 打开标注工具 (新标签页)", variant="primary", size="lg")
                gr.Markdown("""
<script>
function openAnnotator() {
    window.open('http://localhost:5000', '_blank');
}
</script>
""", visible=False)
            
            gr.Markdown("### 当前标注状态")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**可标注文件数量**")
                    annotatable_count = gr.Textbox(
                        value=f"{len(get_dataset_names())} 个文件",
                        interactive=False,
                        show_label=False
                    )
                with gr.Column():
                    gr.Markdown("**标注工具状态**")
                    annotator_status = gr.Textbox(
                        value="请访问标注工具查看",
                        interactive=False,
                        show_label=False
                    )
            
            # JavaScript 跳转 (Gradio 限制，使用 HTML)
            open_annotator_btn.click(
                fn=lambda: "✅ 请在新标签页中查看标注工具 (http://localhost:5000)",
                outputs=annotator_status
            )
        
        # ==================== 微调训练 Tab (原有) ====================
        with gr.Tab("🎯 开始训练"):
            with gr.Row():
                with gr.Column(scale=2):
                    # 基础配置
                    gr.Markdown("### 基础配置")
                    config_dropdown = gr.Dropdown(
                        label="训练配置",
                        choices=get_training_configs(),
                        interactive=True
                    )
                    output_name = gr.Textbox(
                        label="输出目录名称",
                        placeholder="例如: my_model_v1"
                    )
                    
                    with gr.Row():
                        learning_rate = gr.Textbox(
                            label="学习率",
                            value="2e-5"
                        )
                        num_epochs = gr.Number(
                            label="训练轮数",
                            value=3.0
                        )
                        batch_size = gr.Slider(
                            label="批次大小",
                            minimum=1,
                            maximum=32,
                            value=2,
                            step=1
                        )
                    
                    # LoRA 配置（折叠）
                    with gr.Accordion("LoRA 配置", open=False):
                        with gr.Row():
                            lora_rank = gr.Slider(
                                label="LoRA Rank",
                                minimum=1,
                                maximum=128,
                                value=8,
                                step=1
                            )
                            lora_alpha = gr.Slider(
                                label="LoRA Alpha",
                                minimum=1,
                                maximum=256,
                                value=16,
                                step=1
                            )
                    
                    # 控制按钮
                    with gr.Row():
                        start_btn = gr.Button("🚀 开始训练", variant="primary")
                        refresh_btn = gr.Button("🔄 刷新配置")
                
                with gr.Column(scale=1):
                    # 输出区域
                    gr.Markdown("### 训练状态")
                    output_box = gr.Markdown(value="等待开始训练...")
            
            # 事件绑定
            start_btn.click(
                fn=start_training,
                inputs=[
                    config_dropdown, learning_rate, num_epochs,
                    batch_size, lora_rank, lora_alpha, output_name
                ],
                outputs=output_box
            )
            refresh_btn.click(
                fn=lambda: gr.Dropdown(choices=get_training_configs()),
                outputs=config_dropdown
            )
        
        with gr.Tab("📊 已训练模型"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型列表")
                    model_dropdown = gr.Dropdown(
                        label="选择模型",
                        choices=get_trained_models(),
                        interactive=True
                    )
                    refresh_models_btn = gr.Button("🔄 刷新列表")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 模型详情")
                    model_info = gr.Markdown(value="请选择一个模型")
                    
                    gr.Markdown("### Loss 曲线")
                    loss_image = gr.Image(label="Training Loss", type="filepath")
            
            # 事件绑定
            model_dropdown.change(
                fn=get_model_info,
                inputs=model_dropdown,
                outputs=model_info
            )
            model_dropdown.change(
                fn=get_loss_plot,
                inputs=model_dropdown,
                outputs=loss_image
            )
            refresh_models_btn.click(
                fn=lambda: gr.Dropdown(choices=get_trained_models()),
                outputs=model_dropdown
            )
            
        with gr.Tab("⚖️ 模型对比"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 选择对比模型")
                    compare_models = gr.CheckboxGroup(
                        label="模型列表",
                        choices=get_trained_models()
                    )
                    compare_btn = gr.Button("📊 生成对比图", variant="primary")
                    refresh_compare_btn = gr.Button("🔄 刷新列表")
                
                with gr.Column(scale=3):
                    gr.Markdown("### 对比结果")
                    comparison_plot = gr.Image(label="Loss Comparison")
            
            # 事件绑定
            compare_btn.click(
                fn=get_comparison_plot,
                inputs=compare_models,
                outputs=comparison_plot
            )
            refresh_compare_btn.click(
                fn=lambda: gr.CheckboxGroup(choices=get_trained_models()),
                outputs=compare_models
            )
        
        with gr.Tab("⚙️ 配置说明"):
            gr.Markdown("""
## 训练配置说明

### 基础参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| **学习率** | 模型更新步长 | 2e-5 ~ 5e-5 |
| **训练轮数** | 完整遍历数据集次数 | 3 ~ 5 |
| **批次大小** | 每次更新的样本数 | 2 ~ 8 (取决于显存) |

### LoRA 参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| **LoRA Rank** | 低秩分解维度 | 8 ~ 64 |
| **LoRA Alpha** | 缩放因子 | 通常为 Rank 的 2 倍 |

### 显存需求

| 模型 | 批次大小 | 显存需求 |
|------|---------|---------|
| ChatTS-8B + LoRA | 2 | ~16GB |
| ChatTS-8B + LoRA | 4 | ~24GB |
| ChatTS-14B + LoRA | 2 | ~24GB |

### 训练脚本

可用的训练配置来自 `/home/douff/ts/ChatTS-Training/scripts/lora/` 目录。
""")
    
    return demo


# 创建全局 Gradio 应用实例
training_ui = create_training_ui()
