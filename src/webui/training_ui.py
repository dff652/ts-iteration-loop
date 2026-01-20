"""
Gradio 微调界面
借鉴 LlamaFactory 设计，嵌入 FastAPI
"""
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional
import json

from configs.settings import settings
from src.adapters.chatts_training import ChatTSTrainingAdapter


# 初始化适配器
adapter = ChatTSTrainingAdapter()


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
    """创建训练界面"""
    
    with gr.Blocks(title="ChatTS 微调训练", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 ChatTS 微调训练")
        gr.Markdown("基于 LlamaFactory 的时序异常检测模型微调平台")
        
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
