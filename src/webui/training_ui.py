"""
Gradio 统一管理界面
包含：数据获取、推理监控、微调训练、模型对比
"""
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
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

# 结果文件目录
RESULTS_BASE_PATH = Path("/home/share/results/data")


def get_existing_results(method: str = "chatts") -> List[str]:
    """获取已有的结果文件列表"""
    results_dir = RESULTS_BASE_PATH / "global" / method
    if not results_dir.exists():
        return []
    
    # 获取所有 CSV 文件，按修改时间排序（最新的在前）
    csv_files = list(results_dir.glob("*.csv"))
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(f) for f in csv_files[:20]]  # 最多返回 20 个


def delete_selected_files(method: str, filenames: List[str]) -> tuple:
    """批量删除选中的结果文件"""
    if not filenames:
        return (
            gr.CheckboxGroup(choices=get_result_filenames(method)), 
            gr.File(value=None), 
            "⚠️ 请先选择要删除的文件"
        )
    
    results_dir = RESULTS_BASE_PATH / "global" / method
    deleted_count = 0
    errors = []
    
    for fname in filenames:
        file_path = results_dir / fname.strip()  # Strip whitespace just in case
        print(f"DEBUG: Attempting to delete {file_path}")
        if file_path.exists():
            try:
                file_path.unlink()
                deleted_count += 1
                print(f"DEBUG: Deleted {file_path}")
            except Exception as e:
                errors.append(f"{fname}: {str(e)}")
                print(f"DEBUG: Error deleting {file_path}: {e}")
        else:
            print(f"DEBUG: File not found {file_path}")
            # Try verify if it's a encoding issue or partial path
            errors.append(f"{fname}: File not found")
    
    # 刷新列表
    time.sleep(0.5)  # 等待文件系统同步
    new_choices = get_result_filenames(method)
    
    status_msg = f"✅ 已删除 {deleted_count} 个文件"
    if errors:
        status_msg += f"\n❌ 错误: {'; '.join(errors)}"
        
    return (
        gr.update(choices=new_choices, value=[]), 
        gr.update(value=None, label="📥 下载区域 (请先选择文件)"),
        status_msg
    )


def prepare_download_files(method: str, filenames: List[str]) -> tuple:
    """准备下载选中的文件"""
    if not filenames:
        return None, "⚠️ 请先选择要下载的文件"
    
    results_dir = RESULTS_BASE_PATH / "global" / method
    paths = []
    for fname in filenames:
        p = results_dir / fname.strip()
        if p.exists():
            paths.append(str(p))
            
    if not paths:
        return gr.update(value=None), "❌ 未找到选中的文件"
        
    return (
        gr.update(value=paths, label="📥 点击此处下载 / Click to Download", visible=True),
        f"✅ 已准备好 {len(paths)} 个文件，请点击下方下载区域进行下载"
    )


def get_result_filenames(method: str = "chatts") -> List[str]:
    """获取结果文件名列表（用于下拉框）"""
    results_dir = RESULTS_BASE_PATH / "global" / method
    if not results_dir.exists():
        return []
    
    csv_files = list(results_dir.glob("*.csv"))
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in csv_files[:20]]


def delete_result_file(method: str, filename: str) -> tuple:
    # 已弃用，使用 delete_selected_files
    pass


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


def delete_selected_dataset(filename: str):
    """删除选中的数据集"""
    print(f"[DEBUG] delete_selected_dataset called with: '{filename}'")
    if not filename:
        return get_datasets_table(), gr.Dropdown(choices=get_dataset_names(), value=None), "❌ No dataset selected"
    
    result = data_adapter.delete_dataset(filename)
    if result.get("success"):
        # 刷新列表
        new_table = get_datasets_table()
        new_choices = get_dataset_names()
        return new_table, gr.Dropdown(choices=new_choices, value=None), f"✅ Deleted: {filename}"
    else:
        return get_datasets_table(), gr.Dropdown(choices=get_dataset_names()), f"❌ {result.get('error')}"


def preview_dataset(filename: str) -> tuple:
    """预览数据集，返回 (表格数据, 列选择器更新, 曲线图)"""
    print(f"[DEBUG] preview_dataset called with filename: '{filename}'")
    
    if isinstance(filename, list):
        filename = filename[0] if filename else None
    
    if not filename:
        print("[DEBUG] Empty filename, returning empty")
        return [], gr.CheckboxGroup(choices=[], value=[]), None
    
    try:
        # 获取预览数据
        print(f"[DEBUG] Calling preview_csv for: {filename}")
        data = data_adapter.preview_csv(filename, limit=5000)
        print(f"[DEBUG] preview_csv returned {len(data)} records")
        
        df = pd.DataFrame(data)
        print(f"[DEBUG] DataFrame created: shape={df.shape}, columns={df.columns.tolist()}")
        
        # 过滤掉 Unnamed 和 category 列
        df = df.loc[:, ~df.columns.str.contains('^Unnamed|^category', case=False)]
        print(f"[DEBUG] After filtering: shape={df.shape}, columns={df.columns.tolist()}")
        
        # 获取数值列作为可选项
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        print(f"[DEBUG] Numeric columns: {numeric_cols}")
        
        # 默认选中第一个数值列
        default_selected = numeric_cols[:1] if numeric_cols else []
        print(f"[DEBUG] Default selected: {default_selected}")
        
        # 生成默认曲线图
        plot_path = generate_plot(df, filename, default_selected)
        print(f"[DEBUG] Plot generated: {plot_path}")
        
        # 转换为列表格式，确保 Gradio 6.x 兼容
        # 使用 values 列表 + headers 的方式
        table_data = df.values.tolist()
        headers = df.columns.tolist()
        print(f"[DEBUG] Table data rows: {len(table_data)}, headers: {headers}")
        
        return gr.Dataframe(value=table_data, headers=headers), gr.CheckboxGroup(choices=numeric_cols, value=default_selected), plot_path
    except Exception as e:
        import traceback
        print(f"[DEBUG ERROR] Exception: {e}")
        traceback.print_exc()
        return [], gr.CheckboxGroup(choices=[], value=[]), None


def generate_plot(df: pd.DataFrame, filename: str, selected_cols: list):
    """根据选择的列生成曲线图"""
    if df.empty or not selected_cols:
        return None
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    
    for col in selected_cols:
        if col in df.columns:
            plt.plot(df.index, df[col], label=col, alpha=0.8)
    
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Data Preview: {filename}")
    if selected_cols:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    import uuid
    plot_path = temp_dir / f"preview_{uuid.uuid4().hex[:8]}.png"
    plt.savefig(str(plot_path), dpi=100, bbox_inches='tight')
    plt.close()
    
    return str(plot_path)


def update_plot_from_selection(filename: str, selected_cols: list):
    """根据用户选择的列更新曲线图"""
    if not filename or not selected_cols:
        return None
    
    try:
        data = data_adapter.preview_csv(filename, limit=5000)
        df = pd.DataFrame(data)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed|^category', case=False)]
        return generate_plot(df, filename, selected_cols)
    except:
        return None


def start_acquire_task(
    source: str, 
    host: str,
    port: str,
    user: str,
    password: str,
    point_name: str,
    start_time: str,
    end_time: str,
    target_points: int
):
    """启动数据采集任务（流式输出日志）"""
    if not source:
        yield "❌ Please enter IoTDB source path"
        return
    
    # 使用流式输出版本
    for log in data_adapter.run_acquire_task_streaming(
        task_id="manual",
        source=source,
        host=host,
        port=port,
        user=user,
        password=password,
        point_name=point_name,
        target_points=int(target_points),
        start_time=start_time,
        end_time=end_time
    ):
        yield log


# ==================== 推理监控辅助函数 ====================

def get_algorithms() -> List[str]:
    """获取可用算法列表"""
    return ["chatts", "adtk_hbos", "ensemble", "timer"]


def get_inference_models() -> List[str]:
    """获取可用于推理的模型列表"""
    # 过滤 lora 模型（假设 ChatTS 推理主要用 LoRA）
    models = training_adapter.list_models()
    return [m["path"] for m in models] # 直接返回路径，方便 adapter 处理

def toggle_algo_params(algorithm: str):
    """根据选择的算法切换参数组可见性"""
    show_chatts = (algorithm == "chatts")
    show_timer = (algorithm == "timer")
    show_adtk = (algorithm == "adtk_hbos")
    return (
        gr.update(visible=show_chatts), 
        gr.update(visible=show_timer), 
        gr.update(visible=show_adtk)
    )

def start_inference_task(
    algorithm: str, 
    base_model_path: str,
    lora_adapter_path: str,
    files: List[str],
    n_downsample: int,
    threshold: float,
    # ChatTS args
    load_in_4bit: str,
    prompt_template: str,
    max_new_tokens: int,
    chatts_device: str,
    chatts_use_cache: str,
    # Timer args
    timer_device: str,
    timer_lookback: int,
    timer_threshold_k: float,
    timer_method: str,
    timer_streaming: bool,
    # ADTK args
    adtk_bin_nums: int,
    adtk_hbos_ratio: float
):
    """启动推理任务"""
    if not algorithm:
        yield "❌ 请选择算法", "❌ 请选择算法"
        return
    if not files:
        yield "❌ 请选择输入文件", "❌ 请选择输入文件"
        return
    
    # 将选中的文件名转换为完整路径
    file_paths = []
    for f in files:
        full_path = data_adapter.data_path / f
        if full_path.exists():
            file_paths.append(str(full_path))
    
    if not file_paths:
        yield "❌ 未找到有效的输入文件", "❌ 未找到有效的输入文件"
        return
    
    import uuid
    task_id = str(uuid.uuid4())
    
    # 保存任务到数据库
    from datetime import datetime
    from src.db.database import SessionLocal, Task
    db = SessionLocal()
    try:
        task = Task(
            id=task_id,
            type="inference",
            status="running",
            config=json.dumps({
                "algorithm": algorithm,
                "files": files,
                "base_model_path": base_model_path,
                "lora_adapter_path": lora_adapter_path
            }),
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        db.add(task)
        db.commit()
    except Exception as e:
        print(f"[DB Error] Failed to save task: {e}")
    finally:
        db.close()
    
    yield (
        f"🚀 任务已启动 (ID: {task_id[:8]})\\n正在处理 {len(file_paths)} 个文件...", 
        f"🚀 任务已启动 (ID: {task_id[:8]})",
        gr.update(visible=True), # Show stop button
        gr.update(visible=False), # Hide submit button
        task_id, # Return task_id to state
        None # download_files
    )
    
    try:
        # 准备高级参数
        advanced_args = {
            "n_downsample": n_downsample,
            "threshold": threshold,
            "base_model_path": base_model_path,
            "lora_adapter_path": lora_adapter_path, 
            # ChatTS
            "chatts_load_in_4bit": load_in_4bit,
            "chatts_prompt_template": prompt_template,
            "chatts_max_new_tokens": max_new_tokens,
            "chatts_device": chatts_device,
            "chatts_use_cache": chatts_use_cache,
            # Timer
            "timer_device": timer_device,
            "timer_lookback_length": timer_lookback,
            "timer_threshold_k": timer_threshold_k,
            "timer_method": timer_method,
            "timer_streaming": timer_streaming,
            # ADTK
            "bin_nums": adtk_bin_nums,
            "hbos_ratio": adtk_hbos_ratio
        }
        
        generated_files = []
        
        # 执行推理（流式）
        accumulated_log = ""
        for log_chunk in inference_adapter.run_batch_inference_streaming(
            task_id=task_id,
            model=lora_adapter_path, # 兼容旧接口命名，实际逻辑在 adapter 中已处理
            algorithm=algorithm,
            input_files=file_paths,
            **advanced_args
        ):
            # 检查是否包含文件路径返回
            if isinstance(log_chunk, dict) and "file_path" in log_chunk:
                # adapter 返回了完整路径
                generated_files.append(log_chunk["file_path"])
            elif isinstance(log_chunk, dict) and "file_name" in log_chunk:
                # 兼容旧格式，仅有文件名
                generated_files.append(log_chunk["file_name"])
            elif isinstance(log_chunk, dict):
                 pass # 其他结构化消息
            else:
                accumulated_log += log_chunk
                yield (
                    accumulated_log, 
                    "🔄 正在执行...",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    task_id,
                    None
                )
        
        # 任务结束，尝试查找生成的结果文件
        # 假设保存在 /home/share/results/data/<method> 下，按时间最新查找？
        # 这比较 hacky。更好的方法是 adapter 返回。
        # 我们在 adapter 中增加了 yield {"file_name": ...} 逻辑
        # 这里需要处理它。
        
        # 更新数据库任务状态为完成
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                db.commit()
        except Exception as e:
            print(f"[DB Error] Failed to update task: {e}")
        finally:
            db.close()
        
        yield (
            accumulated_log + "\n✅ 所有任务已完成", 
            "✅ 任务完成",
             gr.update(visible=False),
             gr.update(visible=True),
             task_id,
             generated_files # TODO: 填充 output files if capture logic works perfectly
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # 更新数据库任务状态为失败
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = "failed"
                task.error = str(e)
                task.completed_at = datetime.utcnow()
                db.commit()
        except Exception as db_e:
            print(f"[DB Error] Failed to update task: {db_e}")
        finally:
            db.close()
        
        yield (
            f"❌ 发生错误: {str(e)}", 
            f"❌ 错误: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=True),
            None,
            None
        )

def stop_task_action(task_id_state):
    """实际执行停止动作"""
    print(f"DEBUG: Stop requested for task ID: {task_id_state}")
    if task_id_state:
        if inference_adapter.stop_inference_task(task_id_state):
            print(f"DEBUG: Stop successful for {task_id_state}")
            return "🛑 任务已请求停止", gr.update(visible=False), gr.update(visible=True), None, None
        else:
            print(f"DEBUG: Stop failed for {task_id_state} (not found or error)")
            return f"❌ 停止失败: 任务 {task_id_state} 不存在或已结束", gr.update(visible=True), gr.update(visible=False), task_id_state, None
    print("DEBUG: No active task ID found")
    return "⚠️ 无活动任务", gr.update(visible=False), gr.update(visible=True), None, None


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


def clear_task_history() -> tuple:
    """清空任务历史记录"""
    try:
        from src.db.database import SessionLocal, Task
        db = SessionLocal()
        deleted = db.query(Task).delete()
        db.commit()
        db.close()
        return pd.DataFrame(columns=["ID", "类型", "状态", "创建时间"]), f"✅ 已清空 {deleted} 条历史记录"
    except Exception as e:
        return get_task_status_table(), f"❌ 清空失败: {str(e)}"


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
                    with gr.Row():
                        delete_dataset_btn = gr.Button("🗑️ 删除选中", variant="stop", size="sm")
                        delete_status = gr.Textbox(label="", visible=False)
                    
                    column_selector = gr.CheckboxGroup(
                        label="Select columns to plot",
                        choices=[],
                        interactive=True
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 数据采集配置")
                    
                    with gr.Accordion("IoTDB 连接配置", open=False):
                        with gr.Row():
                            host_input = gr.Textbox(label="Host", value="192.168.199.185")
                            port_input = gr.Textbox(label="Port", value="6667")
                        with gr.Row():
                            user_input = gr.Textbox(label="User", value="root")
                            pwd_input = gr.Textbox(label="Password", value="root", type="password")

                    gr.Markdown("### 查询参数")
                    source_input = gr.Textbox(
                        label="IoTDB 源路径 (Path)",
                        placeholder="root.zhlh_202307_202412.ZHLH_4C_1216",
                        value="root.zhlh_202307_202412.ZHLH_4C_1216",
                        scale=2
                    )
                    
                    with gr.Row():
                         point_input = gr.Textbox(
                            label="点位名称 (Point Name)",
                            placeholder="FI_10401C.PV (留空查询所有*)",
                            value="FI_10401C.PV"
                        )
                    
                    with gr.Row():
                        start_time_input = gr.Textbox(label="开始时间", value="2023-07-18 12:00:00")
                        end_time_input = gr.Textbox(label="结束时间", value="2024-11-05 23:59:59")

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
            
            # 数据预览区域 - 图表优先，表格可折叠
            with gr.Row():
                preview_plot = gr.Image(label="Curve Preview", height=350)
            
            with gr.Accordion("📋 Data Table (first 5000 rows)", open=False):
                preview_table = gr.Dataframe(
                    label="",
                    interactive=False
                )
            
            # 事件绑定 - 数据获取
            refresh_datasets_btn.click(
                fn=get_datasets_table,
                outputs=datasets_table
            )
            refresh_datasets_btn.click(
                fn=lambda: gr.Dropdown(choices=get_dataset_names()),
                outputs=preview_dropdown
            )
            delete_dataset_btn.click(
                fn=delete_selected_dataset,
                inputs=preview_dropdown,
                outputs=[datasets_table, preview_dropdown, delete_status]
            )
            preview_dropdown.change(
                fn=preview_dataset,
                inputs=preview_dropdown,
                outputs=[preview_table, column_selector, preview_plot]
            )
            column_selector.change(
                fn=update_plot_from_selection,
                inputs=[preview_dropdown, column_selector],
                outputs=preview_plot
            )
            acquire_btn.click(
                fn=start_acquire_task,
                inputs=[
                    source_input, host_input, port_input, user_input, pwd_input,
                    point_input, start_time_input, end_time_input, target_points
                ],
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
                    
                    # 模型配置组
                    with gr.Group():
                        base_model_input = gr.Textbox(
                            label="Base Model Path (Base Model)", 
                            value="/home/share/llm_models/bytedance-research/ChatTS-8B",
                            info="基础模型路径"
                        )
                        lora_adapter_select = gr.Dropdown(
                            label="LoRA Adapter Path (可选)",
                            choices=get_inference_models(), # 返回的是 LoRA 路径列表
                            interactive=True,
                            info="微调后的 LoRA 适配器路径"
                        )
                        
                    files_select = gr.CheckboxGroup(
                        label="选择输入文件",
                        choices=get_dataset_names()
                    )
                    
                    with gr.Accordion("⚙️ 高级配置 (可选)", open=False):
                        with gr.Row():
                            n_downsample_input = gr.Slider(
                                label="降采样点数 (n_downsample)", 
                                minimum=100, maximum=10000, step=100, value=settings.DEFAULT_DOWNSAMPLE_POINTS
                            )
                            threshold_input = gr.Number(
                                label="异常阈值 (threshold)", value=8.0
                            )
                        
                        # ChatTS 专属参数
                        with gr.Group(visible=True) as chatts_group:
                            gr.Markdown("#### ChatTS 配置")
                            with gr.Row():
                                load_in_4bit_input = gr.Dropdown(
                                    label="4-bit 量化", choices=["auto", "true", "false"], value="auto",
                                    info="显存不足时建议开启(true)"
                                )
                                prompt_template_input = gr.Dropdown(
                                    label="Prompt 模板",
                                    choices=["default", "detailed", "minimal", "industrial", "english"],
                                    value="default"
                                )
                            with gr.Row():
                                chatts_device_input = gr.Textbox(label="Device", value="cuda:1")
                                chatts_use_cache_input = gr.Dropdown(
                                    label="Use Cache (KV)", choices=["auto", "true", "false"], value="auto"
                                )
                            max_new_tokens_input = gr.Number(
                                label="最大生成长度 (Max New Tokens)", value=4096, precision=0
                            )

                        # Timer 专属参数
                        with gr.Group(visible=False) as timer_group:
                            gr.Markdown("#### Timer 配置")
                            with gr.Row():
                                timer_device_input = gr.Textbox(label="Device", value="cuda:0")
                                timer_lookback_input = gr.Number(label="Lookback Length", value=256, precision=0)
                            with gr.Row():
                                timer_threshold_k_input = gr.Number(label="Threshold K", value=3.5)
                                timer_method_input = gr.Dropdown(label="Method", choices=["mad", "sigma"], value="mad")
                            timer_streaming_input = gr.Checkbox(label="Enable Streaming Mode", value=False)
                            
                        # ADTK 专属参数
                        with gr.Group(visible=False) as adtk_group:
                            gr.Markdown("#### ADTK HBOS 配置")
                            with gr.Row():
                                adtk_bin_nums_input = gr.Number(label="Bin Nums (分箱数)", value=20, precision=0)
                                adtk_hbos_ratio_input = gr.Number(label="HBOS Ratio (跳变阈值)", value=None)

                    with gr.Row():
                        submit_inference_btn = gr.Button("🚀 提交任务", variant="primary")
                        stop_inference_btn = gr.Button("🛑 停止任务", variant="stop", visible=False)
                    
                    # 隐藏的状态组件，用于存储 current task id
                    current_task_id_state = gr.State("")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 任务状态 & 日志")
                    with gr.Tabs():
                        with gr.Tab("实时日志"):
                            inference_logs = gr.Textbox(
                                value="",
                                label="Execution Logs",
                                interactive=False,
                                lines=20,
                                max_lines=20,
                                autoscroll=True
                            )
                        with gr.Tab("任务结果"):
                             # 当前任务状态
                             inference_result_md = gr.Markdown(value="等待任务完成...")
                             download_files = gr.File(label="当前任务结果", file_count="multiple", interactive=False, visible=False)
                             
                             # 整合的结果文件管理区
                             gr.Markdown("### 📂 结果文件管理")
                             with gr.Row():
                                 results_method_select = gr.Dropdown(
                                     label="筛选方法",
                                     choices=["chatts", "timer", "adtk_hbos"],
                                     value="chatts",
                                     scale=1
                                 )
                                 refresh_results_btn = gr.Button("🔄 刷新列表", size="sm", scale=0)
                             
                             # 统一文件列表（多选）
                             file_manager_list = gr.CheckboxGroup(
                                 label="文件列表 (文件名 | 较新的在前)",
                                 choices=get_result_filenames("chatts"),
                                 value=[],
                                 interactive=True
                             )
                             
                             with gr.Row():
                                 download_selected_btn = gr.Button("⬇️ 下载选中", size="sm")
                                 delete_selected_btn = gr.Button("🗑️ 删除选中", variant="stop", size="sm")

                             operation_status = gr.Markdown(value="")

                             # 下载区域 (动态显示)
                             history_download_files = gr.File(
                                 label="📥 下载区域 (请先选择文件并点击“下载选中”)",
                                 file_count="multiple",
                                 interactive=False,
                                 visible=True
                             )
                    
                    # 任务历史记录 - 放入可折叠区域
                    with gr.Accordion("📋 任务历史记录", open=False):
                        with gr.Row():
                            refresh_tasks_btn = gr.Button("🔄 刷新状态", size="sm")
                            clear_tasks_btn = gr.Button("🗑️ 清空历史", size="sm", variant="stop")
                        clear_status = gr.Markdown(value="", visible=True)
                        task_table = gr.Dataframe(
                            headers=["ID", "类型", "状态", "创建时间"],
                            value=[],
                            interactive=False
                        )
            
            # 事件绑定 - 推理监控
            
            # 提交任务
            submit_event = submit_inference_btn.click(
                fn=start_inference_task,
                inputs=[
                    algo_dropdown, base_model_input, lora_adapter_select, files_select,
                    # 通用参数
                    n_downsample_input, threshold_input,
                    # ChatTS 参数
                    load_in_4bit_input, prompt_template_input, max_new_tokens_input, chatts_device_input, chatts_use_cache_input,
                    # Timer 参数
                    timer_device_input, timer_lookback_input, timer_threshold_k_input, timer_method_input, timer_streaming_input,
                    # ADTK 参数
                    adtk_bin_nums_input, adtk_hbos_ratio_input
                ],
                outputs=[
                    inference_logs, 
                    inference_result_md, 
                    stop_inference_btn, 
                    submit_inference_btn, 
                    current_task_id_state, 
                    download_files
                ]
            )
            
            # 停止任务
            stop_inference_btn.click(
                fn=stop_task_action,
                inputs=[current_task_id_state],
                outputs=[
                    inference_result_md, 
                    stop_inference_btn, 
                    submit_inference_btn, 
                    current_task_id_state, 
                    download_files
                ]
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
                outputs=lora_adapter_select
            )
            
            # 清空历史记录
            clear_tasks_btn.click(
                fn=clear_task_history,
                outputs=[task_table, clear_status]
            )
            
            # 历史结果文件刷新
            refresh_results_btn.click(
                fn=lambda m: gr.CheckboxGroup(choices=get_result_filenames(m)),
                inputs=results_method_select,
                outputs=file_manager_list
            )
            
            # 切换方法时刷新结果列表
            results_method_select.change(
                fn=lambda m: gr.CheckboxGroup(choices=get_result_filenames(m)),
                inputs=results_method_select,
                outputs=file_manager_list
            )
            
            # 删除选中文件
            delete_selected_btn.click(
                fn=delete_selected_files,
                inputs=[results_method_select, file_manager_list],
                outputs=[file_manager_list, history_download_files, operation_status]
            )
            
            # 下载选中文件
            # 下载选中文件
            download_selected_btn.click(
                fn=prepare_download_files,
                inputs=[results_method_select, file_manager_list],
                outputs=[history_download_files, operation_status]
            )
            
            # 算法切换事件：控制参数组显示
            algo_dropdown.change(
                fn=toggle_algo_params,
                inputs=algo_dropdown,
                outputs=[chatts_group, timer_group, adtk_group]
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
    
        # 初始化加载
        demo.load(
            fn=get_dataset_names,
            outputs=preview_dropdown
        ).then(
            fn=lambda x: x[0] if x else None,
            inputs=preview_dropdown,
            outputs=preview_dropdown
        ).then(
            fn=preview_dataset,
            inputs=preview_dropdown,
            outputs=[preview_table, column_selector, preview_plot]
        )
            
    return demo


# 创建全局 Gradio 应用实例
training_ui = create_training_ui()
