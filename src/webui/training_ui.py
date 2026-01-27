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
import tempfile
import os

from configs.settings import settings
from src.adapters.chatts_training import ChatTSTrainingAdapter
from src.adapters.data_processing import DataProcessingAdapter
from src.adapters.check_outlier import CheckOutlierAdapter
from src.utils.iotdb_config import load_iotdb_config


# 初始化适配器
training_adapter = ChatTSTrainingAdapter()
data_adapter = DataProcessingAdapter()
inference_adapter = CheckOutlierAdapter()

# 为了兼容性保留旧变量名
adapter = training_adapter

# 结果文件目录（使用标准化路径）
RESULTS_BASE_PATH = Path(settings.DATA_INFERENCE_DIR)

# 统一数据源：使用 data_adapter 的实际路径
# 注意：这里不再硬编码路径，而是使用与数据获取页面相同的路径

# 文件名到完整路径的映射 (用于在UI显示文件名，内部使用完整路径)
_unified_file_mapping: Dict[str, str] = {}


def get_unified_file_list() -> List[str]:
    """
    获取统一的文件列表（与数据获取页面相同数据源）
    返回完整路径列表
    """
    global _unified_file_mapping
    all_files = []
    _unified_file_mapping.clear()
    
    # 使用与数据获取页面相同的数据路径
    data_path = data_adapter.data_path
    if data_path.exists():
        for f in data_path.glob("*.csv"):
            if f.exists():
                # 过滤掉推理结果文件 (global_chatts_ 开头, chatts_ 开头, timer_ 开头, 以及包含 _trend_resid 的文件)
                if f.name.startswith(("global_chatts_", "chatts_", "_chatts_", "timer_")):
                    continue
                if "_trend_resid" in f.name:
                    continue
                full_path = str(f)
                all_files.append(full_path)
                _unified_file_mapping[f.name] = full_path
    
    # 按修改时间排序（最新在前）
    def safe_mtime(p):
        try:
            return Path(p).stat().st_mtime
        except OSError:
            return 0
    
    all_files.sort(key=safe_mtime, reverse=True)
    return all_files[:50]  # 最多返回 50 个


def get_unified_file_names() -> List[str]:
    """获取统一文件列表的文件名（不含路径）"""
    # 确保映射已更新
    get_unified_file_list()
    return list(_unified_file_mapping.keys())


def resolve_filenames_to_paths(filenames: List[str]) -> List[str]:
    """将文件名列表转换为完整路径列表"""
    global _unified_file_mapping
    if not _unified_file_mapping:
        get_unified_file_list()
    
    paths = []
    for name in filenames:
        if name in _unified_file_mapping:
            paths.append(_unified_file_mapping[name])
        elif Path(name).exists():
            # 如果已经是完整路径
            paths.append(name)
    return paths



def get_existing_results(method: str = "chatts") -> List[str]:
    """获取已有的结果文件列表"""
    results_dir = RESULTS_BASE_PATH / method
    if not results_dir.exists():
        return []
    
    # 获取所有 CSV 文件，按修改时间排序（最新的在前）
    csv_files = []
    for f in results_dir.glob("*.csv"):
        if f.exists(): # 只包含存在的文件（过滤掉断裂的符号链接）
            csv_files.append(f)
            
    # 安全排序：如果 stat 失败（例如竞态条件），使用 0 作为时间戳
    def safe_get_mtime(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0
            
    csv_files.sort(key=safe_get_mtime, reverse=True)
    return [str(f) for f in csv_files[:20]]  # 最多返回 20 个


def delete_selected_files(method: str, filenames: List[str]) -> tuple:
    """批量删除选中的结果文件"""
    if not filenames:
        return (
            gr.CheckboxGroup(choices=get_result_filenames(method)), 
            gr.File(value=None), 
            "⚠️ 请先选择要删除的文件"
        )
    
    results_dir = RESULTS_BASE_PATH / method
    deleted_count = 0
    errors = []
    
    for fname in filenames:
        file_path = results_dir / fname.strip()  # Strip whitespace just in case
        print(f"DEBUG: Attempting to delete {file_path}")
        
        # 处理符号链接和普通文件
        # is_file() 对符号链接如果指向存在文件则为真
        # is_symlink() 判断是否为符号链接
        # exists() 如果是断裂符号链接则为假
        
        try:
            # 尝试删除（如果是符号链接则删除链接，如果是文件则删除文件）
            if file_path.is_symlink() or file_path.exists():
                file_path.unlink()
                deleted_count += 1
                print(f"DEBUG: Deleted {file_path}")
                
                # 同步删除关联的符号链接 (在 downsampled 和用户目录中)
                try:
                    # 1. 删除 downsampled 目录下的同名链接
                    symlink_path = Path(settings.DATA_DOWNSAMPLED_DIR) / fname.strip()
                    if symlink_path.is_symlink():
                        symlink_path.unlink()
                        print(f"DEBUG: Deleted symlink {symlink_path}")
                        
                    # 2. 删除 Annotator 用户目录下的同名链接
                    annotator_users_file = Path(settings.DATA_PROCESSING_PATH).parent / "annotator" / "backend" / "users.json"
                    if annotator_users_file.exists():
                        import json
                        with open(annotator_users_file, 'r') as f:
                            users = json.load(f)
                        for u_info in users.values():
                            if 'data_path' in u_info:
                                u_link = Path(u_info['data_path']) / fname.strip()
                                if u_link.is_symlink():
                                    u_link.unlink()
                                    print(f"DEBUG: Deleted user symlink {u_link}")
                except Exception as e_link:
                    print(f"DEBUG: Error cleaning up symlinks: {e_link}")
            else:
                # 再次检查是否是“断裂的符号链接”（exists()返回False但链接本身存在）
                # Path.is_symlink() 即使目标不存在也返回 True
                if file_path.is_symlink():
                     file_path.unlink()
                     deleted_count += 1
                     print(f"DEBUG: Deleted broken symlink {file_path}")
                else:
                     print(f"DEBUG: File not found {file_path}")
                     # 此时可能用户选了一个已经不存在的文件（缓存问题），不报错，只记录
        except Exception as e:
            errors.append(f"{fname}: {str(e)}")
            print(f"DEBUG: Error deleting {file_path}: {e}")
    
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
    
    results_dir = RESULTS_BASE_PATH / method
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
    results_dir = RESULTS_BASE_PATH / method  # 新目录结构：/home/share/data/inference/{method}
    if not results_dir.exists():
        return []
    
    csv_files = []
    for f in results_dir.glob("*.csv"):
        # 安全检查：如果是断裂的符号链接，f.exists() 会返回 False
        if f.exists() or f.is_symlink():
            csv_files.append(f)
            
    def safe_get_mtime(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0

    csv_files.sort(key=safe_get_mtime, reverse=True)
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
    
    # 将选中的文件名转换为完整路径（使用统一数据源映射）
    file_paths = resolve_filenames_to_paths(files)
    
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
        
        # 自动将结果文件链接到用户数据目录，以便标注工具默认可见
        try:
            # 目标目录列表：data_adapter 目录 + Annotator 用户目录
            target_dirs = [data_adapter.data_path]
            
            # 尝试读取 Annotator 用户配置
            try:
                annotator_users_file = Path(settings.DATA_PROCESSING_PATH).parent / "annotator" / "backend" / "users.json"
                if annotator_users_file.exists():
                    import json
                    with open(annotator_users_file, 'r') as f:
                        users = json.load(f)
                    # 遍历所有用户，将结果链接到每个用户的 data_path
                    for username, user_info in users.items():
                        if 'data_path' in user_info:
                            user_dir = Path(user_info['data_path'])
                            if user_dir.exists() and user_dir not in target_dirs:
                                target_dirs.append(user_dir)
            except Exception as e:
                print(f"[Auto-Link] Warning: Could not read annotator users.json: {e}")
            
            for res_file in generated_files:
                res_path = Path(res_file)
                # 如果是相对路径或文件名，尝试在结果目录查找
                if not res_path.exists():
                    res_path = RESULTS_BASE_PATH / algorithm / res_file
                
                if res_path.exists():
                    for target_dir in target_dirs:
                        # 避免自引用链接 (当目标目录就是结果文件所在目录时)
                        try:
                            if target_dir.resolve() == res_path.parent.resolve():
                                continue
                        except Exception:
                            pass

                        target_link = target_dir / res_path.name
                        try:
                            # 如果链接不存在或已断裂，重新创建
                            if target_link.is_symlink() or target_link.exists():
                                target_link.unlink()
                            target_link.symlink_to(res_path)
                            print(f"[Auto-Link] Created symlink for {res_path.name} in {target_dir}")
                        except Exception as link_err:
                            print(f"[Auto-Link] Failed to link to {target_dir}: {link_err}")
        except Exception as e:
            print(f"[Auto-Link Error] Failed to link results: {e}")
        
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
    output_name: str,
    model_path: Optional[str] = None,
    dataset_name: Optional[str] = None
) -> str:
    """启动训练 (Quick Start)"""
    if not config_name:
        return "❌ 请选择训练配置 (模板)"
    
    if not output_name:
        return "❌ 请输入输出目录名称"
    
    # 生成任务ID
    import uuid
    task_id = f"job-{str(uuid.uuid4())[:8]}"
    
    try:
        # 调用后端适配器启动训练
        result = training_adapter.run_training(
            task_id=task_id,
            config_name=config_name,
            version_tag=output_name,
            # Quick Start Overrides
            override_model_path=model_path,
            override_dataset=dataset_name,
            override_learning_rate=learning_rate,
            override_epochs=num_epochs,
            override_batch_size=batch_size,
            override_lora_rank=lora_rank,
            override_lora_alpha=lora_alpha
        )
        
        if result.get("success"):
            return f"""✅ 训练任务已成功启动!
            
**任务 ID**: {task_id}
**输出目录**: `{result.get('output_dir')}`
**基础模型**: `{model_path or 'Default (from script)'}`
**数据集**: `{dataset_name or 'Default (from script)'}`

正在后台运行中... 请留意日志输出或稍后刷新模型列表。
"""
        else:
            return f"""❌ 启动失败
            
**错误信息**: {result.get('error')}
"""
            
    except Exception as e:
        return f"❌ 系统错误: {str(e)}"


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
                        # 从共享配置加载默认值
                        _iotdb_cfg = load_iotdb_config()
                        with gr.Row():
                            host_input = gr.Textbox(label="Host", value=_iotdb_cfg.get("host", "192.168.199.185"))
                            port_input = gr.Textbox(label="Port", value=_iotdb_cfg.get("port", "6667"))
                        with gr.Row():
                            user_input = gr.Textbox(label="User", value=_iotdb_cfg.get("user", "root"))
                            pwd_input = gr.Textbox(label="Password", value=_iotdb_cfg.get("password", "root"), type="password")

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
        with gr.Tab("🔍 推理监控") as inference_tab:
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
                        choices=get_unified_file_names()
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
                fn=lambda: gr.CheckboxGroup(choices=get_unified_file_names()),
                outputs=files_select
            )
            refresh_tasks_btn.click(
                fn=lambda: gr.Dropdown(choices=get_inference_models()),
                outputs=lora_adapter_select
            )
            
            # Tab 切换时自动刷新文件列表
            inference_tab.select(
                fn=lambda: gr.CheckboxGroup(choices=get_unified_file_names()),
                outputs=files_select
            )
            inference_tab.select(
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
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 🔗 快速访问")
                    # 使用 HTML 按钮打开链接，更直观
                    gr.HTML("""
                    <div style="padding: 10px; background-color: #f0f9ff; border-radius: 8px; border: 1px solid #bae6fd;">
                        <p style="margin-bottom: 10px; font-weight: bold; color: #0369a1;">
                            标注工具运行在独立服务端口 (5000)
                        </p>
                        <a href="http://192.168.199.126:5000" target="_blank" style="
                            display: inline-block;
                            padding: 10px 20px;
                            background-color: #0284c7;
                            color: white;
                            text-decoration: none;
                            border-radius: 6px;
                            font-weight: bold;
                        ">
                            🚀 打开标注工具 (Open Annotator)
                        </a>
                    </div>
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 状态概览")
                    # 动态获取标注文件数
                    def get_annotation_stats():
                        # 使用 Settings
                        ann_dir = Path(settings.ANNOTATIONS_ROOT) / "douff"
                        if not ann_dir.exists():
                            return "暂无标注目录"
                        count = len(list(ann_dir.glob("*.json")))
                        return f"已标注文件数: {count}"

                    annotation_stats = gr.Textbox(
                        value=get_annotation_stats(),
                        label="当前标注进度",
                        interactive=False
                    )
                    refresh_ann_btn = gr.Button("🔄 刷新状态", size="sm")
                    refresh_ann_btn.click(fn=get_annotation_stats, outputs=annotation_stats)

            gr.Markdown("---")
            gr.Markdown("### 🔄 数据转换 (Annotator -> ChatTS)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 配置区域
                    with gr.Accordion("⚙️ 路径与参数配置 (Settings)", open=False):
                        conf_input_dir = gr.Textbox(
                            label="标注文件来源 (Annotation Dir)", 
                            value=str(Path(settings.ANNOTATIONS_ROOT) / "douff")
                        )
                        conf_image_dir = gr.Textbox(
                            label="图片文件来源 (Image Dir)", 
                            value=settings.DATA_DOWNSAMPLED_DIR
                        )
                        conf_output_path = gr.Textbox(
                            label="转换输出路径 (Output Path)", 
                            value=str(Path(settings.DATA_TRAINING_DIR) / "converted_data.json")
                        )

                    # 获取标注文件列表
                    def get_file_choices(ann_dir):
                        path_obj = Path(ann_dir)
                        if not path_obj.exists():
                            return []
                        try:
                            files = list(path_obj.glob("*.json"))
                            # 按修改时间排序
                            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            return [f.name for f in files]
                        except Exception:
                            return []

                    # 初始加载
                    # 初始加载
                    default_ann_dir = str(Path(settings.ANNOTATIONS_ROOT) / "douff")
                    initial_choices = get_file_choices(default_ann_dir)
                    initial_val = initial_choices[0] if initial_choices else None

                    ann_file_dropdown = gr.Dropdown(
                        label="选择要预览/转换的文件",
                        choices=initial_choices,
                        value=initial_val,
                        multiselect=False,
                        interactive=True,
                        allow_custom_value=False
                    )
                    
                    def refresh_files(ann_dir):
                        choices = get_file_choices(ann_dir)
                        val = choices[0] if choices else None
                        return gr.Dropdown(choices=choices, value=val)
                        
                    refresh_files_btn = gr.Button("🔄 刷新文件列表", size="sm")
                    
                    with gr.Row():
                        convert_curr_btn = gr.Button("🚀 仅转换选中", variant="primary")
                        convert_all_btn = gr.Button("📦 批量转换所有", variant="secondary")
                    
                with gr.Column(scale=2):
                    convert_status = gr.Textbox(label="操作日志 (Execution Log)", lines=10, interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📝 转换前 (Annotator JSON)")
                    before_json = gr.JSON(label="Source Data", height=400)
                with gr.Column():
                    gr.Markdown("#### 🎯 转换后 (ChatTS Training Data)")
                    after_json = gr.JSON(label="Converted Data", height=400)
                
            def preview_source_file(selected_file, input_dir_val, image_dir_val):
                """选择文件时立即预览，并执行真实转换（使用临时文件）"""
                if not selected_file or not input_dir_val:
                    return None, None
                 
                # 1. 读取源文件
                src_p = Path(input_dir_val) / selected_file
                source_content = None
                
                try:
                    if src_p.exists():
                        with open(src_p, 'r', encoding='utf-8') as f:
                            source_content = json.load(f)
                except Exception as e:
                    source_content = {"error": str(e)}
                
                # 2. 执行转换预览 (真实调用适配器)
                converted_content = None
                try:
                    # 使用临时文件作为输出
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    # 默认 image_dir 如果为空
                    img_d = image_dir_val if image_dir_val else settings.DATA_DOWNSAMPLED_DIR
                    
                    res = data_adapter.convert_annotations(
                        input_dir=input_dir_val,
                        output_path=tmp_path,
                        image_dir=img_d,
                        filename=selected_file
                    )
                    
                    if res["success"]:
                        try:
                            with open(tmp_path, 'r', encoding='utf-8') as f:
                                converted_content = json.load(f)
                        except Exception as read_err:
                            converted_content = {"error": f"Read converted file failed: {read_err}"}
                    else:
                        converted_content = {
                            "error": "Conversion failed", 
                            "stderr": res.get("stderr", ""),
                            "stdout": res.get("stdout", "")
                        }
                    
                    # 清理临时文件
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except:
                        pass

                except Exception as e:
                    converted_content = {"error": f"Preview failed: {str(e)}"}
                
                return source_content, converted_content

            def convert_core(selected_file, input_dir, image_dir, output_path, mode="single"):
                """核心转换逻辑"""
                # 创建输出目录
                try:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return f"❌ 创建输出目录失败: {str(e)}", {}, {}
                
                target_filename = selected_file if mode == "single" else None
                
                # 执行转换
                result = data_adapter.convert_annotations(
                    input_dir, 
                    output_path, 
                    image_dir=image_dir,
                    filename=target_filename
                )
                
                status_msg = ""
                source_sample = {}
                target_sample = {}
                
                if result.get("success"):
                    log_output = result.get("stdout", "")
                    prefix = "单文件" if mode == "single" else "批量"
                    status_msg = f"✅ {prefix}转换成功! \n输出: {result.get('output_path')}\n\n执行日志:\n{log_output}"
                    
                    # 尝试读取源文件进行预览
                    try:
                         # 如果是批量转换但依然选了文件，或者单文件模式
                         preview_file = selected_file
                         if not preview_file and Path(input_dir).exists():
                             # 如果都没选，找第一个
                             all_jsons = list(Path(input_dir).glob("*.json"))
                             if all_jsons:
                                 preview_file = all_jsons[0].name
                         
                         if preview_file:
                             src_p = Path(input_dir) / preview_file
                             if src_p.exists():
                                 with open(src_p, 'r', encoding='utf-8') as f:
                                    source_sample = json.load(f)
                    except Exception as e:
                        source_sample = {"error": f"Read source failed: {str(e)}"}

                    # 读取转换后的结果
                    try:
                        with open(output_path, 'r', encoding='utf-8') as f:
                            converted_data = json.load(f)
                            if converted_data and isinstance(converted_data, list):
                                if preview_file:
                                    # 尝试匹配 image 路径
                                    core_name = preview_file.replace("annotations_数据集", "").replace(".json", "")
                                    # 简单去后缀
                                    core_name = core_name.replace(".csv", "")
                                    
                                    matched = False
                                    for item in converted_data:
                                        if core_name in item.get("image", ""):
                                            target_sample = item
                                            matched = True
                                            break
                                    if not matched:
                                        target_sample = converted_data[-1] if mode == "single" else converted_data[0]
                                        status_msg = f"(注意：预览未精确匹配，显示{'最后一条' if mode=='single' else '第一条'})\n" + status_msg
                                else:
                                    target_sample = converted_data[0]
                    except Exception as e:
                        target_sample = {"error": f"Read output failed: {str(e)}"}
                        
                else:
                    status_msg = f"❌ 转换失败: {result.get('error')}\n日志:\n{result.get('stderr', '')}"
                
                return status_msg, source_sample, target_sample

            # 绑定事件
            convert_curr_btn.click(
                fn=lambda f, i, m, o: convert_core(f, i, m, o, mode="single"),
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conf_output_path],
                outputs=[convert_status, before_json, after_json]
            )
            
            convert_all_btn.click(
                fn=lambda f, i, m, o: convert_core(f, i, m, o, mode="batch"),
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conf_output_path],
                outputs=[convert_status, before_json, after_json]
            )
            
            refresh_files_btn.click(
                fn=refresh_files,
                inputs=[conf_input_dir],
                outputs=ann_file_dropdown
            )
            
            # 选择文件立即预览
            ann_file_dropdown.change(
                fn=preview_source_file,
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir],
                outputs=[before_json, after_json]
            )
            
            # 初始化预览
            if initial_val:
                init_src, init_ex = preview_source_file(initial_val, default_ann_dir, settings.DATA_DOWNSAMPLED_DIR)
                before_json.value = init_src
                after_json.value = init_ex

        # ==================== 数据资产管理 Tab (New) ====================
        with gr.Tab("📦 数据资产管理"):
            with gr.Tabs():
                # 1. 标注数据管理
                with gr.Tab("标注数据 (Annotations)"):
                    gr.Markdown("### 📝 标注文件管理")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ann_mgr_dir = gr.Textbox(label="标注目录", value=str(Path(settings.ANNOTATIONS_ROOT) / "douff"), interactive=False)
                            ann_mgr_list = gr.Dropdown(label="选择文件", interactive=True)
                            refresh_ann_mgr = gr.Button("🔄 刷新列表")
                            delete_ann_btn = gr.Button("🗑️ 删除选中文件", variant="stop")
                            ann_op_status = gr.Textbox(label="操作状态", interactive=False)
                        
                        with gr.Column(scale=2):
                            ann_mgr_view = gr.JSON(label="文件内容预览", height=600)

                    # Logic
                    def list_ann_files(path_str):
                        p = Path(path_str)
                        if not p.exists(): return []
                        files = list(p.glob("*.json"))
                        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        return [f.name for f in files]

                    def load_ann_content(path_str, filename):
                        if not filename: return None
                        try:
                            with open(Path(path_str) / filename, 'r') as f:
                                return json.load(f)
                        except Exception as e:
                            return {"error": str(e)}

                    def delete_ann_file(path_str, filename):
                        if not filename: return "未选择文件", gr.update()
                        try:
                            p = Path(path_str) / filename
                            p.unlink()
                            # 刷新列表
                            new_list = list_ann_files(path_str)
                            return f"已删除: {filename}", gr.update(choices=new_list, value=None)
                        except Exception as e:
                            return f"删除失败: {e}", gr.update()

                    # Bindings
                    ann_mgr_dir.change(fn=lambda p: gr.update(choices=list_ann_files(p)), inputs=ann_mgr_dir, outputs=ann_mgr_list)
                    refresh_ann_mgr.click(fn=lambda p: gr.update(choices=list_ann_files(p)), inputs=ann_mgr_dir, outputs=ann_mgr_list)
                    ann_mgr_list.change(fn=load_ann_content, inputs=[ann_mgr_dir, ann_mgr_list], outputs=ann_mgr_view)
                    delete_ann_btn.click(fn=delete_ann_file, inputs=[ann_mgr_dir, ann_mgr_list], outputs=[ann_op_status, ann_mgr_list])

                # 2. 训练数据管理
                with gr.Tab("训练数据 (Training Data)"):
                    gr.Markdown("### 🎯 微调数据管理 (Converted JSONL)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            train_mgr_dir = gr.Textbox(label="数据目录", value=settings.DATA_TRAINING_DIR, interactive=False)
                            train_mgr_list = gr.Dropdown(label="选择文件", interactive=True)
                            refresh_train_mgr = gr.Button("🔄 刷新列表")
                            delete_train_btn = gr.Button("🗑️ 删除选中文件", variant="stop")
                            train_op_status = gr.Textbox(label="操作状态", interactive=False)
                        
                        with gr.Column(scale=2):
                            train_mgr_view = gr.JSON(label="文件内容预览 (Head 50 lines / JSON)", height=600)

                    # Logic
                    def list_train_files(path_str):
                        p = Path(path_str)
                        if not p.exists(): return []
                        files = list(p.glob("*.json")) + list(p.glob("*.jsonl"))
                        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        return [f.name for f in files]

                    def load_train_content(path_str, filename):
                        if not filename: return None
                        try:
                            p = Path(path_str) / filename
                            
                            # Strategy 1: Small file (<50MB) -> Try full JSON load
                            # This handles standard JSON lists (pretty printed or minified)
                            if p.stat().st_size < 50 * 1024 * 1024: 
                                try:
                                    with open(p, 'r') as f:
                                        data = json.load(f)
                                    if isinstance(data, list):
                                        return data[:50]  # Preview first 50 items
                                    return data
                                except:
                                    pass # Fallback to Strategy 2
                            
                            # Strategy 2: JSONL or Large File -> Line-by-line
                            records = []
                            with open(p, 'r') as f:
                                for _ in range(50):
                                    line = f.readline()
                                    if not line: break
                                    line = line.strip()
                                    if not line: continue
                                    try:
                                        records.append(json.loads(line))
                                    except:
                                        # If it looks like start/end of array, skip or show raw
                                        if line in ['[', ']', '],']: continue
                                        records.append({"raw_text": line})
                            return records
                        except Exception as e:
                            return {"error": str(e)}

                    def delete_train_file(path_str, filename):
                        if not filename: return "未选择文件", gr.update()
                        try:
                            p = Path(path_str) / filename
                            p.unlink()
                            new_list = list_train_files(path_str)
                            return f"已删除: {filename}", gr.update(choices=new_list, value=None)
                        except Exception as e:
                            return f"删除失败: {e}", gr.update()

                    # Bindings
                    # Init load
                    train_mgr_dir.change(fn=lambda p: gr.update(choices=list_train_files(p)), inputs=train_mgr_dir, outputs=train_mgr_list)
                    refresh_train_mgr.click(fn=lambda p: gr.update(choices=list_train_files(p)), inputs=train_mgr_dir, outputs=train_mgr_list)
                    train_mgr_list.change(fn=load_train_content, inputs=[train_mgr_dir, train_mgr_list], outputs=train_mgr_view)
                    delete_train_btn.click(fn=delete_train_file, inputs=[train_mgr_dir, train_mgr_list], outputs=[train_op_status, train_mgr_list])
        
        # ==================== 微调训练 Tab (原有) ====================
        with gr.Tab("🎯 开始训练"):
            with gr.Row():
                with gr.Column(scale=2):
                    # 基础配置
                    gr.Markdown("### 基础配置")
                    
                    # New Dropdowns for Quick Start
                    with gr.Row():
                        model_path_dropdown = gr.Dropdown(
                            label="基础模型 (Base Model)",
                            choices=adapter.get_base_models(),
                            value=adapter.get_base_models()[0] if adapter.get_base_models() else None,
                            interactive=True,
                            allow_custom_value=True
                        )
                        dataset_dropdown = gr.Dropdown(
                            label="微调数据集 (Dataset)",
                            choices=adapter.get_dataset_list(),
                            value=adapter.get_dataset_list()[0] if adapter.get_dataset_list() else None,
                            interactive=True
                        )

                    config_dropdown = gr.Dropdown(
                        label="训练模板 (Template Script)",
                        choices=get_training_configs(),
                        interactive=True,
                        info="选择一个脚本作为参数模板 (如 DeepSpeed 配置)"
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
                    # --- Backend Functions ---
                    def start_training_wrap(
                        config_name, lr, epochs, batch_size, rank, alpha, output_name,
                        model_path, dataset_name
                    ):
                        if not config_name:
                            return "❌ 请选择训练模板", ""
                            
                        task_id = f"qs_{int(time.time())}"
                        
                        # Call backend
                        overrides = {
                            "override_learning_rate": lr,
                            "override_epochs": epochs,
                            "override_batch_size": batch_size,
                            "override_lora_rank": rank,
                            "override_lora_alpha": alpha,
                            "override_model_path": model_path,
                            "override_dataset": dataset_name
                        }
                        
                        res = adapter.run_training(task_id, config_name, version_tag=output_name, **overrides)
                        
                        if res.get("success"):
                            return f"✅ 训练任务已成功启动!\n任务ID: {task_id}\n输出目录: {res.get('output_dir')}\n\n正在后台运行中... 请留意下方实时日志。", task_id
                        else:
                            return f"❌ 启动错误: {res.get('error')}", ""

                    def stream_logs(task_id, current_log, offset):
                        if not task_id:
                            return current_log, offset
                        
                        # Read increment
                        res = adapter.get_training_log(task_id, offset)
                        new_content = res.get("log", "")
                        new_offset = res.get("offset", offset)
                        
                        if new_content:
                            current_log = (current_log or "") + new_content
                            
                        return current_log, new_offset

                    def stop_training_wrap(task_id):
                        if not task_id:
                            return "无运行中的任务"
                        res = adapter.stop_training(task_id)
                        if res.get("success"):
                            return "🛑 任务已手动停止"
                        else:
                            return f"停止失败: {res.get('error')}"

                    # --- Layout & Events ---
                    with gr.Column():
                        with gr.Row():
                            start_btn = gr.Button("🚀 (Quick Start) 开始训练", variant="primary", scale=2)
                            stop_btn = gr.Button("🛑 停止训练", variant="stop", scale=1)
                            refresh_btn = gr.Button("🔄 刷新配置", scale=1)
                        
                        # Hidden state for Task ID and Log Offset
                        task_id_state = gr.State("")
                        log_offset_state = gr.State(0)
                        
                        output_box = gr.Textbox(label="训练状态", lines=4)
                        log_box = gr.Code(label="实时日志 (Real-time Logs)", language="shell", lines=15)
                        
                        # Timer for polling
                        timer = gr.Timer(1) # 1s interval

                        # Events
                        start_btn.click(
                            fn=start_training_wrap,
                            inputs=[
                                config_dropdown, learning_rate, num_epochs,
                                batch_size, lora_rank, lora_alpha, output_name,
                                model_path_dropdown, dataset_dropdown
                            ],
                            outputs=[output_box, task_id_state]
                        ).then(
                            fn=lambda: 0, outputs=log_offset_state # Reset offset
                        ).then(
                            fn=lambda: "", outputs=log_box # Clear logs
                        )
                        
                        stop_btn.click(
                            fn=stop_training_wrap,
                            inputs=[task_id_state],
                            outputs=output_box
                        )
                        
                        # Timer ticks -> Update logs
                        timer.tick(
                            fn=stream_logs,
                            inputs=[task_id_state, log_box, log_offset_state],
                            outputs=[log_box, log_offset_state]
                        )
                        refresh_btn.click(
                            fn=lambda: (
                                gr.Dropdown(choices=get_training_configs()),
                                gr.Dropdown(choices=adapter.get_dataset_list())
                            ),
                            outputs=[config_dropdown, dataset_dropdown]
                        )

            # ==================== Advanced Mode (Native Integration) ====================
        with gr.Accordion("⚙️ 高级模式 (Native WebUI Integration)", open=False):
                gr.Markdown("""
                > **专家模式**: 将当前选择的 Shell 脚本自动转换为 LLaMA-Factory 配置，并启动原生 WebUI 进行微调。
                > 适合需要调整 DeepSpeed、LR Scheduler 等高级参数的用户。
                """)
                with gr.Row():
                    convert_btn = gr.Button("🛠️ 1. 转换脚本为模板", variant="secondary")
                    launch_native_btn = gr.Button("🚀 2. 启动原生 WebUI", variant="primary")
                
                native_status = gr.Markdown("等待操作...")
                native_ui_link = gr.Markdown(visible=False)

                # Logic
                def convert_action(script_name):
                    if not script_name: return "⚠️ 请先选择一个脚本配置"
                    res = adapter.convert_script_to_config(script_name)
                    if res["success"]:
                        return f"✅ {res['message']}"
                    return f"❌ {res['error']}"

                def launch_action():
                    res = adapter.start_native_webui()
                    if res["success"]:
                        url = res['url']
                        return f"✅ {res['message']}", gr.Markdown(f"### [👉 点击访问原生 WebUI ({url})]({url})", visible=True)
                    return f"❌ {res['error']}", gr.update(visible=False)

                convert_btn.click(convert_action, inputs=config_dropdown, outputs=native_status)
                launch_native_btn.click(launch_action, outputs=[native_status, native_ui_link])
        
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
