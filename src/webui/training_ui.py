"""
Gradio 统一管理界面
包含：数据获取、推理监控、微调训练、模型对比
"""
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
import uuid
import pandas as pd
import tempfile
import os
import httpx

from configs.settings import settings
from src.core.logging_config import get_logger
from src.adapters.annotation_export import AnnotationExportAdapter
from src.adapters.chatts_training import ChatTSTrainingAdapter
from src.adapters.data_processing import DataProcessingAdapter
from src.adapters.check_outlier import CheckOutlierAdapter
from src.utils.iotdb_config import load_iotdb_config
from src.utils.file_filters import is_inference_or_generated_csv, match_result_method
from src.utils.model_eval import evaluate_model_on_golden
from src.utils.plot_utils import generate_ts_thumbnail, create_ts_image
from src.utils.annotation_store import canonical_point_id
from src.utils.time_utils import utc_now_naive


logger = get_logger(__name__)

TRAINING_MODEL_FAMILIES = ["chatts", "qwen"]
TRAINING_METHODS = ["all", "lora", "full"]

# 算法 -> 默认基础模型路径映射
ALGORITHM_DEFAULT_MODELS = {
    "chatts": "/home/share/llm_models/bytedance-research/ChatTS-8B",
    "qwen": "/home/share/models/Qwen3-VL-8B-train-8192_base",
    "timer": "/home/share/llm_models/thuml/timer-base-84m",
    "adtk_hbos": "",  # 不需要模型路径
    "stl_wavelet": "",
    "iforest": "",
    "piecewise_linear": "",
}

# 初始化适配器
chatts_adapter = ChatTSTrainingAdapter(model_family="chatts")
qwen_adapter = ChatTSTrainingAdapter(model_family="qwen")
training_adapter = chatts_adapter
data_adapter = DataProcessingAdapter()
annotation_export_adapter = AnnotationExportAdapter()
inference_adapter = CheckOutlierAdapter()

# 为了兼容性保留旧变量名
adapter = training_adapter


def get_training_adapter(model_family: str) -> ChatTSTrainingAdapter:
    return qwen_adapter if model_family == "qwen" else chatts_adapter

# 结果文件目录（使用标准化路径）
RESULTS_BASE_PATH = Path(settings.DATA_INFERENCE_DIR)

# 统一数据源：使用 data_adapter 的实际路径
# 注意：这里不再硬编码路径，而是使用与数据获取页面相同的路径

# 文件名到完整路径的映射 (用于在UI显示文件名，内部使用完整路径)
_unified_file_mapping: Dict[str, str] = {}
# 点位ID到最新CSV完整路径映射（point-first推理入口）
_point_file_mapping: Dict[str, str] = {}
# 数据页点位到文件名映射（按修改时间选择最新文件）
_dataset_point_mapping: Dict[str, str] = {}

# UI logs can grow very large; keep a tail to avoid infinite expansion.
LOG_TAIL_MAX_CHARS = 20000

# Force fixed-height scrolling for log widgets in Gradio 6.
LOG_SCROLL_CSS = """
#training-log, #inference-log {
  height: 320px !important;
  overflow: auto !important;
}
#training-log pre, #inference-log pre {
  max-height: 320px;
  overflow: auto;
}
"""

ASSETS_API_BASE = f"http://127.0.0.1:{settings.API_PORT}/api/v1/assets"
INFERENCE_API_BASE = f"http://127.0.0.1:{settings.API_PORT}/api/v1/inference"
DATA_API_BASE = f"http://127.0.0.1:{settings.API_PORT}/api/v1/data"
TRAINING_API_BASE = f"http://127.0.0.1:{settings.API_PORT}/api/v1/training"


def _assets_api_call(method: str, path: str, params: Optional[dict] = None, payload: Optional[dict] = None) -> Dict:
    url = f"{ASSETS_API_BASE}{path}"
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.request(method.upper(), url, params=params, json=payload)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            return {"success": False, "error": detail or f"HTTP {resp.status_code}"}
        if isinstance(data, dict):
            return data
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _inference_api_call(method: str, path: str, params: Optional[dict] = None, payload: Optional[dict] = None) -> Dict:
    url = f"{INFERENCE_API_BASE}{path}"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.request(method.upper(), url, params=params, json=payload)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            return {"success": False, "error": detail or f"HTTP {resp.status_code}"}
        if isinstance(data, dict) and data.get("success") is False:
            return {
                "success": False,
                "error": str(data.get("message") or "API 返回业务失败"),
                "data": data,
            }
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _data_api_call(method: str, path: str, params: Optional[dict] = None, payload: Optional[dict] = None) -> Dict:
    url = f"{DATA_API_BASE}{path}"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.request(method.upper(), url, params=params, json=payload)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            return {"success": False, "error": detail or f"HTTP {resp.status_code}"}
        if isinstance(data, dict) and data.get("success") is False:
            return {
                "success": False,
                "error": str(data.get("message") or "API 返回业务失败"),
                "data": data,
            }
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _training_api_call(method: str, path: str, params: Optional[dict] = None, payload: Optional[dict] = None) -> Dict:
    url = f"{TRAINING_API_BASE}{path}"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.request(method.upper(), url, params=params, json=payload)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            return {"success": False, "error": detail or f"HTTP {resp.status_code}"}
        if isinstance(data, dict) and data.get("success") is False:
            return {
                "success": False,
                "error": str(data.get("message") or "API 返回业务失败"),
                "data": data,
            }
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _extract_task_status_payload(payload: Optional[Dict]) -> Dict:
    """
    兼容两类返回：
    1) 直接 TaskResponse: {"task_id","status","message"}
    2) ApiResponse: {"success":true,"data":{...},"message":"..."}
    """
    if not isinstance(payload, dict):
        return {}
    if "task_id" in payload and "status" in payload:
        return payload

    inner = payload.get("data")
    if isinstance(inner, dict) and ("status" in inner or "task_id" in inner):
        merged = dict(inner)
        if not merged.get("message") and payload.get("message"):
            merged["message"] = payload.get("message")
        return merged
    return {}


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
                # 过滤掉推理结果/中间文件
                if is_inference_or_generated_csv(f.name):
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


def get_unified_point_ids() -> List[str]:
    """获取统一点位列表（由数据文件自动归一化映射）"""
    global _point_file_mapping
    _point_file_mapping.clear()
    for full_path in get_unified_file_list():
        point_id = canonical_point_id(Path(full_path).name)
        if not point_id:
            continue
        # 保留最新文件（get_unified_file_list 已按 mtime desc）
        if point_id not in _point_file_mapping:
            _point_file_mapping[point_id] = full_path
    return list(_point_file_mapping.keys())


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


def resolve_point_ids_to_paths(point_ids: List[str]) -> List[str]:
    """将点位ID列表转换为完整路径列表（兼容旧文件名输入）"""
    global _point_file_mapping
    if not _point_file_mapping:
        get_unified_point_ids()

    paths = []
    seen = set()
    for item in point_ids or []:
        key = str(item or "").strip()
        if not key:
            continue
        path = _point_file_mapping.get(key)
        if path is None:
            # 兼容旧值：文件名或全路径
            if key in _unified_file_mapping:
                path = _unified_file_mapping[key]
            elif Path(key).exists():
                path = key
            else:
                norm = canonical_point_id(key)
                path = _point_file_mapping.get(norm)
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def format_log_html(log_content: str) -> str:
    """Format log content as scrollable HTML"""
    if not log_content:
        log_content = ""
    # Simple escaping
    safe_content = log_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <div style="height: 300px;
                overflow-y: scroll;
                background-color: #f5f5f5;
                font-family: monospace;
                white-space: pre-wrap;
                font-size: 13px;
                line-height: 1.4;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;">
        {safe_content}
    </div>
    """



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
        logger.debug("尝试删除文件: %s", file_path)
        
        # 处理符号链接和普通文件
        # is_file() 对符号链接如果指向存在文件则为真
        # is_symlink() 判断是否为符号链接
        # exists() 如果是断裂符号链接则为假
        
        try:
            # 尝试删除（如果是符号链接则删除链接，如果是文件则删除文件）
            if file_path.is_symlink() or file_path.exists():
                file_path.unlink()
                deleted_count += 1
                logger.debug("已删除: %s", file_path)
                
                # 同步删除关联的符号链接 (在 downsampled 和用户目录中)
                try:
                    # 1. 删除 downsampled 目录下的同名链接
                    symlink_path = Path(settings.DATA_DOWNSAMPLED_DIR) / fname.strip()
                    if symlink_path.is_symlink():
                        symlink_path.unlink()
                        logger.debug("已删除符号链接: %s", symlink_path)
                        
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
                                    logger.debug("已删除用户符号链接: %s", u_link)
                except Exception as e_link:
                    logger.warning("清理符号链接失败: %s", e_link)
            else:
                # 再次检查是否是“断裂的符号链接”（exists()返回False但链接本身存在）
                # Path.is_symlink() 即使目标不存在也返回 True
                if file_path.is_symlink():
                     file_path.unlink()
                     deleted_count += 1
                     logger.debug("已删除断裂符号链接: %s", file_path)
                else:
                     logger.debug("文件未找到: %s", file_path)
                     # 此时可能用户选了一个已经不存在的文件（缓存问题），不报错，只记录
        except Exception as e:
            errors.append(f"{fname}: {str(e)}")
            logger.error("删除文件失败 %s: %s", file_path, e)
    
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
            # 严格过滤 (Addressing Issue: mixed files in directories)
            if not match_result_method(f.name, method):
                continue
            
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


def get_training_configs(model_family: str = "chatts", method: str = "all") -> List[str]:
    """获取训练配置列表"""
    configs = get_training_adapter(model_family).list_configs()
    method = (method or "all").lower()
    if method in ["lora", "full"]:
        configs = [c for c in configs if c.get("method") == method]
    return [c["name"] for c in configs]


def update_training_dropdowns(model_family: str, method: str) -> tuple:
    configs = get_training_configs(model_family, method)
    datasets = get_training_adapter(model_family).get_dataset_list()
    base_models = get_training_adapter(model_family).get_base_models()
    return (
        gr.Dropdown(choices=configs, value=(configs[0] if configs else None)),
        gr.Dropdown(choices=datasets, value=(datasets[0] if datasets else None)),
        gr.Dropdown(choices=base_models, value=(base_models[0] if base_models else None)),
    )


def update_training_config_only(model_family: str, method: str) -> gr.Dropdown:
    configs = get_training_configs(model_family, method)
    return gr.Dropdown(choices=configs, value=(configs[0] if configs else None))


def _is_checkpoint_model(model: Dict) -> bool:
    name = str(model.get("name", ""))
    path = str(model.get("path", ""))
    if name.startswith("checkpoint-"):
        return True
    return "/checkpoint-" in path.replace("\\", "/")


def _filter_models(model_family: str, model_type: str = "all", include_checkpoints: bool = False) -> List[Dict]:
    adapter = get_training_adapter(model_family)
    models = adapter.list_models()
    saves_prefix = str(adapter.saves_path).replace("\\", "/") + "/"
    models = [m for m in models if str(m.get("path", "")).replace("\\", "/").startswith(saves_prefix)]
    model_type = (model_type or "all").lower()
    if model_type in ["lora", "full"]:
        models = [m for m in models if m.get("type") == model_type]
    if not include_checkpoints:
        models = [m for m in models if not _is_checkpoint_model(m)]
    return models


def _relative_model_path(model_path: str) -> str:
    path = Path(model_path)
    parts = path.parts
    if "saves" in parts:
        idx = parts.index("saves")
        return str(Path(*parts[idx:]))
    return str(path)


def _format_model_choice(model: Dict) -> tuple[str, str]:
    label = f"{model.get('name', '')} ({model.get('type', 'unknown')}) · {_relative_model_path(model.get('path', ''))}"
    return (label, str(model.get("path", "")))


def get_trained_model_choices(model_family: str, model_type: str = "all", include_checkpoints: bool = False) -> List[tuple]:
    models = _filter_models(model_family, model_type, include_checkpoints)
    return [_format_model_choice(m) for m in models]


def _sorted_checkpoints(run_path: str) -> List[str]:
    if not run_path:
        return []
    root = Path(run_path)
    if not root.exists():
        return []
    checkpoints = []
    for cp in root.glob("checkpoint-*"):
        try:
            step = int(cp.name.split("-")[1])
        except Exception:
            step = 0
        checkpoints.append((step, cp.name))
    checkpoints.sort(key=lambda x: x[0])
    return [name for _, name in checkpoints]


def get_lora_run_choices(model_family: str) -> List[tuple]:
    """获取可用的 LoRA 训练任务列表，包含"无"选项"""
    choices = get_trained_model_choices(model_family, model_type="lora", include_checkpoints=False)
    # 添加 "无/不使用 LoRA" 选项，允许使用原始模型
    return [("无 (使用原始模型)", "")] + choices


def get_checkpoint_choices(run_path: str) -> List[tuple]:
    names = _sorted_checkpoints(run_path)
    if not names:
        return [("无", "")]
    return [("无", ""), ("最新", "__latest__")] + [(n, n) for n in names]


def resolve_lora_adapter_path(run_path: str, checkpoint_value: str) -> str:
    if not run_path:
        return ""
    if not checkpoint_value:
        return run_path
    if checkpoint_value == "__latest__":
        names = _sorted_checkpoints(run_path)
        if not names:
            return run_path
        return str(Path(run_path) / names[-1])
    return str(Path(run_path) / checkpoint_value)


def update_lora_run_dropdown(model_family: str, current_value: Optional[str] = None) -> gr.Dropdown:
    choices = get_inference_models(model_family)
    values = {v for _, v in choices}
    value = current_value if current_value in values else (choices[0][1] if choices else None)
    return gr.Dropdown(choices=choices, value=value)


def update_checkpoint_dropdown(run_path: str, current_value: Optional[str] = None) -> gr.Dropdown:
    choices = get_checkpoint_choices(run_path)
    values = {v for _, v in choices}
    value = current_value if current_value in values else ""
    return gr.Dropdown(choices=choices, value=value)


def sync_lora_family_from_algo(algorithm: str, current_family: str) -> gr.Dropdown:
    value = algorithm if algorithm in TRAINING_MODEL_FAMILIES else current_family
    return gr.Dropdown(value=value)


def get_trained_models() -> List[str]:
    """获取已训练模型列表 (保留旧接口，默认 chatts/all/不含 checkpoint)"""
    return [label for label, _ in get_trained_model_choices("chatts", "all", False)]


def get_model_info(model_path: str, model_family: str) -> str:
    """获取模型详细信息"""
    if not model_path:
        return "请选择一个模型"
    
    adapter = get_training_adapter(model_family)
    models = adapter.list_models()
    model = next((m for m in models if str(m.get("path")) == str(model_path)), None)
    
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


def run_model_evaluation_ui(
    model_path: str,
    model_family: str,
    truth_dir: str,
    data_dir: str,
    dataset_name: str,
    output_dir: str,
    device: str,
):
    if not model_path:
        return "❌ 请选择模型", {}
    if not truth_dir or not data_dir:
        return "❌ 请填写黄金集标注目录与数据目录", {}

    res = evaluate_model_on_golden(
        model_path=model_path,
        model_family=model_family,
        truth_dir=truth_dir,
        data_dir=data_dir,
        dataset_name=dataset_name or settings.EVAL_DEFAULT_DATASET_NAME,
        output_dir=output_dir or None,
        device=device or None,
    )
    if not res.get("success"):
        return f"❌ 评估失败: {res.get('error')}", {}

    summary = res.get("summary") or {}
    summary.update({
        "results_path": res.get("results_path"),
        "results_csv": res.get("results_csv"),
        "output_dir": res.get("output_dir"),
        "skipped": len(res.get("skipped") or []),
        "points": res.get("points"),
    })
    return "✅ 评估完成", summary


def get_loss_plot(model_path: str, model_family: str):
    """获取 Loss 曲线图"""
    if not model_path:
        return None
    
    models = get_training_adapter(model_family).list_models()
    model = next((m for m in models if str(m.get("path")) == str(model_path)), None)
    
    if not model or not model.get("loss_image"):
        return None
    
    loss_image = model.get("loss_image")
    if Path(loss_image).exists():
        return loss_image
    return None


def get_comparison_plot(model_paths: List[str], model_family: str):
    """获取多个模型的 Loss 对比图 (使用 Matplotlib 动态生成)"""
    if not model_paths or len(model_paths) == 0:
        return None
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.figure(figsize=(10, 6))
    
    models = get_training_adapter(model_family).list_models()
    for path in model_paths:
        model = next((m for m in models if str(m.get("path")) == str(path)), None)
        if not model:
            continue
            
        logs = adapter.get_training_log(model["path"])
        if not logs:
            continue
            
        df = pd.DataFrame([{"step": l.get("current_steps", 0), "loss": l.get("loss")} for l in logs if "loss" in l])
        if not df.empty:
            label = model.get("name") or _relative_model_path(model.get("path", ""))
            plt.plot(df["step"], df["loss"], label=label)
            
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


def get_comparison_metrics(model_paths: List[str]) -> pd.DataFrame:
    """获取选定模型在黄金集上的评估指标对比"""
    if not model_paths:
        return pd.DataFrame()
    
    try:
        from src.db.database import SessionLocal, ModelEval
        db = SessionLocal()
        evals = db.query(ModelEval).filter(ModelEval.model_path.in_(model_paths)).all()
        rows = []
        for e in evals:
            try:
                metrics = json.loads(e.metrics) if e.metrics else {}
                summary = metrics.get("summary", {})
                rows.append({
                    "模型路径": e.model_path.split("/")[-1],
                    "评价集": e.dataset_name or "golden",
                    "F1 Score": summary.get("f1_score", "N/A"),
                    "Precision": summary.get("precision", "N/A"),
                    "Recall": summary.get("recall", "N/A"),
                    "覆盖点位": summary.get("points", 0)
                })
            except Exception:
                pass
        db.close()
        
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["模型路径", "评价集", "F1 Score", "Precision", "Recall", "覆盖点位"])
    except Exception as e:
        logger.error(f"Error fetching comparison metrics: {e}")
        return pd.DataFrame()


# ==================== 数据获取辅助函数 ====================

def _dataset_records() -> List[Dict]:
    rows = data_adapter.list_datasets()
    rows.sort(key=lambda x: x.get("modified_time", 0), reverse=True)
    return rows


def _refresh_dataset_point_mapping() -> Dict[str, str]:
    global _dataset_point_mapping
    _dataset_point_mapping.clear()
    for row in _dataset_records():
        filename = str(row.get("filename") or "").strip()
        if not filename:
            continue
        point_id = canonical_point_id(filename, row.get("name"))
        if point_id and point_id not in _dataset_point_mapping:
            _dataset_point_mapping[point_id] = filename
    return _dataset_point_mapping


def resolve_dataset_identifier_to_filename(identifier: str) -> Optional[str]:
    """
    将点位ID/文件名/路径统一解析为 data_adapter 可读的文件名。
    """
    text = str(identifier or "").strip()
    if not text:
        return None
    records = _dataset_records()
    by_filename = {str(r.get("filename") or ""): str(r.get("filename") or "") for r in records}
    if text in by_filename:
        return text
    base = Path(text).name
    if base in by_filename:
        return base
    mapping = _refresh_dataset_point_mapping()
    key = canonical_point_id(text)
    if key in mapping:
        return mapping[key]
    return None


def get_datasets_table() -> pd.DataFrame:
    """获取数据集列表并返回 DataFrame（点位视角）"""
    datasets = _dataset_records()
    if not datasets:
        return pd.DataFrame(columns=["点位ID", "数据文件", "大小 (KB)", "修改时间"])
    
    from datetime import datetime
    rows = []
    for d in datasets:
        filename = str(d.get("filename") or "")
        point_id = canonical_point_id(filename, d.get("name"))
        rows.append({
            "点位ID": point_id,
            "数据文件": filename,
            "大小 (KB)": round(d["size_bytes"] / 1024, 2),
            "修改时间": datetime.fromtimestamp(d["modified_time"]).strftime("%Y-%m-%d %H:%M")
        })
    return pd.DataFrame(rows)


def get_dataset_names() -> List[str]:
    """获取数据集点位ID列表（默认映射到最新文件）"""
    mapping = _refresh_dataset_point_mapping()
    return list(mapping.keys())


def delete_selected_dataset(dataset_selector: str):
    """删除选中的数据集"""
    logger.debug("delete_selected_dataset: %s", dataset_selector)
    if not dataset_selector:
        return get_datasets_table(), gr.Dropdown(choices=get_dataset_names(), value=None), "❌ No dataset selected"
    filename = resolve_dataset_identifier_to_filename(dataset_selector)
    if not filename:
        return get_datasets_table(), gr.Dropdown(choices=get_dataset_names(), value=None), "❌ 点位未匹配到数据文件"
    
    result = data_adapter.delete_dataset(filename)
    if result.get("success"):
        # 刷新列表
        new_table = get_datasets_table()
        new_choices = get_dataset_names()
        return new_table, gr.Dropdown(choices=new_choices, value=None), f"✅ Deleted: {filename}"
    else:
        return get_datasets_table(), gr.Dropdown(choices=get_dataset_names()), f"❌ {result.get('error')}"


def preview_dataset(dataset_selector: str) -> tuple:
    """预览数据集，返回 (表格数据, 列选择器更新, 曲线图)"""
    logger.debug("preview_dataset: %s", dataset_selector)
    
    if isinstance(dataset_selector, list):
        dataset_selector = dataset_selector[0] if dataset_selector else None
    
    if not dataset_selector:
        logger.debug("preview_dataset: 空文件名")
        return [], gr.CheckboxGroup(choices=[], value=[]), None
    filename = resolve_dataset_identifier_to_filename(dataset_selector)
    if not filename:
        logger.debug("preview_dataset: 未匹配到文件 selector=%s", dataset_selector)
        return [], gr.CheckboxGroup(choices=[], value=[]), None
    
    try:
        # 获取预览数据
        logger.debug("preview_csv: %s", filename)
        data = data_adapter.preview_csv(filename, limit=5000)
        logger.debug("preview_csv 返回 %d 条记录", len(data))
        
        df = pd.DataFrame(data)
        logger.debug("DataFrame shape=%s, columns=%s", df.shape, df.columns.tolist())
        
        # 过滤掉 Unnamed 和 category 列
        df = df.loc[:, ~df.columns.str.contains('^Unnamed|^category', case=False)]
        logger.debug("过滤后 shape=%s, columns=%s", df.shape, df.columns.tolist())
        
        # 获取数值列作为可选项
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        logger.debug("数值列: %s", numeric_cols)
        
        # 默认选中第一个数值列
        default_selected = numeric_cols[:1] if numeric_cols else []
        logger.debug("默认选中: %s", default_selected)
        
        # 生成默认曲线图
        plot_path = None
        
        # 优化：尝试使用预生成的图片 (如果存在)
        # 假设图片在 DATA_IMAGES_DIR 或 DATA_DOWNSAMPLED_DIR (用户可能手动放这)
        # 优先查 DATA_IMAGES_DIR
        possible_img_name = filename.replace(".csv", ".jpg")
        img_dir = Path(settings.DATA_IMAGES_DIR)
        img_dir.mkdir(parents=True, exist_ok=True)
        pre_gen_img_path = img_dir / possible_img_name
        
        # 如果图片不存在，尝试使用公共组件自动生成
        if not pre_gen_img_path.exists():
            logger.debug("生成缩略图: %s", pre_gen_img_path)
            try:
                # 优先使用 value 列，否则使用第一列
                target_col = 'value' if 'value' in df.columns else (numeric_cols[0] if numeric_cols else df.columns[0])
                if target_col in df.columns:
                    generate_ts_thumbnail(df[[target_col]], str(pre_gen_img_path))
            except Exception as e:
                logger.error("自动生成缩略图失败: %s", e)

        if pre_gen_img_path.exists():
            logger.debug("使用图片: %s", pre_gen_img_path)
            plot_path = str(pre_gen_img_path)
            
        # 如果没有找到，或者用户选择了特定的列组合(这里初始化默认选第一列，假设预生成图也是画的主列)
        # 为了严谨，如果使用了预生成图，我们也许应该显示它。
        # 但如果用户后续修改了 Checkbox，会触发 update_plot_from_selection，那时候会重画，这是对的。
        
        if not plot_path:
             plot_path = generate_plot(df, filename, default_selected)
             logger.debug("曲线图生成: %s", plot_path)
        
        # 转换为列表格式，确保 Gradio 6.x 兼容
        # 使用 values 列表 + headers 的方式
        table_data = df.values.tolist()
        headers = df.columns.tolist()
        logger.debug("表格数据: %d 行, 列: %s", len(table_data), headers)
        
        return gr.Dataframe(value=table_data, headers=headers), gr.update(choices=numeric_cols, value=default_selected), plot_path
    except Exception as e:
        import traceback
        logger.error("preview_dataset 异常: %s", e, exc_info=True)
        traceback.print_exc()
        return [], gr.CheckboxGroup(choices=[], value=[]), None


def generate_plot(df: pd.DataFrame, filename: str, selected_cols: list):
    """根据选择的列生成曲线图 (统一风格)"""
    if df.empty or not selected_cols:
        return None
    
    try:
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        import uuid
        plot_path = temp_dir / f"preview_{uuid.uuid4().hex[:8]}.jpg"
        
        # 使用统一绘图组件 (仅支持单列/第一列风格)
        data_subset = df[selected_cols]
        generate_ts_thumbnail(data_subset, str(plot_path))
        
        return str(plot_path)
    except Exception as e:
        logger.warning("曲线图生成失败: %s", e)
        return None


def update_plot_from_selection(dataset_selector: str, selected_cols: list):
    """根据用户选择的列更新曲线图"""
    if not dataset_selector or not selected_cols:
        return None
    
    try:
        filename = resolve_dataset_identifier_to_filename(dataset_selector)
        if not filename:
            return None
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
    """启动数据采集任务（通过 API 提交并轮询状态）"""
    if not source:
        yield "❌ Please enter IoTDB source path"
        return

    # 安全基线：采集接口要求用户名/密码非空；优先使用表单，其次回退到共享配置
    normalized_user = str(user or "").strip()
    normalized_password = str(password or "").strip()
    if not normalized_user or not normalized_password:
        cfg = load_iotdb_config()
        normalized_user = normalized_user or str(cfg.get("user") or "").strip()
        normalized_password = normalized_password or str(cfg.get("password") or "").strip()
    if not normalized_user or not normalized_password:
        yield "❌ 请填写 IoTDB 用户名和密码（当前分支已禁用空凭据提交）"
        return

    payload = {
        "source": source,
        "host": host,
        "port": str(port),
        "user": normalized_user,
        "password": normalized_password,
        "point_name": point_name or "*",
        "target_points": int(target_points),
        "start_time": start_time or None,
        "end_time": end_time or None,
    }

    submit_resp = _data_api_call("post", "/acquire", payload=payload)
    if not submit_resp.get("success"):
        err = submit_resp.get("error")
        if isinstance(err, list):
            loc_keys = []
            for item in err:
                if not isinstance(item, dict):
                    continue
                loc = item.get("loc")
                if isinstance(loc, list) and loc:
                    loc_keys.append(str(loc[-1]))
            if "user" in loc_keys or "password" in loc_keys:
                yield "❌ 数据采集任务提交失败: IoTDB 用户名和密码不能为空"
                return
        yield f"❌ 数据采集任务提交失败: {err or '未知错误'}"
        return

    submit_data = submit_resp.get("data") or {}
    task_id = str(submit_data.get("task_id") or "")
    if not task_id:
        yield "❌ 数据采集提交成功，但未返回 task_id"
        return

    accumulated_log = (
        f"🚀 已提交数据采集任务: {task_id}\n"
        f"Source: {source}\n"
        f"Point: {point_name or '*'}\n"
        f"Target points: {int(target_points)}"
    )
    yield accumulated_log

    poll_interval = 2.0
    max_polls = 1800  # 约 1 小时
    last_status = None
    log_offset = 0

    for i in range(max_polls):
        status = ""
        message = ""

        log_resp = _data_api_call(
            "get",
            f"/log/{task_id}",
            params={"offset": log_offset, "max_bytes": 200000},
        )
        if log_resp.get("success"):
            log_wrapper = log_resp.get("data") or {}
            log_payload = log_wrapper.get("data") if isinstance(log_wrapper, dict) else {}
            if isinstance(log_payload, dict):
                status = str(log_payload.get("status") or "").lower()
                message = str(log_payload.get("message") or "")
                chunk = str(log_payload.get("log") or "")
                if chunk:
                    accumulated_log += f"\n{chunk}"
                try:
                    log_offset = int(log_payload.get("offset") or log_offset)
                except Exception:
                    pass
        else:
            # 兼容兜底：日志接口异常时，仍保留状态轮询
            status_resp = _data_api_call("get", f"/status/{task_id}")
            if status_resp.get("success"):
                status_data = _extract_task_status_payload(status_resp.get("data"))
                status = str(status_data.get("status") or "").lower()
                message = str(status_data.get("message") or "")
            elif i % 5 == 0:
                accumulated_log += f"\n⚠️ 状态查询失败: {status_resp.get('error') or '未知错误'}"

        if status != last_status:
            accumulated_log += f"\n[{time.strftime('%H:%M:%S')}] 任务状态: {status or 'unknown'} {message}".strip()
            last_status = status

        if len(accumulated_log) > LOG_TAIL_MAX_CHARS:
            accumulated_log = accumulated_log[-LOG_TAIL_MAX_CHARS:]

        if status == "completed":
            yield accumulated_log + "\n✅ 数据采集完成，请点击“刷新列表”查看最新点位数据。"
            return
        if status in {"failed", "cancelled"}:
            err = message or "数据采集失败"
            yield accumulated_log + f"\n❌ 数据采集结束: {status} - {err}"
            return

        yield accumulated_log
        time.sleep(poll_interval)

    yield accumulated_log + "\n❌ 数据采集任务轮询超时，请稍后在任务历史中检查状态。"


# ==================== 推理监控辅助函数 ====================

def get_algorithms() -> List[str]:
    """获取可用算法列表"""
    return ["chatts", "qwen", "adtk_hbos", "ensemble", "timer"]


def get_inference_models(model_family: str) -> List[tuple]:
    """获取可用于推理的 LoRA 训练任务列表 (不含 checkpoint)"""
    return get_lora_run_choices(model_family)

def toggle_algo_params(algorithm: str):
    """根据选择的算法切换参数组可见性，并返回默认模型路径"""
    show_chatts = (algorithm == "chatts" or algorithm == "qwen")
    show_timer = (algorithm == "timer")
    show_adtk = (algorithm == "adtk_hbos")
    
    # 获取该算法的默认模型路径
    default_model = ALGORITHM_DEFAULT_MODELS.get(algorithm, "")
    
    return (
        gr.update(visible=show_chatts), 
        gr.update(visible=show_timer), 
        gr.update(visible=show_adtk),
        gr.update(value=default_model),  # 更新 base_model_input
    )

def start_inference_task(
    algorithm: str, 
    base_model_path: str,
    lora_run_path: str,
    lora_checkpoint: str,
    point_ids: List[str],
    n_downsample: int,
    threshold: float,
    downsample_mode: str,
    downsampler: str,
    ratio: float,
    min_threshold: int,
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
    def _ui_payload(log_text: str, status_text: str, stop_visible: bool, submit_visible: bool, task_id_val, files_val):
        return (
            format_log_html(log_text),
            status_text,
            gr.update(visible=stop_visible),
            gr.update(visible=submit_visible),
            task_id_val,
            files_val,
        )

    if not algorithm:
        yield _ui_payload("❌ 请选择算法", "❌ 请选择算法", False, True, None, None)
        return
    if not point_ids:
        yield _ui_payload("❌ 请选择输入点位", "❌ 请选择输入点位", False, True, None, None)
        return
    
    # 验证模型路径与算法是否匹配
    if algorithm in ["chatts", "qwen"] and base_model_path:
        expected_model = ALGORITHM_DEFAULT_MODELS.get(algorithm, "")
        path_lower = base_model_path.lower()
        
        # 简单的关键词检查
        if algorithm == "qwen" and "chatts" in path_lower:
            warning = (
                "⚠️ 警告：选择了 Qwen 算法，但模型路径似乎是 ChatTS 模型。\n"
                f"当前路径: {base_model_path}\n"
                f"建议路径: {expected_model}\n"
                "请确认模型路径是否正确，或点击算法下拉框重新选择以自动切换。"
            )
            yield _ui_payload(warning, "⚠️ 模型路径警告", False, True, None, None)
            return
        if algorithm == "chatts" and "qwen" in path_lower:
            warning = (
                "⚠️ 警告：选择了 ChatTS 算法，但模型路径似乎是 Qwen 模型。\n"
                f"当前路径: {base_model_path}\n"
                f"建议路径: {expected_model}\n"
                "请确认模型路径是否正确，或点击算法下拉框重新选择以自动切换。"
            )
            yield _ui_payload(warning, "⚠️ 模型路径警告", False, True, None, None)
            return
    
    # 将选中的点位ID转换为完整路径（使用统一数据源映射）
    file_paths = resolve_point_ids_to_paths(point_ids)
    
    if not file_paths:
        yield _ui_payload("❌ 未找到有效的输入点位文件", "❌ 未找到有效的输入点位文件", False, True, None, None)
        return

    # 解析 LoRA Adapter 路径（支持 checkpoint 分层选择）
    lora_adapter_path = resolve_lora_adapter_path(lora_run_path, lora_checkpoint)
    
    try:
        # Resolve downsample args
        resolved_downsampler = downsampler or "m4"
        resolved_n_downsample = n_downsample
        resolved_ratio = ratio
        resolved_min_threshold = min_threshold

        if str(downsample_mode).lower() in ["off", "none", "关闭", "no"]:
            resolved_downsampler = "none"
        elif str(downsample_mode).lower() in ["ratio", "比例"]:
            # ratio + min_threshold only affects adtk_hbos/stl_wavelet
            pass
        # auto/fixed keep defaults

        # 准备高级参数
        advanced_args = {
            "n_downsample": resolved_n_downsample,
            "threshold": threshold,
            "downsampler": resolved_downsampler,
            "ratio": resolved_ratio,
            "min_threshold": resolved_min_threshold,
            "base_model_path": base_model_path,
            "lora_adapter_path": lora_adapter_path, 
            # ChatTS
            "chatts_load_in_4bit": load_in_4bit,
            "chatts_prompt_template": prompt_template,
            "chatts_max_new_tokens": max_new_tokens,
            "qwen_max_new_tokens": max_new_tokens,
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

        model_for_inference = lora_adapter_path or base_model_path or ALGORITHM_DEFAULT_MODELS.get(algorithm, "")
        submit_payload = {
            "model": model_for_inference,
            "algorithm": algorithm,
            "input_files": file_paths,
            "params": advanced_args,
        }

        submit_resp = _inference_api_call("post", "/batch", payload=submit_payload)
        if not submit_resp.get("success"):
            err = submit_resp.get("error") or "提交失败"
            yield _ui_payload(f"❌ 提交推理任务失败: {err}", f"❌ {err}", False, True, None, None)
            return

        submit_data = submit_resp.get("data") or {}
        task_id = str(submit_data.get("task_id") or "")
        if not task_id:
            yield _ui_payload("❌ 提交成功但未返回 task_id", "❌ 提交失败", False, True, None, None)
            return

        accumulated_log = (
            f"🚀 已提交推理任务: {task_id}\n"
            f"算法: {algorithm}\n"
            f"输入点位数: {len(point_ids)}\n"
            f"输入文件数: {len(file_paths)}\n"
        )
        yield _ui_payload(accumulated_log, "🔄 已提交，等待执行...", True, False, task_id, None)

        poll_interval = 2.0
        max_polls = 1800  # 约 1 小时
        last_status = None
        final_status = None
        final_message = ""
        log_offset = 0
        for i in range(max_polls):
            status = ""
            message = ""

            log_resp = _inference_api_call(
                "get",
                f"/log/{task_id}",
                params={"offset": log_offset, "max_bytes": 200000},
            )
            if log_resp.get("success"):
                log_wrapper = log_resp.get("data") or {}
                log_payload = log_wrapper.get("data") if isinstance(log_wrapper, dict) else {}
                if isinstance(log_payload, dict):
                    status = str(log_payload.get("status") or "").lower()
                    message = str(log_payload.get("message") or "")
                    chunk = str(log_payload.get("log") or "")
                    if chunk:
                        accumulated_log += f"\n{chunk}"
                    try:
                        log_offset = int(log_payload.get("offset") or log_offset)
                    except Exception:
                        pass
            else:
                # 兼容兜底：日志接口异常时，仍保留状态轮询
                status_resp = _inference_api_call("get", f"/status/{task_id}")
                if status_resp.get("success"):
                    st_data = _extract_task_status_payload(status_resp.get("data"))
                    status = str(st_data.get("status") or "").lower()
                    message = str(st_data.get("message") or "")
                elif i % 5 == 0:
                    err = status_resp.get("error") or "状态查询失败"
                    accumulated_log += f"\n⚠️ 状态查询异常: {err}"

            if status != last_status:
                accumulated_log += f"\n[{time.strftime('%H:%M:%S')}] 状态: {status or 'unknown'}"
                if message:
                    accumulated_log += f" | {message}"
                last_status = status
            if status in {"completed", "failed", "cancelled"}:
                final_status = status
                final_message = message
                break

            if len(accumulated_log) > LOG_TAIL_MAX_CHARS:
                accumulated_log = accumulated_log[-LOG_TAIL_MAX_CHARS:]
            yield _ui_payload(accumulated_log, f"🔄 执行中... ({i+1})", True, False, task_id, None)
            time.sleep(poll_interval)

        if final_status is None:
            yield _ui_payload(
                accumulated_log + "\n❌ 任务轮询超时",
                "❌ 任务超时",
                False,
                True,
                task_id,
                None,
            )
            return

        if final_status == "cancelled":
            yield _ui_payload(accumulated_log + "\n🛑 任务已取消", "🛑 任务已取消", False, True, task_id, None)
            return

        if final_status == "failed":
            error_text = final_message or "推理失败"
            yield _ui_payload(accumulated_log + f"\n❌ 任务失败: {error_text}", f"❌ {error_text}", False, True, task_id, None)
            return

        # completed
        results_resp = _inference_api_call("get", f"/results/{task_id}")
        if not results_resp.get("success"):
            err = results_resp.get("error") or "结果获取失败"
            yield _ui_payload(accumulated_log + f"\n⚠️ 任务完成但结果读取失败: {err}", f"⚠️ {err}", False, True, task_id, None)
            return

        payload = (results_resp.get("data") or {}).get("data")
        rows = []
        if isinstance(payload, dict):
            rows = payload.get("results") if isinstance(payload.get("results"), list) else []
        elif isinstance(payload, list):
            rows = payload

        generated_files = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            nested = row.get("result") if isinstance(row.get("result"), dict) else {}
            candidate = (
                row.get("result_path")
                or row.get("file_path")
                or row.get("output_path")
                or nested.get("result_path")
                or nested.get("file_path")
                or nested.get("output_path")
            )
            if not candidate:
                continue
            candidate_path = Path(str(candidate))
            if not candidate_path.exists():
                alt = RESULTS_BASE_PATH / algorithm / candidate_path.name
                if alt.exists():
                    candidate_path = alt
            if candidate_path.exists():
                p = str(candidate_path)
                if p not in generated_files:
                    generated_files.append(p)

        # fallback: use indexed_results from /results when rows do not expose file paths
        if not generated_files and isinstance(payload, dict):
            for row in payload.get("indexed_results") or []:
                if not isinstance(row, dict):
                    continue
                candidate = row.get("result_path")
                if not candidate:
                    continue
                candidate_path = Path(str(candidate))
                if candidate_path.exists():
                    p = str(candidate_path)
                    if p not in generated_files:
                        generated_files.append(p)
        
        # 自动将结果文件链接到用户数据目录，以便标注工具默认可见
        try:
            # 目标目录列表：data_adapter 目录 + Annotator 用户目录
            target_dirs = [data_adapter.data_path]
            
            # 尝试读取 Annotator 用户配置
            try:
                annotator_users_file = Path(settings.DATA_PROCESSING_PATH).parent / "annotator" / "backend" / "users.json"
                if annotator_users_file.exists():
                    # import json  <-- Removed to avoid shadowing global json
                    with open(annotator_users_file, 'r') as f:
                        users = json.load(f)
                    # 遍历所有用户，将结果链接到每个用户的 data_path
                    for username, user_info in users.items():
                        if 'data_path' in user_info:
                            user_dir = Path(user_info['data_path'])
                            # Only add if directory exists and is writable by current user
                            if user_dir.exists() and user_dir not in target_dirs:
                                if os.access(user_dir, os.W_OK):
                                    target_dirs.append(user_dir)
                                else:
                                    logger.debug("Auto-Link 跳过 %s: 无写权限", user_dir)
            except Exception as e:
                logger.warning("Auto-Link: 读取 users.json 失败: %s", e)
            
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
                            logger.info("Auto-Link: 创建符号链接 %s -> %s", res_path.name, target_dir)
                        except Exception as link_err:
                            logger.warning("Auto-Link: 链接失败 %s: %s", target_dir, link_err)
        except Exception as e:
            logger.error("Auto-Link: 链接结果文件失败: %s", e)
        
        # 读取并汇总评分信息
        score_summary = ""
        for gen_file in generated_files:
            try:
                gen_path = Path(gen_file)
                # 尝试查找 metrics.json 文件
                metrics_path = gen_path.parent / f"{gen_path.stem}_metrics.json"
                if not metrics_path.exists():
                    # 尝试在推理输出目录中查找
                    metrics_path = RESULTS_BASE_PATH / algorithm / f"{gen_path.stem}_metrics.json"
                
                if metrics_path.exists():
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    summary = metrics.get("summary", {})
                    point_name = metrics.get("point_name", gen_path.stem)
                    score_summary += f"\n📊 **{point_name}** 评分:\n"
                    score_summary += f"   - 平均分 (score_avg): {summary.get('score_avg', 0):.4f}\n"
                    score_summary += f"   - 最高分 (score_max): {summary.get('score_max', 0):.4f}\n"
                    score_summary += f"   - 异常段数: {summary.get('segment_count', 0)}\n"
            except Exception as score_err:
                logger.warning("读取评分指标失败 %s: %s", gen_file, score_err)
        
        final_log = accumulated_log + "\n✅ 所有任务已完成"
        if score_summary:
            final_log += "\n\n---\n### 📈 评分摘要" + score_summary

        yield _ui_payload(final_log, "✅ 任务完成", False, True, task_id, generated_files or None)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield _ui_payload(f"❌ 发生错误: {str(e)}", f"❌ 错误: {str(e)}", False, True, None, None)

def stop_task_action(task_id_state):
    """实际执行停止动作"""
    logger.debug("Stop requested: task_id=%s", task_id_state)
    if task_id_state:
        cancel_resp = _inference_api_call("post", f"/cancel/{task_id_state}")
        if cancel_resp.get("success"):
            payload = cancel_resp.get("data") or {}
            status = str(payload.get("status") or "").lower()
            message = str(payload.get("message") or "任务已取消")
            if status in {"completed", "failed"}:
                ui_msg = f"ℹ️ {message}"
            else:
                ui_msg = f"🛑 {message}"
            logger.info("推理任务取消请求完成: task_id=%s, status=%s", task_id_state, status or "unknown")
            return ui_msg, gr.update(visible=False), gr.update(visible=True), None, None

        api_error = str(cancel_resp.get("error") or "取消请求失败")
        if "任务不存在" in api_error:
            logger.info("取消请求返回任务不存在: %s", task_id_state)
            return "ℹ️ 任务不存在或已结束", gr.update(visible=False), gr.update(visible=True), None, None

        logger.warning("API 取消失败，回退本地停止 task_id=%s: %s", task_id_state, api_error)
        if inference_adapter.stop_inference_task(task_id_state):
            return f"⚠️ API 取消失败，已回退本地停止: {api_error}", gr.update(visible=False), gr.update(visible=True), None, None

        logger.warning("停止推理任务失败: %s", task_id_state)
        return f"❌ 停止失败: {api_error}", gr.update(visible=True), gr.update(visible=False), task_id_state, None
    logger.debug("无活动任务ID")
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
    
    with gr.Blocks(title="TS-Iteration-Loop", theme=gr.themes.Soft(), css=LOG_SCROLL_CSS) as demo:
        gr.Markdown("# 🔄 TS-Iteration-Loop 时序迭代平台")
        gr.Markdown("整合数据获取、推理监控、微调训练的统一管理界面")
        
        # ==================== 数据获取 Tab ====================
        with gr.Tab("📁 数据获取"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 点位数据列表")
                    datasets_table = gr.Dataframe(
                        value=get_datasets_table(),
                        label="已有点位（映射到最新CSV）",
                        interactive=False
                    )
                    refresh_datasets_btn = gr.Button("🔄 刷新列表")
                    
                    gr.Markdown("### 预览点位")
                    preview_dropdown = gr.Dropdown(
                        label="选择点位ID",
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
                            user_input = gr.Textbox(label="User", value=_iotdb_cfg.get("user", ""))
                            pwd_input = gr.Textbox(label="Password", value=_iotdb_cfg.get("password", ""), type="password")

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
                        lora_family_select = gr.Dropdown(
                            label="LoRA 模型类型 (Model Family)",
                            choices=TRAINING_MODEL_FAMILIES,
                            value="chatts",
                            interactive=True,
                            visible=False  # 隐藏：自动跟随算法选择
                        )
                        lora_run_select = gr.Dropdown(
                            label="LoRA 适配器 (训练任务)",
                            choices=get_inference_models("chatts"),
                            interactive=True,
                            info="仅显示训练任务目录，不含 checkpoint"
                        )
                        lora_checkpoint_select = gr.Dropdown(
                            label="Checkpoint (可选)",
                            choices=get_checkpoint_choices(""),
                            value="",
                            interactive=True
                        )
                        
                    files_select = gr.CheckboxGroup(
                        label="选择输入点位（自动匹配最新CSV）",
                        choices=get_unified_point_ids()
                    )
                    
                    with gr.Accordion("⚙️ 高级配置 (可选)", open=False):
                        gr.Markdown("#### 降采样配置")
                        with gr.Row():
                            downsample_mode_input = gr.Dropdown(
                                label="降采样模式",
                                choices=["auto", "off", "fixed", "ratio"],
                                value="auto",
                                info="auto: 长度>n_downsample 才降采样; off: 关闭"
                            )
                            downsampler_input = gr.Dropdown(
                                label="降采样算法",
                                choices=["m4", "minmax", "none"],
                                value="m4"
                            )
                        with gr.Row():
                            n_downsample_input = gr.Slider(
                                label="降采样点数 (n_downsample)", 
                                minimum=100, maximum=10000, step=100, value=settings.DEFAULT_DOWNSAMPLE_POINTS
                            )
                            ratio_input = gr.Slider(
                                label="降采样比例 (ratio)", 
                                minimum=0.01, maximum=1.0, step=0.01, value=0.1
                            )
                        min_threshold_input = gr.Number(
                            label="最小点数阈值 (min_threshold)", value=200000, precision=0
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
                            inference_logs = gr.HTML(
                                value=format_log_html("Waiting for task..."),
                                label="Execution Logs",
                                elem_id="inference-log"
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
                                     choices=["chatts", "qwen", "timer", "adtk_hbos"],
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
                    algo_dropdown, base_model_input, lora_run_select, lora_checkpoint_select, files_select,
                    # 通用参数
                    n_downsample_input, threshold_input,
                    downsample_mode_input, downsampler_input, ratio_input, min_threshold_input,
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
                fn=lambda: gr.CheckboxGroup(choices=get_unified_point_ids()),
                outputs=files_select
            )
            refresh_tasks_btn.click(
                fn=update_lora_run_dropdown,
                inputs=[lora_family_select, lora_run_select],
                outputs=lora_run_select
            )
            refresh_tasks_btn.click(
                fn=update_checkpoint_dropdown,
                inputs=[lora_run_select, lora_checkpoint_select],
                outputs=lora_checkpoint_select
            )

            lora_family_select.change(
                fn=update_lora_run_dropdown,
                inputs=[lora_family_select, lora_run_select],
                outputs=lora_run_select
            ).then(
                fn=update_checkpoint_dropdown,
                inputs=[lora_run_select, lora_checkpoint_select],
                outputs=lora_checkpoint_select
            )

            lora_run_select.change(
                fn=update_checkpoint_dropdown,
                inputs=[lora_run_select, lora_checkpoint_select],
                outputs=lora_checkpoint_select
            )
            
            # Tab 切换时自动刷新文件列表
            inference_tab.select(
                fn=lambda: gr.CheckboxGroup(choices=get_unified_point_ids()),
                outputs=files_select
            )
            inference_tab.select(
                fn=update_lora_run_dropdown,
                inputs=[lora_family_select, lora_run_select],
                outputs=lora_run_select
            )
            inference_tab.select(
                fn=update_checkpoint_dropdown,
                inputs=[lora_run_select, lora_checkpoint_select],
                outputs=lora_checkpoint_select
            )
            # 清空历史记录
            clear_tasks_btn.click(
                fn=clear_task_history,
                outputs=[task_table, clear_status]
            )
            
            # 历史结果文件刷新
            refresh_results_btn.click(
                fn=lambda m: gr.CheckboxGroup(choices=get_result_filenames(m), value=[]),
                inputs=results_method_select,
                outputs=file_manager_list
            )
            
            # 切换方法时刷新结果列表
            results_method_select.change(
                fn=lambda m: gr.CheckboxGroup(choices=get_result_filenames(m), value=[]),
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
            
            # 算法切换事件：控制参数组显示 + 切换默认模型路径
            algo_dropdown.change(
                fn=toggle_algo_params,
                inputs=algo_dropdown,
                outputs=[chatts_group, timer_group, adtk_group, base_model_input]
            )

            algo_dropdown.change(
                fn=sync_lora_family_from_algo,
                inputs=[algo_dropdown, lora_family_select],
                outputs=lora_family_select
            ).then(
                fn=update_lora_run_dropdown,
                inputs=[lora_family_select, lora_run_select],
                outputs=lora_run_select
            ).then(
                fn=update_checkpoint_dropdown,
                inputs=[lora_run_select, lora_checkpoint_select],
                outputs=lora_checkpoint_select
            )
            
            # 自动同步筛选方法 (User requested unification)
            algo_dropdown.change(
                fn=lambda x: x if x in ["chatts", "qwen", "timer", "adtk_hbos"] else "chatts",
                inputs=algo_dropdown,
                outputs=results_method_select
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
                    # 动态获取 DB 标注统计
                    def get_annotation_stats():
                        db_count = 0
                        try:
                            from src.db.database import SessionLocal, AnnotationRecord

                            db = SessionLocal()
                            try:
                                db_count = (
                                    db.query(AnnotationRecord)
                                    .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
                                    .count()
                                )
                            finally:
                                db.close()
                        except Exception:
                            db_count = 0

                        return f"DB 标注记录: {db_count}"

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
                    conv_model_family = gr.Radio(
                        choices=["chatts", "qwen"],
                        value="chatts",
                        label="目标模型格式 (Target Format)"
                    )
                    
                    with gr.Accordion("⚙️ 路径与参数配置 (Settings)", open=False):
                        conf_input_dir = gr.Textbox(
                            label="标注来源 (DB-First，目录仅兼容展示)", 
                            value=str(Path(settings.ANNOTATIONS_ROOT) / settings.DEFAULT_USER)
                        )
                        conf_image_dir = gr.Textbox(
                            label="图片文件来源 (Image Dir)", 
                            value=settings.DATA_DOWNSAMPLED_DIR
                        )
                        conf_output_path = gr.Textbox(
                            label="转换输出路径 (Output Path)",
                            value=str(Path(settings.DATA_TRAINING_CHATTS_DIR) / "converted_data.json")
                        )

                    def _normalize_ann_name(name: str) -> str:
                        if not name:
                            return ""
                        text = str(name)
                        for ext in (".csv", ".json"):
                            if text.lower().endswith(ext):
                                text = text[: -len(ext)]
                        if text.startswith("annotations_"):
                            text = text.replace("annotations_", "", 1)
                        return text

                    def _get_approved_set():
                        try:
                            from src.db.database import SessionLocal, ReviewQueue
                            from src.utils.annotation_store import canonical_point_id
                        except Exception:
                            return set()
                        db = SessionLocal()
                        try:
                            rows = db.query(ReviewQueue.point_id, ReviewQueue.source_id).filter(
                                ReviewQueue.source_type == "annotation",
                                ReviewQueue.status == "approved"
                            ).all()
                            return {canonical_point_id(r[0], r[1]) for r in rows if canonical_point_id(r[0], r[1])}
                        finally:
                            db.close()

                    def _list_annotation_payloads_from_db(filter_keyword=None, approved_only_flag=False):
                        try:
                            from src.db.database import SessionLocal, AnnotationRecord, ReviewQueue
                            from src.utils.annotation_store import canonical_point_id, record_to_payload
                        except Exception:
                            return []

                        db = SessionLocal()
                        try:
                            rows = (
                                db.query(AnnotationRecord)
                                .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
                                .order_by(AnnotationRecord.updated_at.desc())
                                .all()
                            )
                            review_rows = (
                                db.query(ReviewQueue.point_id, ReviewQueue.source_id, ReviewQueue.status, ReviewQueue.updated_at)
                                .filter(ReviewQueue.source_type == "annotation")
                                .order_by(ReviewQueue.updated_at.desc())
                                .all()
                            )
                        finally:
                            db.close()

                        review_status_map = {}
                        for point_id, source_id, status, _updated_at in review_rows:
                            key = canonical_point_id(point_id, source_id)
                            if not key or key in review_status_map:
                                continue
                            review_status_map[key] = (status or "").strip().lower() or "unreviewed"

                        status_label_map = {
                            "approved": "通过",
                            "pending": "待审",
                            "rejected": "驳回",
                            "needs_fix": "待修正",
                            "unreviewed": "未入队",
                        }

                        approved_set = _get_approved_set() if approved_only_flag else None
                        family = (filter_keyword or "").strip().lower()

                        payloads = []
                        seen = set()
                        for row in rows:
                            norm = canonical_point_id(getattr(row, "point_id", None), row.source_id, row.filename)
                            if not norm:
                                continue
                            if approved_set is not None and norm not in approved_set:
                                continue

                            name_text = str(row.filename or row.source_id or "").lower()
                            method = str(row.method or "").strip().lower()
                            if family == "qwen":
                                if method and method != "qwen" and "qwen" not in name_text:
                                    continue
                            elif family == "chatts":
                                if method == "qwen" or "qwen" in name_text:
                                    continue

                            payload = record_to_payload(row, fallback_filename=row.filename or f"{norm}.csv")
                            display_name = str(payload.get("filename") or row.filename or f"{norm}.csv")
                            key = _normalize_ann_name(display_name)
                            if key in seen:
                                continue
                            seen.add(key)
                            source_kind = (row.source_kind or "human").strip().lower()
                            kind_tag = "AUTO" if source_kind == "auto" else "HUMAN"
                            review_status = review_status_map.get(key, "unreviewed")
                            status_text = status_label_map.get(review_status, review_status)
                            label = f"{key} | [{kind_tag}] | {status_text}"
                            payloads.append(
                                {
                                    "label": label,
                                    "value": key,
                                    "filename": display_name,
                                    "payload": payload,
                                }
                            )

                        return payloads

                    # 获取标注文件列表（DB-First）
                    def get_file_choices(ann_dir, filter_keyword=None, approved_only_flag=False):
                        _ = ann_dir
                        db_payloads = _list_annotation_payloads_from_db(filter_keyword, approved_only_flag)
                        return [(row["label"], row["value"]) for row in db_payloads]

                    def _resolve_selected_payload(selected_file: str, family: str, approved_only_flag: bool):
                        if not selected_file:
                            return None
                        rows = _list_annotation_payloads_from_db(family, approved_only_flag)
                        selected_norm = _normalize_ann_name(selected_file)
                        for row in rows:
                            value = row.get("value")
                            display_name = row.get("filename")
                            payload = row.get("payload")
                            if value == selected_file or display_name == selected_file:
                                return payload
                            if _normalize_ann_name(value) == selected_norm or _normalize_ann_name(display_name) == selected_norm:
                                return payload
                        return None

                    # 初始加载
                    default_ann_dir = str(Path(settings.ANNOTATIONS_ROOT) / settings.DEFAULT_USER)
                    # 默认 filter="chatts" 对应 conv_model_family default value
                    initial_choices = get_file_choices(default_ann_dir, "chatts", True)
                    initial_approved_only = True
                    if not initial_choices:
                        initial_choices = get_file_choices(default_ann_dir, "chatts", False)
                        initial_approved_only = False
                    initial_val = initial_choices[0][1] if initial_choices else None

                    ann_file_dropdown = gr.Dropdown(
                        label="选择要预览/转换的文件",
                        choices=initial_choices,
                        value=initial_val,
                        multiselect=False,
                        interactive=True,
                        allow_custom_value=False
                    )
                    approved_only = gr.Checkbox(label="仅导出审核通过", value=initial_approved_only, interactive=True)

                    def refresh_files(ann_dir, family, approved_only_flag):
                        choices = get_file_choices(ann_dir, family, approved_only_flag)
                        val = choices[0][1] if choices else None
                        return gr.update(choices=choices, value=val)
                        
                    refresh_files_btn = gr.Button("🔄 刷新列表 (Refresh)", size="sm")
                    
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
                    after_json_label = gr.Markdown("#### 🎯 转换后 (ChatTS Training Data)")
                    after_json = gr.JSON(label="Converted Data", height=400)
                
            def preview_source_file(selected_file, input_dir_val, image_dir_val, model_family="qwen", approved_only_flag=True):
                """选择文件时立即预览，并执行真实转换（优先 DB-First）。"""
                _ = input_dir_val
                if not selected_file:
                    return None, None

                source_content = _resolve_selected_payload(selected_file, model_family, approved_only_flag)

                if source_content is None:
                    return {"error": "未找到标注内容（DB 查询为空）"}, None

                converted_content = None
                try:
                    img_d = image_dir_val if image_dir_val else settings.DATA_DOWNSAMPLED_DIR
                    csv_src = str(RESULTS_BASE_PATH / "qwen") if model_family == "qwen" else settings.DATA_DOWNSAMPLED_DIR
                    norm = _normalize_ann_name(source_content.get("filename") or selected_file)
                    temp_json_name = f"{norm}.json"

                    with tempfile.TemporaryDirectory() as tmp_input:
                        tmp_input_path = Path(tmp_input)
                        with open(tmp_input_path / temp_json_name, "w", encoding="utf-8") as wf:
                            json.dump(source_content, wf, ensure_ascii=False, indent=2)

                        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                            tmp_out = tmp.name

                        res = annotation_export_adapter.convert_annotations(
                            input_dir=str(tmp_input_path),
                            output_path=tmp_out,
                            image_dir=img_d,
                            filename=temp_json_name,
                            model_family=model_family,
                            csv_src_dir=csv_src,
                        )

                        if res.get("success"):
                            try:
                                with open(tmp_out, "r", encoding="utf-8") as f:
                                    converted_content = json.load(f)
                            except Exception as read_err:
                                converted_content = {"error": f"Read converted file failed: {read_err}"}
                        else:
                            converted_content = {
                                "error": "Conversion failed",
                                "stderr": res.get("stderr", ""),
                                "stdout": res.get("stdout", ""),
                            }
                        Path(tmp_out).unlink(missing_ok=True)
                except Exception as e:
                    converted_content = {"error": f"Preview failed: {str(e)}"}

                return source_content, converted_content

            def convert_core(selected_file, input_dir, image_dir, output_path, model_family, approved_only_flag, mode="single"):
                """核心转换逻辑（优先 DB-First）。"""
                _ = input_dir
                try:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return f"❌ 创建输出目录失败: {str(e)}", {}, {}

                csv_src = str(RESULTS_BASE_PATH / "qwen") if model_family == "qwen" else settings.DATA_DOWNSAMPLED_DIR
                img_d = image_dir or settings.DATA_DOWNSAMPLED_DIR

                def _run_convert(work_dir, target_filename, preview_payload, preview_file):
                    result = annotation_export_adapter.convert_annotations(
                        work_dir,
                        output_path,
                        image_dir=img_d,
                        filename=target_filename,
                        model_family=model_family,
                        csv_src_dir=csv_src,
                    )

                    if not result.get("success"):
                        return f"❌ 转换失败: {result.get('error')}\n日志:\n{result.get('stderr', '')}", preview_payload or {}, {}

                    log_output = result.get("stdout", "")
                    prefix = "单文件" if mode == "single" else "批量"
                    final_output_path = result.get("output_path") or output_path
                    status_msg = f"✅ {prefix}转换成功! \n输出: {final_output_path}\n\n执行日志:\n{log_output}"

                    target_sample = {}
                    try:
                        with open(final_output_path, "r", encoding="utf-8") as f:
                            converted_data = json.load(f)
                        if converted_data and isinstance(converted_data, list):
                            core_name = _normalize_ann_name(preview_file) if preview_file else ""
                            if core_name:
                                for item in converted_data:
                                    if core_name in str(item.get("image", "")):
                                        target_sample = item
                                        break
                            if not target_sample:
                                target_sample = converted_data[0]
                    except Exception as e:
                        target_sample = {"error": f"Read output failed: {str(e)}"}

                    return status_msg, preview_payload or {}, target_sample

                # DB-First 路径
                db_payloads = _list_annotation_payloads_from_db(model_family, approved_only_flag)
                if db_payloads:
                    if mode == "single":
                        if not selected_file:
                            return "❌ 未选择文件", {}, {}
                        selected_norm = _normalize_ann_name(selected_file)
                        chosen = [
                            row
                            for row in db_payloads
                            if row.get("value") == selected_file
                            or row.get("filename") == selected_file
                            or _normalize_ann_name(row.get("value")) == selected_norm
                            or _normalize_ann_name(row.get("filename")) == selected_norm
                        ]
                        if not chosen:
                            return "❌ 选中文件不存在于当前筛选结果", {}, {}
                    else:
                        chosen = db_payloads

                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        preview_payload = None
                        preview_file = None
                        target_filename = None

                        for idx, row in enumerate(chosen):
                            payload = row.get("payload") or {}
                            name = row.get("filename") or row.get("value") or ""
                            core = _normalize_ann_name(payload.get("filename") or name)
                            json_name = f"{core}.json"
                            with open(tmp_path / json_name, "w", encoding="utf-8") as wf:
                                json.dump(payload, wf, ensure_ascii=False, indent=2)
                            if idx == 0:
                                preview_payload = payload
                                preview_file = json_name
                                if mode == "single":
                                    target_filename = json_name

                        return _run_convert(str(tmp_path), target_filename, preview_payload, preview_file)

                return "❌ 未找到可转换标注（DB 查询为空）", {}, {}

            # 绑定事件
            # 绑定事件
            convert_curr_btn.click(
                fn=lambda f, i, m, o, fam, approved: convert_core(f, i, m, o, fam, approved, mode="single"),
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conf_output_path, conv_model_family, approved_only],
                outputs=[convert_status, before_json, after_json]
            )
            
            convert_all_btn.click(
                fn=lambda f, i, m, o, fam, approved: convert_core(f, i, m, o, fam, approved, mode="batch"),
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conf_output_path, conv_model_family, approved_only],
                outputs=[convert_status, before_json, after_json]
            )
            
            # 动态更新输出路径和Label
            def on_model_family_change(family, ann_dir, approved_only_flag):
                new_path = settings.DATA_TRAINING_CHATTS_DIR if family == "chatts" else settings.DATA_TRAINING_QWEN_DIR
                new_conf = str(Path(new_path) / "converted_data.json")
                
                # Update output label
                new_label = "#### 🎯 转换后 (ChatTS Training Data)" if family == "chatts" else "#### 🎯 转换后 (Qwen Training Data)"
                
                # Update Image/Data Dir label and value
                if family == "chatts":
                    img_dir_label = "数据文件来源 (Source Data Dir)"
                    img_dir_val = settings.DATA_DOWNSAMPLED_DIR
                else:
                    img_dir_label = "图片文件来源 (Source Image Dir)"
                    img_dir_val = settings.DATA_IMAGES_DIR
                
                # Filter file choices
                new_choices = get_file_choices(ann_dir, family, approved_only_flag)
                new_val = new_choices[0][1] if new_choices else None
                
                return (
                    new_conf, 
                    gr.update(value=new_label),
                    gr.update(value=img_dir_val, label=img_dir_label),
                    gr.update(choices=new_choices, value=new_val)
                )

            conv_model_family.change(
                fn=on_model_family_change,
                inputs=[conv_model_family, conf_input_dir, approved_only],
                outputs=[conf_output_path, after_json_label, conf_image_dir, ann_file_dropdown]
            )
            
            refresh_files_btn.click(
                fn=refresh_files,
                inputs=[conf_input_dir, conv_model_family, approved_only],
                outputs=ann_file_dropdown
            ).then(
                fn=preview_source_file,
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conv_model_family, approved_only],
                outputs=[before_json, after_json]
            )

            conf_input_dir.change(
                fn=refresh_files,
                inputs=[conf_input_dir, conv_model_family, approved_only],
                outputs=ann_file_dropdown
            ).then(
                fn=preview_source_file,
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conv_model_family, approved_only],
                outputs=[before_json, after_json]
            )
            
            approved_only.change(
                fn=refresh_files,
                inputs=[conf_input_dir, conv_model_family, approved_only],
                outputs=ann_file_dropdown
            ).then(
                fn=preview_source_file,
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conv_model_family, approved_only],
                outputs=[before_json, after_json]
            )
            
            # 选择文件立即预览
            ann_file_dropdown.change(
                fn=preview_source_file,
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conv_model_family, approved_only],
                outputs=[before_json, after_json]
            )
            
            # 模型格式切换触发预览更新
            conv_model_family.change(
                fn=preview_source_file,
                inputs=[ann_file_dropdown, conf_input_dir, conf_image_dir, conv_model_family, approved_only],
                outputs=[before_json, after_json]
            )
            
            # 初始化预览 (Default to chatts as per Radio default)
            if initial_val:
                init_src, init_ex = preview_source_file(initial_val, default_ann_dir, settings.DATA_DOWNSAMPLED_DIR, "chatts", True)
                before_json.value = init_src
                after_json.value = init_ex

        # ==================== 数据资产管理 Tab (New) ====================
        with gr.Tab("📦 数据资产管理", render=False) as data_assets_tab:
            with gr.Tabs():
                # 1. 标注数据管理
                with gr.Tab("标注数据 (Annotations)", render=False) as annotations_manage_tab:
                    gr.Markdown("### 📝 标注记录管理（DB-First）")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ann_mgr_family = gr.Dropdown(
                                label="模型类型",
                                choices=["all", "chatts", "qwen", "timer", "adtk_hbos", "ensemble"],
                                value="all",
                                interactive=True
                            )
                            ann_mgr_dir = gr.Textbox(
                                label="标注目录（仅兼容清理）",
                                value=str(Path(settings.ANNOTATIONS_ROOT) / settings.DEFAULT_USER),
                                interactive=False,
                            )
                            ann_mgr_list = gr.Dropdown(label="选择记录", interactive=True)
                            refresh_ann_mgr = gr.Button("🔄 刷新列表")
                            delete_ann_btn = gr.Button("🗑️ 删除选中记录", variant="stop")
                            ann_op_status = gr.Textbox(label="操作状态", interactive=False)
                        
                        with gr.Column(scale=2):
                            ann_mgr_view = gr.JSON(label="记录内容预览", height=600)

                    # Logic
                    def list_ann_files(path_str, model_type="all"):
                        _ = path_str
                        try:
                            from src.db.database import SessionLocal, AnnotationRecord
                            from src.utils.annotation_store import canonical_point_id
                        except Exception:
                            return []

                        mt = (model_type or "all").lower()
                        db = SessionLocal()
                        try:
                            rows = (
                                db.query(AnnotationRecord)
                                .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
                                .order_by(AnnotationRecord.updated_at.desc())
                                .all()
                            )
                        finally:
                            db.close()

                        def _matches(row) -> bool:
                            name = str(getattr(row, "filename", "") or getattr(row, "source_id", "") or "").lower()
                            method = str(getattr(row, "method", "") or "").lower()
                            if mt == "qwen":
                                return method == "qwen" or "qwen" in name
                            if mt == "timer":
                                return method == "timer" or "timer" in name
                            if mt == "adtk_hbos":
                                return method == "adtk_hbos" or "adtk_hbos" in name
                            if mt == "ensemble":
                                return method == "ensemble" or "ensemble" in name
                            if mt == "chatts":
                                if method:
                                    return method == "chatts"
                                return not any(k in name for k in ["qwen", "timer", "adtk_hbos", "ensemble"])
                            return True

                        choices = []
                        seen = set()
                        for row in rows:
                            if not _matches(row):
                                continue
                            point_id = canonical_point_id(getattr(row, "point_id", None), row.source_id, row.filename)
                            if not point_id or point_id in seen:
                                continue
                            seen.add(point_id)
                            source_kind = str(getattr(row, "source_kind", "") or "human").strip().lower()
                            method = str(getattr(row, "method", "") or "unknown").strip().lower()
                            label = f"{point_id} | [{source_kind.upper()}] | {method}"
                            choices.append((label, point_id))
                        return choices

                    def load_ann_content(path_str, point_id):
                        _ = path_str
                        if not point_id:
                            return None
                        try:
                            from sqlalchemy import or_
                            from src.db.database import SessionLocal, AnnotationRecord
                            from src.utils.annotation_store import record_to_payload
                            db = SessionLocal()
                            try:
                                row = (
                                    db.query(AnnotationRecord)
                                    .filter(
                                        AnnotationRecord.user_id == settings.DEFAULT_USER,
                                        or_(
                                            AnnotationRecord.point_id == point_id,
                                            AnnotationRecord.source_id == point_id,
                                            AnnotationRecord.filename == point_id,
                                        ),
                                    )
                                    .order_by(AnnotationRecord.updated_at.desc())
                                    .first()
                                )
                                if not row:
                                    return {"error": f"未找到标注记录: {point_id}"}
                                return record_to_payload(row, fallback_filename=row.filename or f"{point_id}.csv")
                            finally:
                                db.close()
                        except Exception as e:
                            return {"error": str(e)}

                    def delete_ann_file(path_str, point_id, model_type):
                        if not point_id:
                            return "未选择记录", gr.update()
                        try:
                            from sqlalchemy import or_
                            from src.db.database import SessionLocal, AnnotationRecord, AnnotationSegment
                            from src.utils.annotation_store import canonical_point_id

                            db = SessionLocal()
                            try:
                                row = (
                                    db.query(AnnotationRecord)
                                    .filter(
                                        AnnotationRecord.user_id == settings.DEFAULT_USER,
                                        or_(
                                            AnnotationRecord.point_id == point_id,
                                            AnnotationRecord.source_id == point_id,
                                            AnnotationRecord.filename == point_id,
                                        ),
                                    )
                                    .order_by(AnnotationRecord.updated_at.desc())
                                    .first()
                                )
                                if not row:
                                    return f"未找到记录: {point_id}", gr.update()

                                db.query(AnnotationSegment).filter(AnnotationSegment.annotation_id == row.id).delete()
                                db.delete(row)
                                db.commit()

                                # 可选兼容清理：移除同名文件，不作为在线主链依赖
                                ann_dir = Path(path_str)
                                normalized = canonical_point_id(getattr(row, "point_id", None), row.source_id, row.filename)
                                candidates = {
                                    row.filename or "",
                                    f"{normalized}.json" if normalized else "",
                                    f"annotations_{normalized}.json" if normalized else "",
                                }
                                deleted_files = 0
                                for name in candidates:
                                    if not name:
                                        continue
                                    p = ann_dir / name
                                    if p.exists():
                                        p.unlink()
                                        deleted_files += 1
                            finally:
                                db.close()

                            new_list = list_ann_files(path_str, model_type)
                            return f"已删除记录: {point_id}（兼容清理文件 {deleted_files} 个）", gr.update(choices=new_list, value=None)
                        except Exception as e:
                            return f"删除失败: {e}", gr.update()

                    # Bindings
                    ann_mgr_dir.change(
                        fn=lambda p, mt: gr.update(choices=list_ann_files(p, mt)),
                        inputs=[ann_mgr_dir, ann_mgr_family],
                        outputs=ann_mgr_list
                    )
                    ann_mgr_family.change(
                        fn=lambda p, mt: gr.update(choices=list_ann_files(p, mt)),
                        inputs=[ann_mgr_dir, ann_mgr_family],
                        outputs=ann_mgr_list
                    )
                    refresh_ann_mgr.click(
                        fn=lambda p, mt: gr.update(choices=list_ann_files(p, mt)),
                        inputs=[ann_mgr_dir, ann_mgr_family],
                        outputs=ann_mgr_list
                    )
                    ann_mgr_list.change(fn=load_ann_content, inputs=[ann_mgr_dir, ann_mgr_list], outputs=ann_mgr_view)
                    delete_ann_btn.click(
                        fn=delete_ann_file,
                        inputs=[ann_mgr_dir, ann_mgr_list, ann_mgr_family],
                        outputs=[ann_op_status, ann_mgr_list],
                    )

                # 1.5 审核队列管理
                with gr.Tab("审核队列 (Review Queue)", render=False) as review_queue_tab:
                    gr.Markdown("### ✅ 训练样本审核流转管理")
                    review_rows_state = gr.State({})

                    with gr.Row():
                        with gr.Column(scale=1):
                            review_training_family = gr.Dropdown(
                                label="训练数据模型",
                                choices=["all"] + TRAINING_MODEL_FAMILIES,
                                value="all",
                                interactive=True,
                            )
                            review_annotation_kind_filter = gr.Dropdown(
                                label="标注来源筛选",
                                choices=[
                                    ("All (全部)", "all"),
                                    ("Auto (未人工改)", "auto"),
                                    ("Human (人工改)", "human"),
                                ],
                                value="all",
                                interactive=True,
                            )
                            review_method = gr.Dropdown(
                                label="算法筛选",
                                choices=[""] + TRAINING_MODEL_FAMILIES + ["timer", "adtk_hbos", "ensemble"],
                                value="",
                                interactive=True,
                            )
                            review_status_filter = gr.Dropdown(
                                label="状态筛选",
                                choices=[
                                    ("全部", ""),
                                    ("重新审核", "pending"),
                                    ("已通过", "approved"),
                                    ("需标注修订", "needs_fix"),
                                ],
                                value="",
                                interactive=True,
                            )
                            review_keyword = gr.Textbox(label="关键字筛选", placeholder="输入点位名称...")
                            review_limit = gr.Slider(label="列表数量", minimum=20, maximum=500, step=20, value=200)
                            review_reviewer = gr.Textbox(label="审核人", value=settings.DEFAULT_USER)
                            review_refresh_btn = gr.Button("🔄 刷新队列")

                        with gr.Column(scale=2):
                            with gr.Row():
                                review_stats_box = gr.Textbox(label="队列统计", interactive=False)
                                review_action_status = gr.Textbox(label="操作状态", interactive=False)
                            review_queue_items = gr.CheckboxGroup(label="审核项（可多选）", choices=[])
                            with gr.Row():
                                review_mark_approved_btn = gr.Button("通过", variant="primary")
                                review_mark_fix_btn = gr.Button("需标注修订")
                                review_mark_pending_btn = gr.Button("重新审核")

                    with gr.Row():
                        with gr.Column(scale=2):
                            review_preview_meta = gr.Markdown("未选择审核项")
                            review_preview_plot = gr.Image(label="统一曲线预览（异常区间高亮）", height=360)
                        with gr.Column(scale=1):
                            review_preview_segments = gr.Dataframe(
                                label="异常段预览",
                                headers=["ann_id", "label", "start", "end", "count", "score"],
                                interactive=False,
                                row_count=(10, "dynamic"),
                            )
                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.Tab("训练样本"):
                                    review_preview_training = gr.JSON(label="训练样本预览", height=260)
                                with gr.Tab("标注溯源"):
                                    review_preview_json = gr.JSON(label="选中项详情", height=260)

                    def _normalize_review_method(text: str) -> Optional[str]:
                        value = (text or "").strip()
                        return value or None

                    def _normalize_review_name(name: str) -> str:
                        text = str(name or "").strip()
                        text = Path(text).name
                        lower = text.lower()
                        for ext in (".csv", ".json"):
                            if lower.endswith(ext):
                                text = text[: -len(ext)]
                                lower = text.lower()
                        if text.startswith("annotations_"):
                            text = text.replace("annotations_", "", 1)
                        return text.replace("数据集", "").strip()

                    def _empty_review_preview():
                        empty_df = pd.DataFrame(columns=["ann_id", "label", "start", "end", "count", "score"])
                        return {}, None, "未选择审核项", empty_df, {}

                    def _extract_point_id_from_text(text: str) -> Optional[str]:
                        import re

                        raw = str(text or "").strip()
                        if not raw:
                            return None
                        base = Path(raw).name
                        matches = re.findall(r"([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9]+)", base)
                        candidates = []
                        for m in matches:
                            lower = m.lower()
                            if lower.endswith((".csv", ".json", ".jpg", ".jpeg", ".png", ".webp")):
                                continue
                            candidates.append(m)
                        if candidates:
                            candidates.sort(key=lambda s: (s.count("_"), len(s)), reverse=True)
                            return _normalize_review_name(candidates[0])
                        return _normalize_review_name(base)

                    def _extract_intervals_from_text(output_text: str) -> List[Dict]:
                        import re

                        text = str(output_text or "")
                        rows = []
                        for idx, m in enumerate(re.finditer(r'"interval"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text), 1):
                            start = int(m.group(1))
                            end = int(m.group(2))
                            if end < start:
                                start, end = end, start
                            rows.append(
                                {
                                    "ann_id": f"train_{idx}",
                                    "label": "training_output",
                                    "start": start,
                                    "end": end,
                                    "count": max(end - start + 1, 1),
                                    "score": None,
                                }
                            )
                        return rows

                    def _iter_training_records_for_review(file_path: Path):
                        suffix = file_path.suffix.lower()
                        if suffix == ".jsonl":
                            with file_path.open("r", encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        item = json.loads(line)
                                    except Exception:
                                        continue
                                    if isinstance(item, dict):
                                        yield item
                            return

                        if suffix != ".json":
                            return

                        try:
                            data = json.loads(file_path.read_text(encoding="utf-8"))
                        except Exception:
                            return

                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    yield item
                            return

                        if isinstance(data, dict):
                            if isinstance(data.get("data"), list):
                                for item in data["data"]:
                                    if isinstance(item, dict):
                                        yield item
                                return
                            yield data

                    def _build_training_preview_item(record: Dict, model_family: str, source_file: str, index: int) -> Optional[Dict]:
                        point_name = None
                        image_path = ""
                        user_text = ""
                        assistant_text = ""

                        if model_family == "qwen":
                            image_path = str(record.get("image") or "").strip()
                            point_name = _extract_point_id_from_text(image_path)
                            conversations = record.get("conversations") or []
                            if isinstance(conversations, list):
                                for c in conversations:
                                    if not isinstance(c, dict):
                                        continue
                                    if c.get("from") == "user":
                                        user_text = str(c.get("value") or "")
                                    elif c.get("from") == "assistant":
                                        assistant_text = str(c.get("value") or "")
                        else:
                            point_name = _extract_point_id_from_text(record.get("id") or record.get("source_id") or record.get("point_name") or "")
                            user_text = str(record.get("input") or "")
                            assistant_text = str(record.get("output") or "")
                            image_path = str(record.get("image") or "").strip()

                        if not point_name:
                            return None

                        intervals = _extract_intervals_from_text(assistant_text)
                        return {
                            "id": point_name,
                            "source_type": "training",
                            "source_id": point_name,
                            "point_name": point_name,
                            "model_family": model_family,
                            "source_file": source_file,
                            "record_index": index,
                            "training_input": user_text,
                            "training_output": assistant_text,
                            "training_record": record,
                            "training_intervals": intervals,
                            "training_image_path": image_path,
                        }

                    def _find_training_preview_image(item: Dict) -> Optional[Path]:
                        image_path = str(item.get("training_image_path") or "").strip()
                        if image_path:
                            p = Path(image_path)
                            if p.exists():
                                return p

                        point_name = _normalize_review_name(item.get("point_name") or item.get("source_id") or "")
                        if not point_name:
                            return None

                        candidates = [f"{point_name}.jpg", f"{point_name}.png", f"{point_name}.jpeg"]
                        image_dir = Path(settings.DATA_IMAGES_DIR)
                        if image_dir.exists():
                            for name in candidates:
                                p = image_dir / name
                                if p.exists():
                                    return p
                            for child in image_dir.glob("*.jpg"):
                                if point_name in child.name:
                                    return child
                        return None

                    def _render_review_image_overlay(image_path: Path, segments: List[Dict], title: str):
                        if not image_path.exists():
                            return None, "未找到训练图片"
                        try:
                            import matplotlib.pyplot as plt

                            img = plt.imread(str(image_path))
                            fig, ax = plt.subplots(figsize=(10, 4.2))
                            ax.imshow(img)
                            ax.axis("off")
                            ax.set_title(title)

                            lines = []
                            for seg in segments[:12]:
                                lines.append(f"[{int(seg.get('start') or 0)}, {int(seg.get('end') or 0)}]")
                            if not lines:
                                lines = ["无区间(请检查训练输出)"]
                            text = "最终标签索引:\n" + "\n".join(lines)
                            ax.text(
                                0.01,
                                0.02,
                                text,
                                transform=ax.transAxes,
                                fontsize=10,
                                color="white",
                                bbox={"facecolor": "black", "alpha": 0.6, "pad": 6},
                                va="bottom",
                            )

                            temp_dir = Path("temp_images")
                            temp_dir.mkdir(exist_ok=True)
                            out_path = temp_dir / f"review_img_{uuid.uuid4().hex[:8]}.jpg"
                            fig.tight_layout()
                            fig.savefig(str(out_path), dpi=120)
                            plt.close(fig)
                            return str(out_path), f"已叠加 {len(lines)} 条索引区间"
                        except Exception as e:
                            return None, f"图片渲染失败: {e}"

                    def _resolve_review_csv_path(item: Dict) -> Optional[Path]:
                        result_path = str(item.get("result_path") or "").strip()
                        if result_path:
                            p = Path(result_path)
                            if p.exists() and p.suffix.lower() == ".csv":
                                return p

                        point_name = _normalize_review_name(item.get("point_name") or item.get("source_id") or "")
                        filename = str(item.get("filename") or "").strip()
                        candidates = []
                        if filename:
                            candidates.append(filename)
                        if point_name:
                            candidates.append(f"{point_name}.csv")

                        seen = set()
                        unique_candidates = []
                        for name in candidates:
                            key = name.lower()
                            if key in seen:
                                continue
                            seen.add(key)
                            unique_candidates.append(name)

                        downsampled_dir = Path(settings.DATA_DOWNSAMPLED_DIR)
                        inference_dir = Path(settings.DATA_INFERENCE_DIR)
                        for name in unique_candidates:
                            p = downsampled_dir / name
                            if p.exists():
                                return p
                            p = inference_dir / name
                            if p.exists():
                                return p
                            if inference_dir.exists():
                                for child in inference_dir.iterdir():
                                    if not child.is_dir():
                                        continue
                                    cp = child / name
                                    if cp.exists():
                                        return cp
                        return None

                    def _load_annotation_segments(source_id: str):
                        from src.db.database import SessionLocal, AnnotationRecord, init_db
                        from src.utils.annotation_store import record_to_payload

                        normalized = _normalize_review_name(source_id)
                        if not normalized:
                            return None, [], "human", 0, 0

                        init_db()
                        db = SessionLocal()
                        try:
                            row = (
                                db.query(AnnotationRecord)
                                .filter(
                                    AnnotationRecord.user_id == settings.DEFAULT_USER,
                                    AnnotationRecord.point_id == normalized,
                                )
                                .order_by(AnnotationRecord.updated_at.desc())
                                .first()
                            )
                            if row is None:
                                row = (
                                    db.query(AnnotationRecord)
                                    .filter(
                                        AnnotationRecord.user_id == settings.DEFAULT_USER,
                                        AnnotationRecord.source_id == normalized,
                                    )
                                    .order_by(AnnotationRecord.updated_at.desc())
                                    .first()
                                )
                        finally:
                            db.close()

                        if row is None:
                            return None, [], "human", 0, 0

                        payload = record_to_payload(row)
                        annotations = payload.get("annotations") or []
                        segments = []
                        for ann in annotations:
                            if not isinstance(ann, dict):
                                continue
                            ann_id = ann.get("id")
                            label_obj = ann.get("label")
                            if isinstance(label_obj, dict):
                                label = label_obj.get("text") or label_obj.get("id") or ""
                            else:
                                label = str(label_obj or "")
                            for seg in (ann.get("segments") or []):
                                if not isinstance(seg, dict):
                                    continue
                                segments.append(
                                    {
                                        "ann_id": ann_id or "",
                                        "label": label,
                                        "start": int(seg.get("start") or 0),
                                        "end": int(seg.get("end") or 0),
                                        "count": int(seg.get("count") or 0),
                                        "score": seg.get("score"),
                                    }
                                )

                        return payload, segments, (row.source_kind or "human"), int(row.annotation_count or 0), int(row.segment_count or 0)

                    def _render_review_plot(csv_path: Optional[Path], segments: List[Dict], title: str):
                        if csv_path is None or not csv_path.exists():
                            return None, "未找到对应 CSV 文件"

                        try:
                            df = pd.read_csv(csv_path, nrows=5000)
                        except Exception as e:
                            return None, f"读取 CSV 失败: {e}"

                        if df.empty:
                            return None, "CSV 为空"

                        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        if not numeric_cols:
                            return None, "CSV 无数值列，无法绘制曲线"

                        y_col = "value" if "value" in df.columns else numeric_cols[0]
                        y = pd.to_numeric(df[y_col], errors="coerce").ffill().bfill().fillna(0.0)

                        try:
                            highlight_regions = []
                            max_idx = len(y)
                            for seg in segments[:200]:
                                start = max(int(seg.get("start") or 0), 0)
                                end = int(seg.get("end") or start)
                                if start >= max_idx:
                                    continue
                                # create_ts_image uses [start, end) semantics; force at least 1 point width.
                                end = max(start + 1, end)
                                end = min(end, max_idx)
                                if start < end:
                                    highlight_regions.append((start, end))

                            img = create_ts_image(
                                data=y,
                                y_range=None,
                                start_index=0,
                                highlight_regions=highlight_regions,
                                show_grid=True,
                            )

                            temp_dir = Path("temp_images")
                            temp_dir.mkdir(exist_ok=True)
                            out_path = temp_dir / f"review_{uuid.uuid4().hex[:8]}.jpg"
                            img.save(str(out_path), format="JPEG", quality=95)
                            return str(out_path), f"统一曲线与异常区间绘制成功（列: {y_col}，点数: {len(df)}，异常段: {len(highlight_regions)}）"
                        except Exception as e:
                            return None, f"绘图失败: {e}"

                    def _build_review_preview(item: Optional[Dict]):
                        if not item:
                            return _empty_review_preview()

                        preview_item = dict(item)
                        source_type = "training"
                        source_kind = preview_item.get("source_kind") or "human"
                        annotation_count = int(preview_item.get("annotation_count") or 0)
                        segment_count = int(preview_item.get("segment_count") or 0)
                        train_segments = list(preview_item.get("training_intervals") or [])
                        payload, ann_segments, detected_kind, ann_count, seg_count = _load_annotation_segments(
                            str(preview_item.get("source_id") or preview_item.get("point_name") or "")
                        )
                        if payload:
                            preview_item["annotation_payload"] = payload
                        source_kind = detected_kind or source_kind
                        annotation_count = ann_count
                        segment_count = seg_count
                        segments = train_segments or ann_segments

                        preview_item["source_kind"] = source_kind
                        preview_item["annotation_count"] = annotation_count
                        preview_item["segment_count"] = segment_count

                        title_name = preview_item.get("point_name") or preview_item.get("source_id") or "-"
                        csv_path = _resolve_review_csv_path(preview_item)
                        plot_path, plot_msg = _render_review_plot(csv_path, segments, f"训练样本审核: {title_name}")
                        if not plot_path:
                            image_path = _find_training_preview_image(preview_item)
                            if image_path:
                                plot_path, plot_msg = _render_review_image_overlay(image_path, segments, f"训练样本审核(图片回退): {title_name}")

                        seg_df = pd.DataFrame(segments, columns=["ann_id", "label", "start", "end", "count", "score"])
                        training_preview = {
                            "model_family": preview_item.get("model_family"),
                            "source_file": preview_item.get("source_file"),
                            "record_index": preview_item.get("record_index"),
                            "point_name": preview_item.get("point_name"),
                            "input": str(preview_item.get("training_input") or "")[:1200],
                            "output": str(preview_item.get("training_output") or "")[:1200],
                            "interval_count": len(train_segments),
                        }

                        meta = (
                            f"**点位**: {title_name}  \n"
                            f"**来源类型**: {source_type}  \n"
                            f"**训练模型**: {preview_item.get('model_family') or '-'}  \n"
                            f"**训练文件**: {preview_item.get('source_file') or '-'}#{preview_item.get('record_index')}  \n"
                            f"**标注来源**: {source_kind}  \n"
                            f"**标注数/段数**: {annotation_count}/{segment_count}  \n"
                            f"**关联分数**: {(preview_item.get('score') if preview_item.get('score') is not None else '-')} ({preview_item.get('method') or '-'})  \n"
                            f"**可视化状态**: {plot_msg}"
                        )

                        return preview_item, plot_path, meta, seg_df, training_preview

                    def _load_review_rows(training_family: str, status: str, method: str, keyword: str, limit: int, annotation_kind: str):
                        status_label_map = {
                            "pending": "重新审核",
                            "approved": "已通过",
                            "needs_fix": "需标注修订",
                        }
                        try:
                            from src.db.database import SessionLocal, ReviewQueue, InferenceResult, AnnotationRecord, init_db
                            init_db()
                            db = SessionLocal()
                            try:
                                norm_method = _normalize_review_method(method)
                                ann_rows = (
                                    db.query(
                                        AnnotationRecord.point_id,
                                        AnnotationRecord.source_id,
                                        AnnotationRecord.source_kind,
                                        AnnotationRecord.annotation_count,
                                        AnnotationRecord.segment_count,
                                        )
                                    .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
                                    .all()
                                )
                                ann_meta_map: Dict[str, Dict] = {}
                                for point_id, source_id, source_kind, ann_count, seg_count in ann_rows:
                                    key = _normalize_review_name(point_id or source_id)
                                    if not key:
                                        continue
                                    ann_meta_map[key] = {
                                        "source_kind": (source_kind or "human").strip().lower(),
                                        "annotation_count": int(ann_count or 0),
                                        "segment_count": int(seg_count or 0),
                                    }

                                review_rows = (
                                    db.query(
                                        ReviewQueue.point_id,
                                        ReviewQueue.source_id,
                                        ReviewQueue.status,
                                        ReviewQueue.reviewer,
                                        ReviewQueue.updated_at,
                                    )
                                    .filter(ReviewQueue.source_type == "annotation")
                                    .order_by(ReviewQueue.updated_at.desc())
                                    .all()
                                )
                                review_status_map: Dict[str, Dict] = {}
                                for point_id, source_id, st, reviewer, updated_at in review_rows:
                                    key = _normalize_review_name(point_id or source_id)
                                    if not key or key in review_status_map:
                                        continue
                                    mapped = "needs_fix" if (st or "").strip().lower() == "rejected" else ((st or "").strip().lower() or "pending")
                                    review_status_map[key] = {
                                        "status": mapped,
                                        "reviewer": reviewer,
                                        "updated_at": updated_at.isoformat() if updated_at else None,
                                    }

                                inf_rows = db.query(InferenceResult).order_by(InferenceResult.created_at.desc()).all()
                                score_map: Dict[str, Dict] = {}
                                for row in inf_rows:
                                    key = _normalize_review_name(getattr(row, "point_id", None) or row.point_name or "")
                                    if not key or key in score_map:
                                        continue
                                    if norm_method and (row.method or "").strip() != norm_method:
                                        continue
                                    score_map[key] = {
                                        "method": row.method,
                                        "score": float(row.score_avg) if row.score_avg is not None else None,
                                    }
                            finally:
                                db.close()
                        except Exception as e:
                            empty = _empty_review_preview()
                            return gr.update(choices=[], value=[]), f"加载失败: {e}", {}, *empty

                        roots = []
                        fam = (training_family or "all").strip().lower()
                        if fam in {"all", "chatts"}:
                            roots.append(("chatts", Path(settings.DATA_TRAINING_CHATTS_DIR)))
                        if fam in {"all", "qwen"}:
                            roots.append(("qwen", Path(settings.DATA_TRAINING_QWEN_DIR)))

                        entries: List[Dict] = []
                        seen_points = set()
                        for family, root in roots:
                            if not root.exists():
                                continue
                            files = sorted(
                                [
                                    p for p in root.iterdir()
                                    if p.is_file()
                                    and p.suffix.lower() in {".json", ".jsonl"}
                                    and not p.name.startswith(".")
                                    and not p.name.startswith("_")
                                    and not p.name.startswith("dataset_info")
                                ],
                                key=lambda p: p.stat().st_mtime,
                                reverse=True,
                            )
                            for fp in files:
                                for idx, rec in enumerate(_iter_training_records_for_review(fp), 1):
                                    item = _build_training_preview_item(rec, family, fp.name, idx)
                                    if not item:
                                        continue
                                    point_name = _normalize_review_name(item.get("point_name") or "")
                                    if not point_name or point_name in seen_points:
                                        continue
                                    if norm_method and point_name not in score_map:
                                        continue
                                    seen_points.add(point_name)
                                    entries.append(item)

                        if limit and int(limit) > 0:
                            entries = entries[: int(limit)]

                        items = []
                        state_map: Dict[str, Dict] = {}
                        normalized_ann_kind = (annotation_kind or "all").strip().lower()
                        keyword_text = (keyword or "").strip().lower()
                        stats = {"total": 0, "pending": 0, "approved": 0, "needs_fix": 0}
                        status_filter = (status or "").strip().lower()
                        for item in entries:
                            key = _normalize_review_name(item.get("point_name") or item.get("source_id") or "")
                            meta = ann_meta_map.get(key, {})
                            source_kind = meta.get("source_kind", "human")
                            if normalized_ann_kind in {"auto", "human"} and source_kind != normalized_ann_kind:
                                continue
                            if keyword_text and keyword_text not in key.lower():
                                continue
                            review_meta = review_status_map.get(key, {})
                            item_status = review_meta.get("status", "pending")
                            if status_filter and item_status != status_filter:
                                continue
                            score_meta = score_map.get(key, {})
                            item.update(
                                {
                                    "id": key,
                                    "status": item_status,
                                    "source_kind": source_kind,
                                    "annotation_count": int(meta.get("annotation_count", 0)),
                                    "segment_count": int(meta.get("segment_count", 0)),
                                    "method": score_meta.get("method"),
                                    "score": score_meta.get("score"),
                                    "reviewer": review_meta.get("reviewer"),
                                    "updated_at": review_meta.get("updated_at"),
                                }
                            )
                            stats["total"] += 1
                            stats[item_status] = stats.get(item_status, 0) + 1
                            kind_tag = "[AUTO]" if source_kind == "auto" else "[HUMAN]"
                            score_text = f"{float(item['score']):.3f}" if item.get("score") is not None else "-"
                            status_text = status_label_map.get(item_status, item_status)
                            label = (
                                f"[{status_text}] [{str(item.get('model_family') or '').upper()}] {kind_tag} "
                                f"{item.get('point_name') or item.get('source_id')} | {item.get('method') or '-'} | {score_text}"
                            )
                            items.append((label, key))
                            state_map[key] = item

                        stats_text = (
                            f"总 {stats.get('total', 0)} | 重新审核 {stats.get('pending', 0)} | "
                            f"已通过 {stats.get('approved', 0)} | 需标注修订 {stats.get('needs_fix', 0)}"
                        )
                        first_item = next(iter(state_map.values()), None)
                        preview = _build_review_preview(first_item)
                        return gr.update(choices=items, value=[]), stats_text, state_map, *preview

                    def _batch_update_review_status(selected_ids, status, reviewer, state_map):
                        ids = list(selected_ids or [])
                        if not ids:
                            return "❌ 请先选择审核项"
                        if status not in {"pending", "approved", "needs_fix"}:
                            return "❌ 非法状态"
                        status_label_map = {
                            "pending": "重新审核",
                            "approved": "已通过",
                            "needs_fix": "需标注修订",
                        }
                        try:
                            from src.db.database import SessionLocal, ReviewQueue, init_db
                            init_db()
                            db = SessionLocal()
                            try:
                                updated = 0
                                mapping = state_map or {}
                                reviewer_text = (reviewer or "").strip() or None
                                for rid in ids:
                                    item = mapping.get(rid, {})
                                    point_name = _normalize_review_name(item.get("point_name") or rid)
                                    if not point_name:
                                        continue
                                    row = (
                                        db.query(ReviewQueue)
                                        .filter(ReviewQueue.source_type == "annotation", ReviewQueue.point_id == point_name)
                                        .order_by(ReviewQueue.updated_at.desc())
                                        .first()
                                    )
                                    if row is None:
                                        row = (
                                            db.query(ReviewQueue)
                                            .filter(
                                                ReviewQueue.source_type == "annotation",
                                                ReviewQueue.source_id == point_name,
                                            )
                                            .order_by(ReviewQueue.updated_at.desc())
                                            .first()
                                        )
                                    if row is None:
                                        row = ReviewQueue(
                                            id=str(uuid.uuid4()),
                                            source_type="annotation",
                                            source_id=point_name,
                                            point_id=_normalize_review_name(point_name),
                                            method=item.get("method"),
                                            model=None,
                                            point_name=point_name,
                                            score=item.get("score"),
                                            strategy="training_review",
                                            status=status,
                                            reviewer=reviewer_text,
                                            updated_at=utc_now_naive(),
                                        )
                                        db.add(row)
                                    else:
                                        row.status = status
                                        row.reviewer = reviewer_text
                                        row.updated_at = utc_now_naive()
                                    updated += 1
                                db.commit()
                            except Exception:
                                db.rollback()
                                raise
                            finally:
                                db.close()
                            return f"✅ 已更新 {updated} 条为 {status_label_map.get(status, status)}"
                        except Exception as e:
                            return f"❌ 更新失败: {e}"

                    def _preview_review_items(selected_ids, state_map):
                        if not selected_ids:
                            return _empty_review_preview()
                        mapping = state_map or {}
                        first = None
                        for rid in selected_ids:
                            if rid in mapping:
                                first = mapping[rid]
                                break
                        return _build_review_preview(first)

                    review_refresh_btn.click(
                        fn=_load_review_rows,
                        inputs=[review_training_family, review_status_filter, review_method, review_keyword, review_limit, review_annotation_kind_filter],
                        outputs=[
                            review_queue_items,
                            review_stats_box,
                            review_rows_state,
                            review_preview_json,
                            review_preview_plot,
                            review_preview_meta,
                            review_preview_segments,
                            review_preview_training,
                        ],
                    )

                    review_mark_approved_btn.click(
                        fn=lambda ids, reviewer, state: _batch_update_review_status(ids, "approved", reviewer, state),
                        inputs=[review_queue_items, review_reviewer, review_rows_state],
                        outputs=review_action_status,
                    ).then(
                        fn=_load_review_rows,
                        inputs=[review_training_family, review_status_filter, review_method, review_keyword, review_limit, review_annotation_kind_filter],
                        outputs=[
                            review_queue_items,
                            review_stats_box,
                            review_rows_state,
                            review_preview_json,
                            review_preview_plot,
                            review_preview_meta,
                            review_preview_segments,
                            review_preview_training,
                        ],
                    )

                    review_mark_fix_btn.click(
                        fn=lambda ids, reviewer, state: _batch_update_review_status(ids, "needs_fix", reviewer, state),
                        inputs=[review_queue_items, review_reviewer, review_rows_state],
                        outputs=review_action_status,
                    ).then(
                        fn=_load_review_rows,
                        inputs=[review_training_family, review_status_filter, review_method, review_keyword, review_limit, review_annotation_kind_filter],
                        outputs=[
                            review_queue_items,
                            review_stats_box,
                            review_rows_state,
                            review_preview_json,
                            review_preview_plot,
                            review_preview_meta,
                            review_preview_segments,
                            review_preview_training,
                        ],
                    )

                    review_mark_pending_btn.click(
                        fn=lambda ids, reviewer, state: _batch_update_review_status(ids, "pending", reviewer, state),
                        inputs=[review_queue_items, review_reviewer, review_rows_state],
                        outputs=review_action_status,
                    ).then(
                        fn=_load_review_rows,
                        inputs=[review_training_family, review_status_filter, review_method, review_keyword, review_limit, review_annotation_kind_filter],
                        outputs=[
                            review_queue_items,
                            review_stats_box,
                            review_rows_state,
                            review_preview_json,
                            review_preview_plot,
                            review_preview_meta,
                            review_preview_segments,
                            review_preview_training,
                        ],
                    )

                    review_queue_items.change(
                        fn=_preview_review_items,
                        inputs=[review_queue_items, review_rows_state],
                        outputs=[review_preview_json, review_preview_plot, review_preview_meta, review_preview_segments, review_preview_training],
                    )

                # 2. 训练数据管理
                with gr.Tab("训练数据 (Training Data)", render=False) as training_data_tab:
                    gr.Markdown("### 🎯 微调数据管理 (Converted JSONL)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            train_mgr_family = gr.Dropdown(
                                label="模型类型",
                                choices=TRAINING_MODEL_FAMILIES,
                                value="chatts",
                                interactive=True
                            )
                            train_mgr_dir = gr.Textbox(
                                label="数据目录",
                                value=settings.DATA_TRAINING_CHATTS_DIR,
                                interactive=False
                            )
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
                        # Filter useless files
                        files = [
                            f for f in files 
                            if not f.name.startswith("dataset_info") 
                            and not f.name.startswith(".") 
                            and not f.name.startswith("_")
                        ]
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

                    def resolve_train_mgr_dir(model_family: str) -> str:
                        if model_family == "qwen":
                            return settings.DATA_TRAINING_QWEN_DIR
                        return settings.DATA_TRAINING_CHATTS_DIR

                    train_mgr_family.change(
                        fn=lambda mf: gr.update(value=resolve_train_mgr_dir(mf)),
                        inputs=train_mgr_family,
                        outputs=train_mgr_dir
                    ).then(
                        fn=lambda p: gr.update(choices=list_train_files(p), value=None),
                        inputs=train_mgr_dir,
                        outputs=train_mgr_list
                    )

                # 2.5 降采样数据预览
                with gr.Tab("降采样数据 (Downsampled)", render=False) as downsampled_data_tab:
                    gr.Markdown("### 📉 降采样数据预览 (CSV + 曲线)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ds_preview_dropdown = gr.Dropdown(
                                label="选择点位ID",
                                choices=get_dataset_names(),
                                interactive=True
                            )
                            ds_refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                            ds_column_selector = gr.CheckboxGroup(
                                label="Select columns to plot",
                                choices=[],
                                interactive=True
                            )
                        with gr.Column(scale=2):
                            ds_preview_plot = gr.Image(label="Curve Preview", height=350)

                    with gr.Accordion("📋 Data Table (first 5000 rows)", open=False):
                        ds_preview_table = gr.Dataframe(label="", interactive=False)

                    ds_refresh_btn.click(
                        fn=lambda: gr.Dropdown(choices=get_dataset_names(), value=None),
                        outputs=ds_preview_dropdown
                    )
                    ds_preview_dropdown.change(
                        fn=preview_dataset,
                        inputs=ds_preview_dropdown,
                        outputs=[ds_preview_table, ds_column_selector, ds_preview_plot]
                    )
                    ds_column_selector.change(
                        fn=update_plot_from_selection,
                        inputs=[ds_preview_dropdown, ds_column_selector],
                        outputs=ds_preview_plot
                    )

                # 3. 数据集管理 (Redesigned)
                with gr.Tab("数据集构建 (Construction)", render=False) as construction_tab:
                    gr.Markdown("### 🛠️ 数据集构建 (Source -> Target)")
                    
                    # State to hold the current list of points in the target dataset
                    target_items_state = gr.State([])

                    with gr.Row():
                        # --- Left: Source Panel ---
                        with gr.Column(scale=5):
                            gr.Markdown("#### 1. 数据源 (Source)")
                            with gr.Row():
                                src_filter_source = gr.Dropdown(
                                    label="来源类型", 
                                    choices=["Annotations (已标注)", "Training (已转换)"], 
                                    value="Annotations (已标注)"
                                )
                                src_filter_annotation_kind = gr.Dropdown(
                                    label="标注来源",
                                    choices=[
                                        ("All (全部)", "all"),
                                        ("Auto (未人工改)", "auto"),
                                        ("Human (人工改)", "human"),
                                    ],
                                    value="all",
                                )
                                src_filter_training_family = gr.Dropdown(
                                    label="训练数据模型",
                                    choices=[
                                        ("All", "all"),
                                        ("ChatTS", "chatts"),
                                        ("Qwen", "qwen"),
                                    ],
                                    value="all",
                                    visible=False,
                                )
                                src_filter_sort = gr.Dropdown(
                                    label="排序",
                                    choices=[
                                        ("Score ↓", "score_desc"),
                                        ("Score ↑", "score_asc"),
                                        ("最新优先", "updated_desc"),
                                        ("最早优先", "updated_asc"),
                                        ("名称 A-Z", "name_asc"),
                                        ("名称 Z-A", "name_desc"),
                                    ],
                                    value="score_desc",
                                )
                            with gr.Accordion("高级筛选", open=False):
                                with gr.Row():
                                    src_filter_method = gr.Dropdown(
                                        label="关联算法 (用于筛选分数)",
                                        choices=[""] + TRAINING_MODEL_FAMILIES + ["timer", "adtk_hbos", "ensemble"],
                                        value=""
                                    )
                                    src_filter_approved_only = gr.Checkbox(
                                        label="仅审核通过",
                                        value=True,
                                    )
                                    src_filter_keyword = gr.Textbox(label="关键字筛选", placeholder="输入点位名称...")

                            src_refresh_btn = gr.Button("🔄 刷新源列表 (Search)")
                            
                            # Source List
                            src_list = gr.CheckboxGroup(label="待选点位 (Source)", choices=[], container=True, elem_classes="transfer-list")
                            src_list_msg = gr.Markdown("Found: 0", elem_id="src_count")

                        # --- Middle: Operations ---
                        with gr.Column(scale=1, min_width=60, elem_classes="transfer-buttons"):
                            gr.Markdown("<br><br><br><br>") # Spacer
                            btn_add = gr.Button("添加 ➡️", variant="primary")
                            gr.Markdown("<br>")
                            btn_remove = gr.Button("⬅️ 移除")
                            
                        # --- Right: Target Panel ---
                        with gr.Column(scale=5):
                            gr.Markdown("#### 2. 目标数据集 (Target Dataset)")
                            
                            with gr.Group():
                                target_ds_dropdown = gr.Dropdown(label="加载已有数据集 (编辑模式)", choices=[])
                                target_refresh_btn = gr.Button("🔄 刷新数据集列表", size="sm")
                            
                            target_ds_name = gr.Textbox(label="数据集名称 (Name)", placeholder="e.g. train_v2")
                            with gr.Row():
                                target_ds_type = gr.Dropdown(label="类型 (Type)", choices=["train", "golden"], value="train")
                            
                            target_ds_note = gr.Textbox(label="备注 (Note)", lines=2)
                            
                            # Target List (Staging)
                            target_list = gr.CheckboxGroup(label="已选点位 (Staging)", choices=[], container=True, elem_classes="transfer-list")
                            
                            with gr.Row():
                                target_allow_overwrite = gr.Checkbox(label="允许覆盖", value=False)
                                target_freeze = gr.Checkbox(label="保存并冻结", value=False)
                            
                            target_save_btn = gr.Button("✅ 保存数据集", variant="primary")
                            target_msg = gr.Textbox(label="状态", interactive=False, show_label=False)
                            target_delete_btn = gr.Button("🗑️ 删除当前数据集", variant="stop", size="sm")
                            target_del_confirm = gr.Checkbox(label="确认删除", value=False)

                    # --- Helper Functions for Construction ---

                    def _get_dataset_items_count(ds_id: str):
                        if not ds_id:
                            return 0
                        resp = _assets_api_call("GET", f"/datasets/{ds_id}")
                        if not resp.get("success"):
                            return 0
                        asset = (resp.get("data") or {}).get("asset") or {}
                        return int(asset.get("point_count") or 0)

                    def _normalize_source_type(source_type: str) -> str:
                        text = (source_type or "").lower()
                        if "training" in text:
                            return "training"
                        if "inference" in text:
                            return "inference"
                        return "annotations"

                    def _toggle_source_kind_filter(source_type):
                        source = _normalize_source_type(source_type)
                        ann_visible = source == "annotations"
                        train_visible = source == "training"
                        method_visible = source == "annotations"
                        return (
                            gr.Dropdown(visible=ann_visible, value="all"),
                            gr.Dropdown(visible=train_visible, value="all"),
                            gr.Dropdown(visible=method_visible, value=""),
                        )

                    def refresh_source_list_logic(source_type, source_kind, training_family, sort_by, method, approved_only, keyword):
                        """Fetch source items via assets API."""
                        normalized_source = _normalize_source_type(source_type)
                        params = {
                            "source_type": normalized_source,
                            "sort_by": sort_by or "score_desc",
                            "method": method or None,
                            "approved_only": bool(approved_only),
                            "keyword": keyword or None,
                            "limit": 200,
                        }
                        if normalized_source == "annotations" and source_kind in {"auto", "human"}:
                            params["source_kind"] = source_kind
                        if normalized_source == "training" and training_family in {"chatts", "qwen"}:
                            params["model_family"] = training_family
                        resp = _assets_api_call("GET", "/sources", params=params)
                        if not resp.get("success"):
                            return gr.CheckboxGroup(choices=[], value=[]), f"Error: {resp.get('error')}"

                        choices_data = ((resp.get("data") or {}).get("choices") or [])
                        choices = [(str(c.get("label", "")), str(c.get("value", ""))) for c in choices_data]
                        return gr.CheckboxGroup(choices=choices, value=[]), f"Found: {len(choices)}"

                    def transfer_add_logic(src_selected, current_target_items):
                        if not src_selected:
                            return current_target_items, gr.CheckboxGroup(choices=current_target_items), [], "No items selected"
                        
                        # Merge and Deduplicate
                        new_set = set(current_target_items)
                        added_count = 0
                        for item in src_selected:
                            if item not in new_set:
                                new_set.add(item)
                                added_count += 1
                        
                        new_list = sorted(list(new_set))
                        return new_list, gr.CheckboxGroup(choices=new_list, value=[]), [], f"Added {added_count} items"

                    def transfer_remove_logic(target_selected, current_target_items):
                        if not target_selected:
                            return current_target_items, gr.CheckboxGroup(choices=current_target_items), [], "No items selected to remove"
                        
                        new_list = sorted([x for x in current_target_items if x not in target_selected])
                        removed_count = len(current_target_items) - len(new_list)
                        
                        return new_list, gr.CheckboxGroup(choices=new_list, value=[]), [], f"Removed {removed_count} items"

                    def _list_dataset_assets_simple():
                        resp = _assets_api_call("GET", "/datasets")
                        if not resp.get("success"):
                            return []
                        rows = (resp.get("data") or {}).get("assets") or []
                        return [
                            (f"{r.get('name')} ({r.get('dataset_type')}, {r.get('point_count', 0)} pts)", r.get("id"))
                            for r in rows if r.get("id")
                        ]

                    def refresh_target_ds_list_logic():
                        choices = _list_dataset_assets_simple()
                        return gr.Dropdown(choices=choices)

                    def load_target_ds_logic(ds_id):
                        if not ds_id:
                            return [], [], "", "train", "", "No dataset selected" # Reset

                        resp = _assets_api_call("GET", f"/datasets/{ds_id}")
                        if not resp.get("success"):
                            return [], [], "", "train", "", f"Error: {resp.get('error')}"

                        data = resp.get("data") or {}
                        asset = data.get("asset") or {}
                        point_list = sorted(data.get("items") or [])
                        return (
                            point_list,
                            gr.CheckboxGroup(choices=point_list, value=[]),
                            asset.get("name", ""),
                            asset.get("dataset_type", "train"),
                            asset.get("note", ""),
                            f"Loaded {len(point_list)} items",
                        )

                    def save_target_ds_logic(target_items, name, ds_type, note, overwrite, freeze):
                        if not name: return "❌ 必须输入数据集名称"
                        if not target_items: return "❌ 列表为空，无法保存"

                        resp = _assets_api_call(
                            "POST",
                            "/datasets/save",
                            payload={
                                "name": name,
                                "dataset_type": ds_type,
                                "items": list(target_items),
                                "note": note or "",
                                "overwrite": bool(overwrite),
                                "freeze": bool(freeze),
                            },
                        )
                        if not resp.get("success"):
                            return f"❌ {resp.get('error')}"
                        return f"✅ {resp.get('message')}"

                    def delete_target_ds_logic(ds_id, confirm):
                        if not ds_id: return "❌ 未选择数据集"
                        if not confirm: return "❌ 请先勾选确认删除"
                        resp = _assets_api_call("DELETE", f"/datasets/{ds_id}")
                        if not resp.get("success"):
                            return f"❌ {resp.get('error')}"
                        return "✅ 已删除"

                    # --- Bindings (Construction) ---
                    src_refresh_btn.click(
                        fn=refresh_source_list_logic,
                        inputs=[src_filter_source, src_filter_annotation_kind, src_filter_training_family, src_filter_sort, src_filter_method, src_filter_approved_only, src_filter_keyword],
                        outputs=[src_list, src_list_msg]
                    )

                    src_filter_source.change(
                        fn=_toggle_source_kind_filter,
                        inputs=src_filter_source,
                        outputs=[src_filter_annotation_kind, src_filter_training_family, src_filter_method],
                    )

                    btn_add.click(
                        fn=transfer_add_logic,
                        inputs=[src_list, target_items_state],
                        outputs=[target_items_state, target_list, src_list, target_msg] # src_list clear selection?
                    )
                    
                    btn_remove.click(
                        fn=transfer_remove_logic,
                        inputs=[target_list, target_items_state],
                        outputs=[target_items_state, target_list, target_list, target_msg]
                    )

                    target_refresh_btn.click(
                        fn=refresh_target_ds_list_logic,
                        outputs=target_ds_dropdown
                    )

                    target_ds_dropdown.change(
                        fn=load_target_ds_logic,
                        inputs=target_ds_dropdown,
                        outputs=[target_items_state, target_list, target_ds_name, target_ds_type, target_ds_note, target_msg]
                    )

                    target_save_btn.click(
                        fn=save_target_ds_logic,
                        inputs=[target_items_state, target_ds_name, target_ds_type, target_ds_note, target_allow_overwrite, target_freeze],
                        outputs=target_msg
                    ).then(
                        fn=refresh_target_ds_list_logic,
                        outputs=target_ds_dropdown
                    )
                    
                    target_delete_btn.click(
                        fn=delete_target_ds_logic,
                        inputs=[target_ds_dropdown, target_del_confirm],
                        outputs=target_msg
                    ).then(
                        fn=refresh_target_ds_list_logic,
                        outputs=target_ds_dropdown
                    )


                with gr.Tab("数据集导出 (Export)", render=False) as export_tab:
                    gr.Markdown("### 📤 训练集导出 (JSONL)")
                    with gr.Row():
                        with gr.Column(scale=2):
                            export_family = gr.Dropdown(
                                label="模型类型",
                                choices=TRAINING_MODEL_FAMILIES,
                                value="chatts",
                                interactive=True
                            )
                            export_dataset_dropdown = gr.Dropdown(
                                label="选择数据集",
                                choices=[], # Need init
                                interactive=True
                            )
                            export_refresh_btn = gr.Button("🔄 刷新")
                            
                            export_output_name = gr.Textbox(
                                label="输出文件名 (Optional)",
                                placeholder="Default: dataset name"
                            )
                            export_approved_only = gr.Checkbox(
                                label="仅导出已审核通过 (approved)",
                                value=True
                            )
                        with gr.Column(scale=1):
                            export_btn = gr.Button("📤 导出训练数据", variant="primary")
                            export_status = gr.Textbox(label="导出状态", interactive=False)

                    # Export Logic
                    def refresh_export_ds_list():
                         choices = _list_dataset_assets_simple()
                         return gr.Dropdown(choices=choices)

                    def export_training_dataset(dataset_id: str, model_family: str, output_name: str, approved_only: bool):
                        if not dataset_id:
                            return "❌ 未选择数据集"
                        resp = _assets_api_call(
                            "POST",
                            "/export/training",
                            payload={
                                "dataset_id": dataset_id,
                                "model_family": model_family,
                                "output_name": (output_name or "").strip() or None,
                                "approved_only": bool(approved_only),
                            },
                        )
                        if not resp.get("success"):
                            return f"❌ 导出失败: {resp.get('error')}"
                        output_path = ((resp.get("data") or {}).get("output_path")) or ""
                        return f"✅ 导出完成: {output_path}"

                    export_refresh_btn.click(fn=refresh_export_ds_list, outputs=export_dataset_dropdown)
                    
                    export_btn.click(
                        fn=export_training_dataset,
                        inputs=[export_dataset_dropdown, export_family, export_output_name, export_approved_only],
                        outputs=export_status
                    )

                # 数据资产管理子 Tab 顺序按数据流程：
                # 降采样 -> 标注 -> 审核 -> 构建 -> 导出 -> 训练数据
                downsampled_data_tab.render()
                annotations_manage_tab.render()
                review_queue_tab.render()
                construction_tab.render()
                export_tab.render()
                training_data_tab.render()
        
        # ==================== 模型训练 Tab (原有) ====================
        with gr.Tab("🎯 模型训练", render=False) as model_training_tab:
            with gr.Row():
                with gr.Column(scale=2):
                    # 基础配置
                    gr.Markdown("### 基础配置")
                    
                    # New Dropdowns for Quick Start
                    with gr.Row():
                        model_family_dropdown = gr.Dropdown(
                            label="模型类型 (Model Family)",
                            choices=TRAINING_MODEL_FAMILIES,
                            value="chatts",
                            interactive=True
                        )
                        model_path_dropdown = gr.Dropdown(
                            label="基础模型 (Base Model)",
                            choices=get_training_adapter("chatts").get_base_models(),
                            value=get_training_adapter("chatts").get_base_models()[0] if get_training_adapter("chatts").get_base_models() else None,
                            interactive=True,
                            allow_custom_value=True
                        )
                        dataset_dropdown = gr.Dropdown(
                            label="微调数据集 (Dataset)",
                            choices=get_training_adapter("chatts").get_dataset_list(),
                            value=get_training_adapter("chatts").get_dataset_list()[0] if get_training_adapter("chatts").get_dataset_list() else None,
                            interactive=True
                        )
                    with gr.Row():
                        train_method_dropdown = gr.Dropdown(
                            label="训练方式 (Method)",
                            choices=TRAINING_METHODS,
                            value="lora",
                            interactive=True
                        )
                        config_dropdown = gr.Dropdown(
                            label="训练模板 (Template Script)",
                            choices=get_training_configs("chatts", "lora"),
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

                    # 运行与显存配置（折叠）
                    with gr.Accordion("运行与显存配置", open=False):
                        with gr.Row():
                            nproc_per_node = gr.Slider(
                                label="NPROC_PER_NODE (进程数)",
                                minimum=1,
                                maximum=8,
                                value=1,
                                step=1
                            )
                            cuda_visible_devices = gr.Textbox(
                                label="CUDA_VISIBLE_DEVICES",
                                value="0",
                                placeholder="例如: 0 或 0,1 (留空=不设置)"
                            )
                        with gr.Row():
                            grad_accum_steps = gr.Slider(
                                label="Gradient Accumulation Steps",
                                minimum=1,
                                maximum=64,
                                value=8,
                                step=1
                            )
                            precision = gr.Dropdown(
                                label="Precision",
                                choices=["bf16", "fp16", "fp32"],
                                value="bf16"
                            )
                        with gr.Row():
                            cutoff_len = gr.Number(
                                label="cutoff_len",
                                value=4096
                            )
                            image_max_pixels = gr.Number(
                                label="image_max_pixels",
                                value=3200000
                            )
                            image_min_pixels = gr.Number(
                                label="image_min_pixels",
                                value=1024
                            )
                        extra_args = gr.Textbox(
                            label="额外参数 (Raw Args)",
                            placeholder="例如: --logging_steps 5 --save_steps 50",
                            lines=2
                        )

                    # 高级训练参数（折叠）
                    with gr.Accordion("高级训练参数", open=False):
                        with gr.Row():
                            logging_steps = gr.Textbox(
                                label="logging_steps (可选)",
                                placeholder="留空不覆盖"
                            )
                            save_steps = gr.Textbox(
                                label="save_steps (可选)",
                                placeholder="留空不覆盖"
                            )
                            lr_scheduler_type = gr.Dropdown(
                                label="lr_scheduler_type (可选)",
                                choices=["", "cosine", "linear", "constant", "polynomial", "cosine_with_restarts", "constant_with_warmup"],
                                value=""
                            )
                        with gr.Row():
                            warmup_steps = gr.Textbox(
                                label="warmup_steps (可选)",
                                placeholder="留空不覆盖"
                            )
                            warmup_ratio = gr.Textbox(
                                label="warmup_ratio (可选)",
                                placeholder="留空不覆盖"
                            )
                        with gr.Row():
                            lora_dropout = gr.Textbox(
                                label="lora_dropout (可选)",
                                placeholder="留空不覆盖"
                            )
                            lora_target = gr.Textbox(
                                label="lora_target (可选)",
                                placeholder="如: q_proj,k_proj,v_proj,o_proj"
                            )
                        with gr.Row():
                            freeze_vision_tower = gr.Dropdown(
                                label="freeze_vision_tower (可选)",
                                choices=["", "True", "False"],
                                value=""
                            )
                            freeze_multi_modal_projector = gr.Dropdown(
                                label="freeze_multi_modal_projector (可选)",
                                choices=["", "True", "False"],
                                value=""
                            )
                        with gr.Row():
                            freeze_trainable_layers = gr.Textbox(
                                label="freeze_trainable_layers (可选)",
                                placeholder="留空不覆盖"
                            )
                            freeze_trainable_modules = gr.Textbox(
                                label="freeze_trainable_modules (可选)",
                                placeholder="如: all"
                            )

                    with gr.Accordion("评估设置 (黄金集)", open=False):
                        auto_eval = gr.Checkbox(
                            label="训练完成后自动评估",
                            value=False
                        )
                        eval_truth_dir = gr.Textbox(
                            label="黄金标注目录 (truth_dir)",
                            value=settings.EVAL_GOLDEN_TRUTH_DIR,
                            placeholder="例如: /path/to/ground_truth"
                        )
                        eval_data_dir = gr.Textbox(
                            label="黄金数据目录 (data_dir)",
                            value=settings.EVAL_GOLDEN_DATA_DIR,
                            placeholder="例如: /path/to/data"
                        )
                        eval_dataset_name = gr.Textbox(
                            label="数据集名称",
                            value=settings.EVAL_DEFAULT_DATASET_NAME,
                            placeholder="例如: golden_100"
                        )
                        eval_output_dir = gr.Textbox(
                            label="评估输出目录 (可选)",
                            value=settings.EVAL_DEFAULT_OUTPUT_DIR,
                            placeholder="留空则写入模型目录/eval_outputs"
                        )
                        eval_device = gr.Textbox(
                            label="评估设备 (可选)",
                            placeholder="例如: cuda:0 或 cpu"
                        )
                    
                    # 控制按钮
                    # --- Backend Functions ---
                    def validate_dataset_wrap(model_family: str, dataset_name: Optional[str]) -> str:
                        data_dir = (settings.DATA_TRAINING_QWEN_DIR
                                    if model_family == "qwen"
                                    else settings.DATA_TRAINING_CHATTS_DIR)
                        info_path = Path(data_dir) / "dataset_info.json"
                        if not info_path.exists():
                            return f"❌ 找不到 dataset_info.json: {info_path}"
                        try:
                            with open(info_path, "r", encoding="utf-8") as f:
                                info = json.load(f)
                        except Exception as e:
                            return f"❌ 读取失败: {e}"

                        if not info:
                            return f"❌ dataset_info.json 为空: {info_path}"
                        if not dataset_name:
                            return "⚠️ 未选择数据集"
                        if dataset_name not in info:
                            return f"❌ 数据集未注册: {dataset_name}"

                        file_name = info[dataset_name].get("file_name")
                        if not file_name:
                            return f"❌ 未配置 file_name: {dataset_name}"
                        data_path = Path(file_name)
                        if not data_path.is_absolute():
                            data_path = Path(data_dir) / file_name

                        if not data_path.exists():
                            return f"❌ 数据不存在: {data_path}"

                        if data_path.is_dir():
                            files = list(data_path.glob("*.json")) + list(data_path.glob("*.jsonl"))
                            if not files:
                                return f"❌ 目录下没有 json/jsonl: {data_path}"
                            return f"✅ 校验通过: {dataset_name} (目录, {len(files)} 个文件)"

                        return f"✅ 校验通过: {dataset_name} ({data_path.name})"

                    def start_training_wrap(
                        model_family, config_name, lr, epochs, batch_size, rank, alpha, output_name,
                        model_path, dataset_name, nproc, cuda_devices, grad_accum, prec, cutoff,
                        img_max, img_min, extra_args, log_steps, save_steps, lr_sched, warm_steps,
                        warm_ratio, lora_drop, lora_tgt, freeze_vision, freeze_proj, freeze_layers, freeze_modules,
                        auto_eval_enabled, eval_truth, eval_data, eval_dataset, eval_output, eval_device_val
                    ):
                        if not config_name:
                            return "❌ 请选择训练模板", ""

                        cuda_devices = (cuda_devices or "").strip() or None
                        extra_args = (extra_args or "").strip() or None
                        log_steps = (log_steps or "").strip() or None
                        save_steps = (save_steps or "").strip() or None
                        lr_sched = (lr_sched or "").strip() or None
                        warm_steps = (warm_steps or "").strip() or None
                        warm_ratio = (warm_ratio or "").strip() or None
                        lora_drop = (lora_drop or "").strip() or None
                        lora_tgt = (lora_tgt or "").strip() or None
                        freeze_vision = (freeze_vision or "").strip() or None
                        freeze_proj = (freeze_proj or "").strip() or None
                        freeze_layers = (freeze_layers or "").strip() or None
                        freeze_modules = (freeze_modules or "").strip() or None
                        eval_truth = (eval_truth or "").strip() or None
                        eval_data = (eval_data or "").strip() or None
                        eval_dataset = (eval_dataset or "").strip() or None
                        eval_output = (eval_output or "").strip() or None
                        eval_device_val = (eval_device_val or "").strip() or None
                        version_tag = (output_name or "").strip() or None

                        overrides = {
                            "override_learning_rate": lr,
                            "override_epochs": epochs,
                            "override_batch_size": batch_size,
                            "override_lora_rank": rank,
                            "override_lora_alpha": alpha,
                            "override_model_path": model_path,
                            "override_dataset": dataset_name,
                            "override_nproc_per_node": nproc,
                            "override_cuda_visible_devices": cuda_devices,
                            "override_grad_accum_steps": grad_accum,
                            "override_precision": prec,
                            "override_cutoff_len": cutoff,
                            "override_image_max_pixels": img_max,
                            "override_image_min_pixels": img_min,
                            "override_extra_args": extra_args,
                            "override_logging_steps": log_steps,
                            "override_save_steps": save_steps,
                            "override_lr_scheduler_type": lr_sched,
                            "override_warmup_steps": warm_steps,
                            "override_warmup_ratio": warm_ratio,
                            "override_lora_dropout": lora_drop,
                            "override_lora_target": lora_tgt,
                            "override_freeze_vision_tower": freeze_vision,
                            "override_freeze_multi_modal_projector": freeze_proj,
                            "override_freeze_trainable_layers": freeze_layers,
                            "override_freeze_trainable_modules": freeze_modules
                        }

                        submit_payload = {
                            "config_name": config_name,
                            "version_tag": version_tag,
                            "model_family": model_family,
                            "auto_eval": bool(auto_eval_enabled),
                            "eval_truth_dir": eval_truth,
                            "eval_data_dir": eval_data,
                            "eval_dataset_name": eval_dataset,
                            "eval_output_dir": eval_output,
                            "eval_device": eval_device_val,
                            "params": {k: v for k, v in overrides.items() if v is not None},
                        }
                        submit_resp = _training_api_call("post", "/start", payload=submit_payload)
                        if not submit_resp.get("success"):
                            return f"❌ 启动错误: {submit_resp.get('error')}", ""

                        submit_data = submit_resp.get("data") or {}
                        task_id = str(submit_data.get("task_id") or "")
                        if not task_id:
                            return "❌ 启动成功但未返回 task_id", ""

                        adapter = get_training_adapter(model_family)
                        output_dir = adapter.saves_path / f"{config_name}_{version_tag or task_id[:8]}"
                        logger.info("训练任务已提交: task_id=%s family=%s config=%s", task_id, model_family, config_name)

                        return (
                            f"✅ 训练任务已成功提交!\n任务ID: {task_id}\n输出目录: {output_dir}\n\n正在后台运行中... 请留意下方实时日志与状态。",
                            task_id,
                        )

                    def stream_logs(task_id, current_log_text, offset, current_status_text):
                        if not task_id:
                            return format_log_html(current_log_text), current_log_text, offset, current_status_text, ""

                        new_offset = int(offset or 0)
                        log_resp = _training_api_call(
                            "get",
                            f"/log/{task_id}",
                            params={"offset": new_offset, "max_bytes": 200000},
                        )
                        if not log_resp.get("success"):
                            status_text = f"⚠️ 日志查询失败: {log_resp.get('error') or '未知错误'}"
                            return format_log_html(current_log_text), current_log_text, new_offset, status_text, task_id

                        log_payload = log_resp.get("data") or {}
                        log_data = log_payload.get("data") if isinstance(log_payload, dict) else {}
                        new_content = str((log_data or {}).get("log") or "")
                        try:
                            new_offset = int((log_data or {}).get("offset", new_offset))
                        except Exception:
                            pass
                        task_status = str((log_data or {}).get("status") or "").lower()
                        task_error = str((log_data or {}).get("error") or "")

                        if new_content:
                            current_log_text = (current_log_text or "") + new_content
                            if len(current_log_text) > LOG_TAIL_MAX_CHARS:
                                current_log_text = current_log_text[-LOG_TAIL_MAX_CHARS:]

                        if task_status in {"pending", "running"}:
                            status_text = f"🔄 训练进行中: {task_id} ({task_status})"
                            return format_log_html(current_log_text), current_log_text, new_offset, status_text, task_id

                        if task_status == "completed":
                            status_text = f"✅ 训练任务已完成: {task_id}"
                            return format_log_html(current_log_text), current_log_text, new_offset, status_text, ""

                        if task_status == "cancelled":
                            status_text = f"🛑 训练任务已取消: {task_id}"
                            return format_log_html(current_log_text), current_log_text, new_offset, status_text, ""

                        if task_status == "failed":
                            msg = task_error or "训练失败"
                            status_text = f"❌ 训练任务失败: {msg}"
                            return format_log_html(current_log_text), current_log_text, new_offset, status_text, ""

                        status_text = f"ℹ️ 训练状态: {task_status or 'unknown'}"
                        return format_log_html(current_log_text), current_log_text, new_offset, status_text, task_id

                    def stop_training_wrap(task_id):
                        if not task_id:
                            return "无运行中的任务", ""

                        resp = _training_api_call("post", f"/stop/{task_id}")
                        if resp.get("success"):
                            payload = resp.get("data") or {}
                            msg = str(payload.get("message") or "训练已停止")
                            return f"🛑 {msg}", ""

                        api_error = str(resp.get("error") or "停止失败")
                        if "任务不存在" in api_error or "任务未在运行中" in api_error:
                            return f"ℹ️ {api_error}", ""
                        return f"❌ 停止失败: {api_error}", task_id

                    # --- Layout & Events ---
                    with gr.Column():
                        with gr.Row():
                            start_btn = gr.Button("🚀 (Quick Start) 开始训练", variant="primary", scale=2)
                            stop_btn = gr.Button("🛑 停止训练", variant="stop", scale=1)
                            validate_btn = gr.Button("✅ 校验数据", scale=1)
                            refresh_btn = gr.Button("🔄 刷新配置", scale=1)
                        
                        # Hidden state for Task ID and Log Offset
                        task_id_state = gr.State("")
                        log_offset_state = gr.State(0)
                        training_log_state = gr.State("") # Raw text state
                        
                        output_box = gr.Textbox(label="训练状态", lines=4)
                        validate_box = gr.Textbox(label="数据校验结果", lines=2)
                        log_box = gr.HTML(label="实时日志 (Real-time Logs)", value=format_log_html(""), elem_id="training-log")
                        
                        # Timer for polling
                        timer = gr.Timer(1) # 1s interval

                        # Events
                        start_btn.click(
                            fn=start_training_wrap,
                            inputs=[
                                model_family_dropdown, config_dropdown, learning_rate, num_epochs,
                                batch_size, lora_rank, lora_alpha, output_name,
                                model_path_dropdown, dataset_dropdown,
                                nproc_per_node, cuda_visible_devices, grad_accum_steps, precision,
                                cutoff_len, image_max_pixels, image_min_pixels, extra_args,
                                logging_steps, save_steps, lr_scheduler_type, warmup_steps,
                                warmup_ratio, lora_dropout, lora_target, freeze_vision_tower,
                                freeze_multi_modal_projector, freeze_trainable_layers, freeze_trainable_modules,
                                auto_eval, eval_truth_dir, eval_data_dir, eval_dataset_name, eval_output_dir, eval_device
                            ],
                            outputs=[output_box, task_id_state],
                            queue=False
                        ).then(
                            fn=lambda: 0, outputs=log_offset_state # Reset offset
                        ).then(
                            fn=lambda: "", outputs=training_log_state # Clear raw log
                        ).then(
                            fn=lambda: format_log_html(""), outputs=log_box # Clear visual log
                        )

                        model_family_dropdown.change(
                            fn=update_training_dropdowns,
                            inputs=[model_family_dropdown, train_method_dropdown],
                            outputs=[config_dropdown, dataset_dropdown, model_path_dropdown]
                        )

                        train_method_dropdown.change(
                            fn=update_training_config_only,
                            inputs=[model_family_dropdown, train_method_dropdown],
                            outputs=config_dropdown
                        )
                        
                        stop_btn.click(
                            fn=stop_training_wrap,
                            inputs=[task_id_state],
                            outputs=[output_box, task_id_state]
                        )

                        validate_btn.click(
                            fn=validate_dataset_wrap,
                            inputs=[model_family_dropdown, dataset_dropdown],
                            outputs=validate_box
                        )
                        
                        # Timer ticks -> Update logs
                        timer.tick(
                            fn=stream_logs,
                            inputs=[task_id_state, training_log_state, log_offset_state, output_box],
                            outputs=[log_box, training_log_state, log_offset_state, output_box, task_id_state],
                            queue=False
                        )
                        refresh_btn.click(
                            fn=update_training_dropdowns,
                            inputs=[model_family_dropdown, train_method_dropdown],
                            outputs=[config_dropdown, dataset_dropdown, model_path_dropdown]
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
        
        with gr.Tab("📊 已训练模型", render=False) as trained_models_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型列表")
                    with gr.Row():
                        model_family_view = gr.Dropdown(
                            label="模型类型 (Model Family)",
                            choices=TRAINING_MODEL_FAMILIES,
                            value="chatts",
                            interactive=True
                        )
                        model_type_view = gr.Dropdown(
                            label="训练方式 (Method)",
                            choices=TRAINING_METHODS,
                            value="all",
                            interactive=True
                        )
                        include_ckpt_view = gr.Checkbox(
                            label="显示 checkpoint",
                            value=False
                        )
                    model_dropdown = gr.Dropdown(
                        label="选择模型",
                        choices=get_trained_model_choices("chatts", "all", False),
                        interactive=True
                    )
                    refresh_models_btn = gr.Button("🔄 刷新列表")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 模型详情")
                    model_info = gr.Markdown(value="请选择一个模型")
                    
                    gr.Markdown("### Loss 曲线")
                    loss_image = gr.Image(label="Training Loss", type="filepath")

                    gr.Markdown("### 模型评估 (黄金集)")
                    eval_status = gr.Textbox(label="评估状态", lines=2)
                    eval_summary = gr.JSON(label="评估摘要")
                    eval_truth_dir_ui = gr.Textbox(
                        label="黄金标注目录 (truth_dir)",
                        value=settings.EVAL_GOLDEN_TRUTH_DIR,
                    )
                    eval_data_dir_ui = gr.Textbox(
                        label="黄金数据目录 (data_dir)",
                        value=settings.EVAL_GOLDEN_DATA_DIR,
                    )
                    eval_dataset_name_ui = gr.Textbox(
                        label="数据集名称",
                        value=settings.EVAL_DEFAULT_DATASET_NAME,
                    )
                    eval_output_dir_ui = gr.Textbox(
                        label="评估输出目录 (可选)",
                        value=settings.EVAL_DEFAULT_OUTPUT_DIR,
                    )
                    eval_device_ui = gr.Textbox(
                        label="评估设备 (可选)",
                        placeholder="例如: cuda:0 或 cpu"
                    )
                    eval_btn = gr.Button("✅ 运行评估", variant="primary")
            
            # 事件绑定
            model_dropdown.change(
                fn=get_model_info,
                inputs=[model_dropdown, model_family_view],
                outputs=model_info
            )
            model_dropdown.change(
                fn=get_loss_plot,
                inputs=[model_dropdown, model_family_view],
                outputs=loss_image
            )
            eval_btn.click(
                fn=run_model_evaluation_ui,
                inputs=[
                    model_dropdown,
                    model_family_view,
                    eval_truth_dir_ui,
                    eval_data_dir_ui,
                    eval_dataset_name_ui,
                    eval_output_dir_ui,
                    eval_device_ui,
                ],
                outputs=[eval_status, eval_summary],
            )
            refresh_models_btn.click(
                fn=lambda mf, mt, ck: gr.Dropdown(choices=get_trained_model_choices(mf, mt, ck)),
                inputs=[model_family_view, model_type_view, include_ckpt_view],
                outputs=model_dropdown
            )
            model_family_view.change(
                fn=lambda mf, mt, ck: gr.Dropdown(choices=get_trained_model_choices(mf, mt, ck)),
                inputs=[model_family_view, model_type_view, include_ckpt_view],
                outputs=model_dropdown
            )
            model_type_view.change(
                fn=lambda mf, mt, ck: gr.Dropdown(choices=get_trained_model_choices(mf, mt, ck)),
                inputs=[model_family_view, model_type_view, include_ckpt_view],
                outputs=model_dropdown
            )
            include_ckpt_view.change(
                fn=lambda mf, mt, ck: gr.Dropdown(choices=get_trained_model_choices(mf, mt, ck)),
                inputs=[model_family_view, model_type_view, include_ckpt_view],
                outputs=model_dropdown
            )
            
        with gr.Tab("⚖️ 模型对比", render=False) as model_compare_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 选择对比模型")
                    with gr.Row():
                        compare_family = gr.Dropdown(
                            label="模型类型 (Model Family)",
                            choices=TRAINING_MODEL_FAMILIES,
                            value="chatts",
                            interactive=True
                        )
                        compare_type = gr.Dropdown(
                            label="训练方式 (Method)",
                            choices=TRAINING_METHODS,
                            value="all",
                            interactive=True
                        )
                        compare_include_ckpt = gr.Checkbox(
                            label="显示 checkpoint",
                            value=False
                        )
                    compare_models = gr.CheckboxGroup(
                        label="模型列表",
                        choices=get_trained_model_choices("chatts", "all", False)
                    )
                    compare_btn = gr.Button("📊 生成对比图", variant="primary")
                    refresh_compare_btn = gr.Button("🔄 刷新列表")
                
                with gr.Column(scale=3):
                    gr.Markdown("### 对比结果")
                    eval_metrics_table = gr.Dataframe(label="🏅 黄金集评估指标 (F1/Precision/Recall)", interactive=False)
                    comparison_plot = gr.Image(label="Loss Comparison")
            
            compare_btn.click(
                fn=get_comparison_plot,
                inputs=[compare_models, compare_family],
                outputs=comparison_plot
            ).then(
                fn=get_comparison_metrics,
                inputs=[compare_models],
                outputs=eval_metrics_table
            )
            refresh_compare_btn.click(
                fn=lambda mf, mt, ck: gr.CheckboxGroup(choices=get_trained_model_choices(mf, mt, ck)),
                inputs=[compare_family, compare_type, compare_include_ckpt],
                outputs=compare_models
            )
            compare_family.change(
                fn=lambda mf, mt, ck: gr.CheckboxGroup(choices=get_trained_model_choices(mf, mt, ck)),
                inputs=[compare_family, compare_type, compare_include_ckpt],
                outputs=compare_models
            )
            compare_type.change(
                fn=lambda mf, mt, ck: gr.CheckboxGroup(choices=get_trained_model_choices(mf, mt, ck)),
                inputs=[compare_family, compare_type, compare_include_ckpt],
                outputs=compare_models
            )
            compare_include_ckpt.change(
                fn=lambda mf, mt, ck: gr.CheckboxGroup(choices=get_trained_model_choices(mf, mt, ck)),
                inputs=[compare_family, compare_type, compare_include_ckpt],
                outputs=compare_models
            )
        
        with gr.Tab("⚙️ 配置说明", render=False) as config_help_tab:
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

可用的训练配置来自 `/home/douff/ts/ts-iteration-loop/services/training/scripts/chatts/lora/` 和
`/home/douff/ts/ts-iteration-loop/services/training/scripts/qwen/lora/` 目录。
""")
    
        # ==================== 顶层 Tab 顺序重排 ====================
        # 用户要求顺序:
        # 数据获取 -> 推理监控 -> 标注工具 -> 模型训练 -> 数据资产管理 -> 模型资产管理
        model_training_tab.render()
        data_assets_tab.render()
        with gr.Tab("🧠 模型资产管理"):
            with gr.Tabs():
                trained_models_tab.render()
                model_compare_tab.render()
                config_help_tab.render()

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
