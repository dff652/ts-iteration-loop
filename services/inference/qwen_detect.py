import os
import io
import re
import json
import logging
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# 配置日志
logger = logging.getLogger("QwenDetect")

# ========== 单例模型缓存 ==========
_qwen_model = None
_qwen_processor = None
_qwen_config = None


def get_qwen_model(model_path: str, device: str = "cuda:0", load_in_4bit: bool = True):
    """
    获取 Qwen VL 模型实例（单例模式）

    Args:
        model_path: 模型路径
        device: GPU 设备, e.g. "cuda:0", "cuda:1"
        load_in_4bit: 是否 4-bit 量化（27B 模型必须开启）
    """
    global _qwen_model, _qwen_processor, _qwen_config

    new_config = {"model_path": model_path, "device": device, "load_in_4bit": load_in_4bit}

    if _qwen_model is not None and _qwen_config == new_config:
        return _qwen_model, _qwen_processor

    print(f"[Qwen] 正在加载模型: {model_path} 到 {device} (4bit={load_in_4bit})...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config

    model_kwargs["dtype"] = compute_dtype

    # 解析 device：支持 "cuda:0,cuda:1" 多卡格式和 "auto"
    if "," in device:
        gpu_ids = [int(d.strip().replace("cuda:", "")) for d in device.split(",")]
        max_memory = {}
        for gid in gpu_ids:
            if gid < torch.cuda.device_count():
                total = torch.cuda.get_device_properties(gid).total_memory
                max_memory[gid] = f"{int(total / 1024**3) - 1}GiB"
        max_memory["cpu"] = "32GiB"
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = max_memory
        print(f"[Qwen] 多 GPU 模式: {gpu_ids}, max_memory: {max_memory}")
    elif device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = device

    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)

    _qwen_model = model
    _qwen_processor = processor
    _qwen_config = new_config
    print(f"[Qwen] 模型加载完成")

    return model, processor


class JSONParser:
    """JSON解析，处理非标准JSON格式"""

    @staticmethod
    def robust_json_loads(json_str: str):
        """鲁棒的JSON解析"""
        if not json_str or not json_str.strip():
            return None

        json_str = json_str.strip()

        # 尝试清理 markdown 代码块标记
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        # 方法1: 标准JSON解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 方法2: 预处理后JSON解析
        try:
            fixed = JSONParser._preprocess_json_str(json_str)
            return json.loads(fixed)
        except Exception:
            pass

        # 方法3: 正则提取
        try:
            return JSONParser._extract_with_regex(json_str)
        except Exception:
            pass

        return None

    @staticmethod
    def _preprocess_json_str(json_str: str) -> str:
        """预处理JSON字符串，修复常见格式错误"""
        if not json_str: return ""
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # 去掉尾部逗号
        return json_str

    @staticmethod
    def _extract_with_regex(json_str: str):
        """使用正则表达式提取 interval 和 type"""
        result = {"detected_anomalies": []}
        interval_patterns = [r'\[(\d+)\s*,\s*(\d+)\]', r'\((\d+)\s*,\s*(\d+)\)']

        for pattern in interval_patterns:
            found = re.findall(pattern, json_str)
            for start, end in found:
                result["detected_anomalies"].append({
                    "interval": [int(start), int(end)],
                    "type": "unknown",
                    "reason": "regex_extracted"
                })
        return result

class ImageGenerator:
    """生成时序图"""
    @staticmethod
    def create_single_image(data: pd.Series, title: str = "", dpi: int = 100):
        fig, ax = plt.subplots(figsize=(20, 4), dpi=dpi)
        ax.plot(data.values, color='black', linewidth=1)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img

def qwen_detect(
    data: pd.DataFrame,
    model_path: str,
    device: str = "cuda:0",
    prompt_template_name: str = "default",
    n_downsample: int = 5000,
    downsampler: str = "m4",
    load_in_4bit: bool = True,
    **kwargs,
):
    """
    Qwen VL 模型推理入口函数

    Args:
        data: 输入DataFrame，通常包含单列数值
        model_path: 模型路径
        device: GPU 设备, e.g. "cuda:0", "cuda:1"
        prompt_template_name: 提示词模板名称
        n_downsample: 降采样目标点数
        downsampler: 降采样方法 (m4/minmax)
        load_in_4bit: 是否 4-bit 量化

    Returns:
        mask: 异常掩码 (numpy array, 0/1)
        anomalies: 异常列表
        position_index: 降采样后的索引
    """

    # 1. 数据准备
    if data.empty:
        return np.zeros(0), [], None

    series = data.iloc[:, 0]

    # 降采样
    try:
        if downsampler is None or str(downsampler).lower() == "none":
            series_ds = series
            position_index = np.arange(len(series))
        elif len(series) > n_downsample:
            if str(downsampler).lower() == "minmax":
                from tsdownsample import MinMaxLTTBDownsampler
                sampler = MinMaxLTTBDownsampler()
            else:
                from tsdownsample import M4Downsampler
                sampler = M4Downsampler()
            idx = sampler.downsample(series.values, n_out=n_downsample)
            series_ds = series.iloc[idx]
            position_index = idx
        else:
            series_ds = series
            position_index = np.arange(len(series))
    except ImportError:
        logger.warning("tsdownsample not found, using raw data")
        series_ds = series
        position_index = np.arange(len(series))

    # 2. 模型加载（单例）
    try:
        model, processor = get_qwen_model(model_path, device, load_in_4bit)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return np.zeros(len(data)), [], position_index

    # 3. 图像生成
    image = ImageGenerator.create_single_image(series_ds, title="Time Series Limit Check")

    # 4. 构造 Prompt
    user_prompt = """分析图中的时间序列数据，基于信号特征识别异常区域。
输出必须是标准JSON格式：{"detected_anomalies":[{"interval":[start,end],"type":"类型","reason":"原因"}]}；若无异常：{"detected_anomalies":[]}。
异常区域必须以连续索引区间 [start, end] 表示，且满足 end - start + 1 > 5。
请精确标注异常区间的起止索引。"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image"},
            ],
        }
    ]

    # 5. 推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    # 4-bit 量化模型已在 device 上，input tensors 需移过去
    input_device = next(model.parameters()).device
    inputs = inputs.to(input_device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"[Qwen] 模型输出: {output_text[:500]}")

    # 6. 解析结果
    parsed = JSONParser.robust_json_loads(output_text)

    # 7. 生成 Mask
    mask = np.zeros(len(series_ds), dtype=int)
    anomalies = []

    if parsed and "detected_anomalies" in parsed:
        for item in parsed["detected_anomalies"]:
            start, end = item.get("interval", [0, 0])
            start = max(0, min(start, len(series_ds)-1))
            end = max(start, min(end, len(series_ds)))
            mask[start:end] = 1
            anomalies.append(item)

    # 如果进行了降采样，映射回原始长度
    if len(mask) != len(data):
        full_mask = np.zeros(len(data), dtype=int)
        for item in anomalies:
            start, end = item.get("interval", [0, 0])
            if hasattr(position_index, '__getitem__'):
                real_start = position_index[min(start, len(position_index)-1)]
                real_end = position_index[min(end-1, len(position_index)-1)]
                full_mask[real_start:real_end+1] = 1

        return full_mask, anomalies, position_index

    return mask, anomalies, position_index
