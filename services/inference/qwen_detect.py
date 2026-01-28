import os
import io
import re
import json
import logging
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# 配置日志
logger = logging.getLogger("QwenDetect")

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
        
        # 方法2: 预处理后JSON解析 (保留原文的正则替换逻辑)
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
        # 简单空格清理
        # 此处简化保留原文核心逻辑
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str) # 去掉尾部逗号
        return json_str

    @staticmethod
    def _extract_with_regex(json_str: str):
        """使用正则表达式提取 interval 和 type"""
        # 简化版实现，提取 interval=[start, end]
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
        # 简化绘图逻辑，用于模型输入
        fig, ax = plt.subplots(figsize=(20, 4), dpi=dpi)
        ax.plot(data.values, color='black', linewidth=1)
        ax.set_title(title)
        ax.axis('off') # 关闭坐标轴以减少干扰? 原文保留了坐标轴，这里保留基本绘图
        
        # 紧凑布局
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img

def qwen_detect(data: pd.DataFrame, model_path: str, device: str = "cuda", 
               prompt_template_name: str = "default", **kwargs):
    """
    Qwen 模型推理入口函数
    
    Args:
        data: 输入DataFrame，通常包含单列数值
        model_path: 模型路径
        device: 'cuda' 或 'cpu'
        prompt_template_name: 提示词模板名称 (预留)
        
    Returns:
        mask: 异常掩码 (numpy array, 0/1)
        anomalies: 异常列表
        position_index: 降采样后的索引 (如果进行了降采样)
    """
    
    # 1. 数据准备
    if data.empty:
        return np.zeros(0), [], None
        
    # 提取第一列作为数值列
    series = data.iloc[:, 0]
    
    # 降采样逻辑 (简单实现，如果需要复杂M4降采样可引用 tsdownsample)
    # 这里假设传入数据长度适中，或者直接使用 tsdownsample
    try:
        from tsdownsample import M4Downsampler
        if len(series) > 5000:
            downsampler = M4Downsampler()
            idx = downsampler.downsample(series.values, n_out=5000)
            series_ds = series.iloc[idx]
            position_index = idx
        else:
            series_ds = series
            position_index = np.arange(len(series))
    except ImportError:
        logger.warning("tsdownsample not found, using raw data")
        series_ds = series
        position_index = np.arange(len(series))

    # 2. 模型加载
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device
        )
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        # 返回全0掩码
        return np.zeros(len(data)), [], position_index

    # 3. 图像生成
    image = ImageGenerator.create_single_image(series_ds, title="Time Series Limit Check")
    
    # 4. 构造 Prompt
    user_prompt = """分析图中的时间序列数据，基于信号特征识别异常区域。
    输出必须是标准JSON格式：{"detected_anomalies":[{"interval":[start,end],"type":"类型","reason":"原因"}]}；若无异常：{"detected_anomalies":[]}。
    异常区域必须以连续索引区间 [start, end] 表示，且满足 end - start + 1 > 5。
    请精确标注异常区间的起止索引。"""
    
    # 参考 Qwen3-VL_test.py 的格式
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
    
    # 移除 process_vision_info, 直接传 images=[image]
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 6. 解析结果
    parsed = JSONParser.robust_json_loads(output_text)
    
    # 7. 生成 Mask
    mask = np.zeros(len(series_ds), dtype=int)
    anomalies = []
    
    if parsed and "detected_anomalies" in parsed:
        for item in parsed["detected_anomalies"]:
            start, end = item.get("interval", [0, 0])
            # 确保范围有效
            start = max(0, min(start, len(series_ds)-1))
            end = max(start, min(end, len(series_ds)))
            mask[start:end] = 1
            anomalies.append(item)
            
    # 如果进行了降采样，需要映射回原始长度? 
    # run.py 的逻辑通常是返回降采样后的 mask 和 position_index，由 run.py 或後处理负责
    # 但根据 run.py `model_inference` 后的代码，它似乎期望 global_mask 也是降采样后的?
    # 不，run.py line 793 `data = raw_data.copy()` 且 line 799 `data['outlier_mask'] = outlier_mask`
    # 这意味着 run.py 期望 outlier_mask 与 raw_data 长度一致。
    # 如果 chatts_detect 返回了降采样 mask, run.py 需要做插值还原。
    # 让我们检查 chatts_detect 的返回值约定。
    # run.py line 768: global_mask, anomalies, position_index_ds = chatts_detect(...)
    # 并没有看到后续有插值代码，直接赋值给了 data['outlier_mask']?
    # No, wait. 
    # 如果 outlier_mask 长度 != len(data)，赋值会报错。
    # 所以 chatts_detect 必须返回原始长度的 mask。

    if len(mask) != len(data):
        # 简单还原：创建一个全长mask
        full_mask = np.zeros(len(data), dtype=int)
        # 将降采样索引对应的位置设为 mask 值
        # 这只是近似
        # 正确做法：如果是区间 [s, e] 在降采样空间，对应原始空间的 [pos[s], pos[e]]
        for start, end in [item.get("interval", [0, 0]) for item in anomalies]:
             # 找到对应的原始索引
             if hasattr(position_index, '__getitem__'):
                 real_start = position_index[min(start, len(position_index)-1)]
                 real_end = position_index[min(end-1, len(position_index)-1)] # end是exclusive
                 full_mask[real_start:real_end+1] = 1
        
        return full_mask, anomalies, position_index

    return mask, anomalies, position_index
