"""
Qwen VL 时序异常检测模块

与 /home/douff/ilabel/qwen3-vl-8B-test/reasoning/Qwen3-VL_test.py 保持一致的图像生成和推理逻辑
"""

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from io import BytesIO
from typing import Tuple, List, Optional, Dict

# 配置日志
logger = logging.getLogger("QwenDetect")


# ==================== 自定义异常 ====================
class QwenInferenceError(Exception):
    """Qwen 推理过程中的错误，调用方应捕获此异常并处理失败逻辑"""
    pass


# ==================== 图像配置 ====================
class Config:
    """图像生成配置，与测试脚本保持一致"""
    FIGURE_WIDTH: int = 20
    FIGURE_HEIGHT: int = 4
    FIGURE_DPI: int = 200
    PLOT_LINEWIDTH: float = 0.5
    X_TICKS_COUNT: int = 150


# ==================== JSON解析工具 ====================
class JSONParser:
    """JSON解析，处理非标准JSON格式"""
    
    @staticmethod
    def robust_json_loads(json_str: str) -> Optional[Dict]:
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
        if not json_str:
            return ""
        # 去掉尾部逗号
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        return json_str

    @staticmethod
    def _extract_with_regex(json_str: str) -> Optional[Dict]:
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


# ==================== 图像生成器 ====================
class ImageGenerator:
    """时间序列图像生成，与测试脚本保持一致"""
    
    @staticmethod
    def create_single_image(
        data: pd.Series,
        y_range: Optional[Tuple[float, float]] = None,
        start_index: int = 0,
        highlight_regions: Optional[List[Tuple[int, int]]] = None,
        dpi: int = Config.FIGURE_DPI,
    ) -> Image.Image:
        """
        创建单个时间序列图像，与测试脚本 ImageGenerator.create_single_image 完全一致
        """
        fig, ax = plt.subplots(figsize=(Config.FIGURE_WIDTH, Config.FIGURE_HEIGHT), dpi=dpi)
        
        # X轴使用索引
        x_indices = np.arange(start_index, start_index + len(data))
        
        # 绘制曲线
        ax.plot(x_indices, data.values, color='black', linewidth=Config.PLOT_LINEWIDTH, alpha=1)
        
        # 禁用科学计数法
        ax.ticklabel_format(useOffset=False, style='plain', axis='both')
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        
        # X轴刻度设置
        if len(data) > 1:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=min(Config.X_TICKS_COUNT, len(data)), integer=True))
            ax.tick_params(axis='x', rotation=90, labelsize=6)
        
        # 添加网格线
        ax.figure.canvas.draw()
        xticks = ax.get_xticks()
        for tick in xticks:
            if start_index <= tick <= start_index + len(data):
                ax.axvline(x=tick, color='lightblue', linewidth=0.5, alpha=0.3, zorder=0)
        
        # 高亮异常区域
        if highlight_regions:
            for start, end in highlight_regions:
                if start < end <= start_index + len(data):
                    ax.axvspan(start, end, color='#e74c3c', alpha=0.3)
        
        # Y轴范围设置
        if y_range is None:
            y_min, y_max = data.min(), data.max()
        else:
            y_min, y_max = y_range
        
        if abs(y_max - y_min) > 1e-10:
            pad = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - pad, y_max + pad)
        
        # 保存到内存
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        
        img = Image.open(buf).convert("RGB")
        return img


# ==================== 主推理函数 ====================
def qwen_detect(
    data: pd.DataFrame,
    model_path: str,
    device: str = "cuda",
    prompt_template_name: str = "default",
    n_downsample: int = 5000,
    downsampler: str = "m4",
    max_new_tokens: int = 2048,
    image_path: Optional[str] = None,
    debug_output_path: Optional[str] = None,
    **kwargs,
):
    """
    Qwen 模型推理入口函数
    
    Args:
        data: 输入DataFrame，通常包含单列数值
        model_path: 模型路径
        device: 'cuda' 或 'cpu'
        prompt_template_name: 提示词模板名称 (预留)
        n_downsample: 降采样点数
        downsampler: 降采样方法 'm4' 或 'minmax'
        max_new_tokens: 最大生成token数 (默认2048)
        
    Returns:
        mask: 异常掩码 (numpy array, 0/1)
        anomalies: 异常列表
        position_index: 降采样后的索引
    """
    
    # 1. 数据准备
    if data.empty:
        return np.zeros(0), [], None
        
    # 提取第一列作为数值列
    series = data.iloc[:, 0]
    
    # 降采样逻辑
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

    # 2. 模型加载
    if debug_output_path:
        try:
            os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)
            with open(debug_output_path, "w", encoding="utf-8") as f:
                f.write("=== QWEN DEBUG ===\n")
                f.write(f"model_path: {model_path}\n")
                f.write(f"device: {device}\n")
                f.write(f"data_len: {len(data)}\n")
                f.write(f"image_path: {image_path}\n")
                f.write(f"downsampler: {downsampler}, n_downsample: {n_downsample}\n")
                f.write("\n")
        except Exception as e:
            print(f"[QWEN DEBUG] Failed to init debug file: {e}")

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device
        )
    except Exception as e:
        error_msg = f"Failed to load model from {model_path}: {e}"
        logger.error(error_msg)
        print(f"[QWEN DEBUG] Model load failed: {e}")
        if debug_output_path:
            try:
                with open(debug_output_path, "a", encoding="utf-8") as f:
                    f.write("=== ERROR ===\n")
                    f.write(f"Model load failed: {e}\n")
            except Exception:
                pass
        # 抛出异常而非静默返回，让调用方知道失败
        raise QwenInferenceError(error_msg)

    # 3. 图像生成（使用与测试脚本一致的方法）
    image = None
    if image_path:
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                logger.info(f"Using pre-generated image: {image_path}")
        except Exception as e:
            logger.warning(f"Failed to load pre-generated image: {image_path} | {e}")

    if image is None:
        y_range = (series_ds.min(), series_ds.max())
        image = ImageGenerator.create_single_image(
            data=series_ds.reset_index(drop=True),
            y_range=y_range,
            start_index=0,
            highlight_regions=None,
            dpi=Config.FIGURE_DPI,
        )
    
    # 4. 构造 Prompt (与测试脚本完全一致，包括缩进)
    # 测试脚本中的 Prompt 第一行无缩进，后续行有 8 个空格缩进
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
    inputs = inputs.to(device)

    try:
        # 使用传入参数 max_new_tokens
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        full_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    except Exception as e:
        error_msg = f"Qwen generate failed: {e}"
        logger.error(error_msg)
        print(f"[QWEN DEBUG] Generate failed: {e}")
        if debug_output_path:
            try:
                with open(debug_output_path, "a", encoding="utf-8") as f:
                    f.write("=== ERROR ===\n")
                    f.write(f"Generate failed: {e}\n")
            except Exception:
                pass
        # 抛出异常而非静默返回
        raise QwenInferenceError(error_msg)

    # Print debug snippets to stdout (keep short to avoid log spam)
    try:
        print("=== QWEN output_text (trimmed, first 2000 chars) ===")
        print(output_text[:2000])
        print("=== QWEN full_text (first 2000 chars) ===")
        print(full_text[:2000])
    except Exception:
        pass

    if debug_output_path:
        try:
            os.makedirs(os.path.dirname(debug_output_path), exist_ok=True)
            with open(debug_output_path, "a", encoding="utf-8") as f:
                f.write("=== OUTPUT ===\n")
                f.write("=== output_text (trimmed) ===\n")
                f.write(output_text)
                f.write("\n\n=== full_text (includes prompt) ===\n")
                f.write(full_text)
            logger.info(f"Saved Qwen raw output to: {debug_output_path}")
        except Exception as e:
            logger.warning(f"Failed to write Qwen debug output: {e}")
            print(f"[QWEN DEBUG] Failed to write debug output: {e}")

    # 6. 解析结果
    parsed = JSONParser.robust_json_loads(output_text)
    
    # 7. 生成 Mask
    mask = np.zeros(len(series_ds), dtype=int)
    anomalies = []
    
    if parsed and "detected_anomalies" in parsed:
        for item in parsed["detected_anomalies"]:
            start, end = item.get("interval", [0, 0])
            start = max(0, min(start, len(series_ds) - 1))
            end = max(start, min(end, len(series_ds)))
            mask[start:end] = 1
            anomalies.append(item)
            
    # 8. 映射回原始长度
    if len(mask) != len(data):
        full_mask = np.zeros(len(data), dtype=int)
        for item in anomalies:
            start, end = item.get("interval", [0, 0])
            if hasattr(position_index, '__getitem__'):
                real_start = position_index[min(start, len(position_index) - 1)]
                real_end = position_index[min(end - 1, len(position_index) - 1)]
                full_mask[real_start:real_end + 1] = 1
        
        return full_mask, anomalies, position_index

    return mask, anomalies, position_index
