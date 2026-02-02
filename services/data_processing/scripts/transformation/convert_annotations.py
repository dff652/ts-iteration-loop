#!/usr/bin/env python3
"""
将标注JSON文件从当前格式转换为对话格式
"""
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 定义异常类型映射
ANOMALY_TYPE_MAPPING = {
    "上行尖峰": "竖直尖峰",
    "下行尖峰": "竖直尖峰",
    "双向尖峰": "竖直尖峰",
    "点异常": "竖直尖峰",
    "阶跃": "阶跃突变",
    "突然漂移": "阶跃突变",
    "缓慢漂移": "模式变化",
    "模式突变": "模式变化",
    "异常平台": "异常平台"
}

def extract_point_name(annotation_filename):
    """从标注文件名中提取点位名称
    例如: annotations_数据集zhlh_100_AC6403B.PV.json -> AC6403B.PV
    如果无法匹配特定模式，则返回去除扩展名的文件名
    """
    # 移除 ".json" 后缀
    basename = os.path.splitext(annotation_filename)[0]
    
    # 尝试匹配特定前缀 "annotations_数据集zhlh_100_"
    match = re.search(r'annotations_数据集zhlh_100_(.+)$', basename)
    if match:
        return match.group(1)
    
    # 如果没有特定前缀，移除可能存在的 "annotations_" 前缀
    if basename.startswith("annotations_"):
        return basename.replace("annotations_", "", 1)
        
    return basename

def extract_point_id(name):
    """Extract point id like ZHLH_XXX.PV from a filename or path."""
    if not name:
        return None
    candidates = []
    for match in re.findall(r'([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9]+)', name):
        lower = match.lower()
        if lower.endswith((".csv", ".json", ".jpg", ".png")):
            continue
        if not re.search(r'[A-Za-z]', match):
            continue
        candidates.append(match)
    if not candidates:
        return None
    candidates.sort(key=lambda s: (s.count("_"), len(s)), reverse=True)
    return candidates[0]

def extract_suffix_hint(name):
    """Extract suffix hint like trend_resid for fuzzy match."""
    if not name:
        return None
    for key in ["trend_resid", "resid", "trend"]:
        if key in name:
            return key
    return None

def find_latest_auto_file(output_dir, format_type):
    if not output_dir or not os.path.exists(output_dir):
        return None
    pattern = re.compile(rf"^{re.escape(format_type)}(?:_converted)?_(?:n)?\\d+_\\d{{8}}\\.json$")
    candidates = []
    for fname in os.listdir(output_dir):
        if pattern.match(fname):
            candidates.append(fname)
    if not candidates:
        return None
    candidates.sort(
        key=lambda fn: os.path.getmtime(os.path.join(output_dir, fn)),
        reverse=True
    )
    return os.path.join(output_dir, candidates[0])

def fuzzy_find_file(dir_path, point_id=None, suffix_hint=None, exts=None):
    """Fuzzy find a file by point id and optional suffix hint."""
    if not dir_path or not os.path.exists(dir_path):
        return None
    exts = exts or []
    matches = []
    for fname in os.listdir(dir_path):
        if exts and not any(fname.lower().endswith(ext) for ext in exts):
            continue
        if point_id and point_id not in fname:
            continue
        if suffix_hint and suffix_hint not in fname:
            continue
        matches.append(fname)
    if not matches and point_id:
        for fname in os.listdir(dir_path):
            if exts and not any(fname.lower().endswith(ext) for ext in exts):
                continue
            if point_id and point_id not in fname:
                continue
            matches.append(fname)
    if not matches:
        return None
    def score(fn):
        s = 0
        lower = fn.lower()
        if point_id:
            if fn.startswith(point_id):
                s += 100
            if f"_{point_id}" in fn:
                s += 60
            if point_id in fn:
                s += 30
        if suffix_hint and suffix_hint in fn:
            s += 10
        if lower.startswith("_"):
            s -= 3
        if "qwen" in lower:
            s -= 5
        if "chatts" in lower:
            s -= 5
        s -= len(fn) / 200.0
        return s

    matches.sort(
        key=lambda fn: (score(fn), os.path.getmtime(os.path.join(dir_path, fn))),
        reverse=True
    )
    return os.path.join(dir_path, matches[0])

def plot_csv_to_image(csv_path, output_image_path):
    """Generate a plot from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Assuming second column is the value
        value_col = df.columns[1]
        
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df[value_col], label=value_col, linewidth=1)
        plt.title(f"Time Series: {Path(csv_path).stem}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=100)
        plt.close()
        return True
    except Exception as e:
        print(f"⚠️ Plotting failed for {csv_path}: {e}")
        return False

def read_csv_values(csv_path):
    """Read values from CSV file for ChatTS format"""
    try:
        df = pd.read_csv(csv_path)
        # Assuming second column is the value (feature), similar to plot logic
        # If multiple columns, we might need more logic. 
        # For now, take the first non-index column or specific strategy.
        # plot_csv_to_image uses df.columns[1]
        if len(df.columns) > 1:
            value_col = df.columns[1]
            values = df[value_col].tolist()
            # Handle NaN
            values = [0.0 if pd.isna(x) else x for x in values]
            return [values] # Return as shape [1, N] matches [[v1, v2, ...]]
        return []
    except Exception as e:
        print(f"⚠️ Reading CSV failed for {csv_path}: {e}")
        return []

def find_csv_file(point_name, image_dir, csv_filename=None, csv_src_dir=None):
    """Find CSV file for a given point in image_dir or csv_src_dir"""
    possible_names = []
    if csv_filename:
        possible_names.append(csv_filename)

    if ".csv" in point_name:
        possible_names.append(point_name)
    else:
        possible_names.append(f"{point_name}.csv")

    # Direct match in image_dir
    for name in possible_names:
        csv_path = os.path.join(image_dir, name)
        if os.path.exists(csv_path):
            return csv_path

    # Direct match in csv_src_dir
    if csv_src_dir:
        for name in possible_names:
            csv_path = os.path.join(csv_src_dir, name)
            if os.path.exists(csv_path):
                return csv_path

    # Try searching by stem
    stem = point_name.replace(".csv", "")
    for f in os.listdir(image_dir):
        if f.endswith(".csv") and stem in f:
            return os.path.join(image_dir, f)

    if csv_src_dir and os.path.exists(csv_src_dir):
        for f in os.listdir(csv_src_dir):
            if f.endswith(".csv") and stem in f:
                return os.path.join(csv_src_dir, f)

    point_id = extract_point_id(csv_filename or point_name)
    suffix_hint = extract_suffix_hint(csv_filename or point_name)
    csv_path = fuzzy_find_file(image_dir, point_id=point_id, suffix_hint=suffix_hint, exts=[".csv"])
    if csv_path:
        return csv_path
    if csv_src_dir:
        csv_path = fuzzy_find_file(csv_src_dir, point_id=point_id, suffix_hint=suffix_hint, exts=[".csv"])
        if csv_path:
            return csv_path

    return None

def find_image_file(point_name, image_dir, csv_filename=None, csv_src_dir=None):
    """根据点位名称查找对应的图片文件
    1. 尝试标准格式: zhlh_100_{point_name}.jpg
    2. 尝试直接匹配: {point_name}.jpg / .png
    3. 尝试移除 .csv 后缀匹配
    4. 如果找不到图片但有CSV(在 image_dir 或 csv_src_dir)，尝试生成图片
    """
    candidates = [
        f"zhlh_100_{point_name}.jpg",
        f"{point_name}.jpg",
        f"{point_name}.png",
        f"{point_name}"
    ]
    
    if ".csv" in point_name:
        stem = point_name.replace(".csv", "")
        candidates.extend([
            f"{stem}.jpg",
            f"zhlh_100_{stem}.jpg"
        ])

    # Check existing images in image_dir
    for fname in candidates:
        image_path = os.path.join(image_dir, fname)
        if os.path.exists(image_path) and (fname.endswith('.jpg') or fname.endswith('.png')):
            return image_path

    point_id = extract_point_id(csv_filename or point_name)
    image_path = fuzzy_find_file(image_dir, point_id=point_id, suffix_hint=None, exts=[".jpg", ".png"])
    if image_path:
        return image_path
            
    # Try to generate if csv available
    possible_csvs = []
    if csv_filename:
        possible_csvs.append(csv_filename)
        
    if ".csv" in point_name:
         possible_csvs.append(point_name)
    else:
         possible_csvs.append(f"{point_name}.csv")
         
    for csv_name in possible_csvs:
        # 1. Look in image_dir
        csv_path = os.path.join(image_dir, csv_name)
        
        # 2. Look in csv_src_dir if fallback needed
        if not os.path.exists(csv_path) and csv_src_dir and os.path.exists(csv_src_dir):
             fallback_path = os.path.join(csv_src_dir, csv_name)
             if os.path.exists(fallback_path):
                 print(f"ℹ️  Found CSV in source dir: {fallback_path}")
                 csv_path = fallback_path
        
        if os.path.exists(csv_path):
            # Generate image in image_dir (always target the image_dir)
            stem = csv_name.replace(".csv", "")
            target_img_name = f"{stem}.jpg"
            target_img_path = os.path.join(image_dir, target_img_name)
            
            print(f"🔄 Generating image from CSV: {csv_name} -> {target_img_name}")
            if plot_csv_to_image(csv_path, target_img_path):
                return target_img_path

    if point_id:
        suffix_hint = extract_suffix_hint(csv_filename or point_name)
        csv_path = fuzzy_find_file(image_dir, point_id=point_id, suffix_hint=suffix_hint, exts=[".csv"])
        if not csv_path and csv_src_dir:
            csv_path = fuzzy_find_file(csv_src_dir, point_id=point_id, suffix_hint=suffix_hint, exts=[".csv"])
        if csv_path:
            stem = Path(csv_path).stem
            target_img_name = f"{stem}.jpg"
            target_img_path = os.path.join(image_dir, target_img_name)
            print(f"🔄 Generating image from CSV: {os.path.basename(csv_path)} -> {target_img_name}")
            if plot_csv_to_image(csv_path, target_img_path):
                return target_img_path
    
    return None

def map_anomaly_type(original_type):
    """将原始异常类型映射到目标类型"""
    return ANOMALY_TYPE_MAPPING.get(original_type, "模式变化")

def generate_reason(label_text, interval):
    """根据标签类型和区间生成原因说明"""
    start, end = interval
    length = end - start + 1
    
    # 根据实际的标签类型生成更精确的描述
    if label_text == "上行尖峰":
        if start == end:
            return f"索引位置{start}处数据出现向上尖峰,数值急剧上升后迅速回落"
        else:
            return f"索引区间[{start},{end}]内数据出现向上尖峰,数值急剧上升后迅速回落"
    
    elif label_text == "下行尖峰":
        if start == end:
            return f"索引位置{start}处数据出现向下尖峰,数值急剧下降后迅速回升"
        else:
            return f"索引区间[{start},{end}]内数据出现向下尖峰,数值急剧下降后迅速回升"
    
    elif label_text == "双向尖峰":
        if start == end:
            return f"索引位置{start}处数据出现双向尖峰,数值先急剧变化后反向变化"
        else:
            return f"索引区间[{start},{end}]内数据出现双向尖峰,数值呈现剧烈震荡"
    
    elif label_text == "阶跃上升":
        return f"索引区间[{start},{end}]内数据发生阶跃上升,数值快速跃迁到更高水平"
    
    elif label_text == "阶跃下降":
        return f"索引区间[{start},{end}]内数据发生阶跃下降,数值快速跃迁到更低水平"
    
    elif label_text == "突然漂移":
        return f"索引区间[{start},{end}]内数据发生突然漂移,基线水平发生突变"
    
    elif label_text == "缓慢漂移":
        return f"索引区间[{start},{end}]内数据发生缓慢漂移,基线水平逐渐偏移"
    
    elif label_text == "点异常":
        if start == end:
            return f"索引位置{start}处存在孤立异常点,与周围数据明显不符"
        else:
            return f"索引区间[{start},{end}]内存在多个孤立异常点"
    
    elif label_text == "区间异常":
        return f"索引区间[{start},{end}]内数据整体异常,与正常模式存在显著差异"
    
    elif label_text == "震荡区间":
        return f"索引区间[{start},{end}]内数据呈现异常震荡,波动幅度明显增大"
    
    elif label_text == "上下文异常":
        return f"索引区间[{start},{end}]内数据与前后上下文不一致,存在模式突变"
    
    # 兜底描述（不应该到这里）
    else:
        if start == end:
            return f"索引位置{start}处数据存在异常"
        else:
            return f"索引区间[{start},{end}]内数据存在异常"

def convert_overall_attribute_to_chinese(overall_attr):
    """将overall_attribute转换为中文描述"""
    # 定义映射关系
    mappings = {
        "frequency": {
            "high_freq": "高频",
            "low_freq": "低频"
        },
        "noise": {
            "noisy": "高噪声",
            "clean": "中等噪声",
            "almost_no_noise": "低噪声",
            "label_1766469170459": "无噪声"
        },
        "seasonal": {
            "has_periodic": "有周期性",
            "no_periodic": "无周期性",
            "label_1766475567943": "局部有周期性"
        },
        "trend": {
            "increase": "上升趋势",
            "decrease": "下降趋势",
            "stable": "趋势稳定",
            "multiple": "多段式趋势",
            "label_1766469189296": "无明显趋势"
        }
    }
    
    descriptions = []
    for key, value in overall_attr.items():
        if value and key in mappings and value in mappings[key]:
            descriptions.append(mappings[key][value])
    
    return "、".join(descriptions) if descriptions else ""

def convert_annotation_to_conversation(annotation_data, image_path):
    """将标注数据转换为对话格式"""
    
    # 生成用户提示词 - 使用实际的标签类型
    user_prompt = """<image>
你是一位时间序列异常检测专家。请分析图中的时间序列数据,识别异常区间。

异常类型：
1. 尖峰类：上行尖峰、下行尖峰、双向尖峰
2. 阶跃类：阶跃上升、阶跃下降
3. 漂移类：突然漂移、缓慢漂移
4. 特殊异常段：点异常、区间异常、震荡区间、上下文异常

请基于全局信号特征识别异常。输出必须是标准JSON格式：

{"status":"success","detected_anomalies":[{"interval":[start,end],"type":"类型","reason":"原因"}]}

若无异常：
{"status":"success","detected_anomalies":[]}

请精确标注异常区间的起止索引。"""
    
    # 生成助手响应（包含所有异常检测结果和overall_attribute）
    detected_anomalies = []
    
    # 处理每个标注
    for annotation in annotation_data.get("annotations", []):
        label_text = annotation["label"]["text"]
        # 直接使用原始标签文本作为type，不再映射
        anomaly_type = label_text
        
        # 处理每个segment
        for segment in annotation.get("segments", []):
            start = segment["start"]
            end = segment["end"]
            reason = generate_reason(label_text, [start, end])
            
            anomaly = {
                "interval": [start, end],
                "type": anomaly_type,
                "reason": reason
            }
            detected_anomalies.append(anomaly)
    
    # 特殊处理：无异常的情况（没有annotations或annotations为空）
    if not detected_anomalies:
        # 无异常时，输出格式：{"status":"success","detected_anomalies":[{"type":"无异常","reason":"该数据不做异常检测"}]}
        no_anomaly_result = {
            "status": "success",
            "detected_anomalies": [
                {
                    "type": "无异常",
                    "reason": "该数据不做异常检测"
                }
            ]
        }
        assistant_value = json.dumps(no_anomaly_result, ensure_ascii=False)
        
        # 如果有overall_attribute，添加全局属性
        if "overall_attribute" in annotation_data:
            overall_attr = annotation_data["overall_attribute"]
            filtered_attr = {k: v for k, v in overall_attr.items() if v}
            if filtered_attr:
                chinese_desc = convert_overall_attribute_to_chinese(filtered_attr)
                if chinese_desc:
                    overall_result = {
                        "status": "success",
                        "detected_anomalies": [
                            {
                                "type": "全局属性",
                                "reason": chinese_desc
                            }
                        ]
                    }
                    assistant_value = assistant_value + "," + json.dumps(overall_result, ensure_ascii=False)
    else:
        # 有异常的情况：为每个异常创建单独的JSON对象，然后用逗号连接
        individual_results = []
        for anomaly in detected_anomalies:
            result = {
                "status": "success",
                "detected_anomalies": [anomaly]
            }
            individual_results.append(json.dumps(result, ensure_ascii=False))
        
        assistant_value = ",".join(individual_results)
        
        # 添加overall_attribute到最后（转换为中文格式）
        if "overall_attribute" in annotation_data:
            overall_attr = annotation_data["overall_attribute"]
            filtered_attr = {k: v for k, v in overall_attr.items() if v}
            if filtered_attr:
                chinese_desc = convert_overall_attribute_to_chinese(filtered_attr)
                if chinese_desc:
                    overall_result = {
                        "status": "success",
                        "detected_anomalies": [
                            {
                                "type": "全局属性",
                                "reason": chinese_desc
                            }
                        ]
                    }
                    assistant_value = assistant_value + "," + json.dumps(overall_result, ensure_ascii=False)
    
    # 构建对话格式
    conversation = {
        "image": image_path,
        "conversations": [
            {
                "from": "user",
                "value": user_prompt
            },
            {
                "from": "assistant",
                "value": assistant_value
            }
        ]
    }
    
    return conversation

def convert_to_chatts_format(conversation, csv_path, identifier=None):
    """Convert ShareGPT format to ChatTS format (Alpaca-like with timeseries content)"""
    # content from csv
    target_values = []
    if csv_path:
        target_values = read_csv_values(csv_path)
    
    conversations = conversation.get("conversations", [])
    
    user_text = ""
    assistant_text = ""
    
    for conv in conversations:
        if conv["from"] == "user":
            user_text = conv["value"]
        elif conv["from"] == "assistant":
            assistant_text = conv["value"]
            
    return {
        "target": target_values, # List of lists
        "input": user_text,
        "output": assistant_text,
        "start": ["2023-01-01 00:00:00"], # Dummy start time if not available, matching reference
        "id": identifier or "unknown"
    }

def convert_annotations(input_dir, output_file, image_dir, filename=None, format_type="qwen", csv_src_dir=None):
    """
    转换标注文件，支持单个文件或批量转换。
    :param input_dir: 包含标注JSON文件的输入目录。
    :param output_file: 输出JSON文件的路径。
    :param image_dir: 包含对应图片文件的目录。
    :param filename: 如果指定，则只转换此单个文件；否则批量转换input_dir中的所有文件。
    :param format_type: 输出格式 ("qwen" 或 "chatts")。
    :param csv_src_dir: 查找CSV文件的备用目录。
    """
    
    all_conversations = []
    success_count = 0
    failed_files = []

    if filename: # Single file conversion
        annotation_path = os.path.join(input_dir, filename)
        if not os.path.exists(annotation_path):
            print(f"❌ 文件不存在: {annotation_path}")
            return None

        print(f"[{1}/{1}] 处理: {filename}", end=" ... ")
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
            
            point_name = extract_point_name(filename)
            csv_filename = annotation_data.get("filename")

            csv_path = find_csv_file(point_name, image_dir, csv_filename, csv_src_dir)
            image_path = find_image_file(point_name, image_dir, csv_filename, csv_src_dir)

            if not image_path and csv_filename:
                image_path = find_image_file(csv_filename, image_dir, csv_filename, csv_src_dir)
                if not point_name:
                    point_name = csv_filename

            if format_type == "qwen" and not image_path:
                print("❌ 找不到对应图片/CSV")
                msg = f"缺失源文件 (点位: {point_name})。请检查 {image_dir} 或 {csv_src_dir}。如已丢失，请前往[数据获取]模块重新采集。"
                failed_files.append((filename, msg))
                return all_conversations
            if format_type == "chatts" and not csv_path:
                 print("⚠️ 找不到对应CSV (将使用空数据)")
                 print(f"  建议前往[数据获取]模块重新采集点位: {point_name}")
            
            conversation = convert_annotation_to_conversation(annotation_data, image_path or "")
            
            if format_type == "chatts":
                final_data = convert_to_chatts_format(conversation, csv_path, point_name or filename)
            else:
                final_data = conversation
            
            all_conversations.append(final_data)
            success_count += 1
            print("✅")

        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            failed_files.append((filename, str(e)))

    else: # Batch conversion
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        total_files = len(json_files)
        
        print(f"🔄 开始批量转换，共 {total_files} 个文件...")
        print("=" * 80)
        
        for idx, current_filename in enumerate(json_files, 1):
            annotation_path = os.path.join(input_dir, current_filename)
            
            print(f"[{idx}/{total_files}] 处理: {current_filename}", end=" ... ")
            
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                
                point_name = extract_point_name(current_filename)
                csv_filename = annotation_data.get("filename")
                
                csv_path = find_csv_file(point_name, image_dir, csv_filename, csv_src_dir)
                image_path = find_image_file(point_name, image_dir, csv_filename, csv_src_dir)
                
                if not image_path:
                    if csv_filename:
                        image_path = find_image_file(csv_filename, image_dir, csv_filename, csv_src_dir)
                
                if format_type == "qwen" and not image_path:
                    print("❌ 找不到对应图片/CSV")
                    msg = f"缺失源文件 (点位: {point_name})。请检查 {image_dir} 或 {csv_src_dir}。如已丢失，请前往[数据获取]模块重新采集。"
                    failed_files.append((current_filename, msg))
                    continue
                if format_type == "chatts" and not csv_path:
                    print("⚠️ 找不到对应CSV (将使用空数据)")
                    print(f"  建议前往[数据获取]模块重新采集点位: {point_name}")

                conversation = convert_annotation_to_conversation(annotation_data, image_path or "")
                
                if format_type == "chatts":
                    final_data = convert_to_chatts_format(conversation, csv_path, point_name or current_filename)
                else:
                    final_data = conversation
                    
                all_conversations.append(final_data)
                
                success_count += 1
                print("✅")
                
            except Exception as e:
                print(f"❌ 错误: {str(e)}")
                failed_files.append((current_filename, str(e)))
    
    print("\n" + "=" * 80)
    print(f"📊 转换统计：成功 {success_count}/{len(json_files) if not filename else 1}")
    
    if failed_files:
        print(f"\n⚠️  失败的文件 ({len(failed_files)}):")
        for fname, reason in failed_files:
            print(f"  - {fname}: {reason}")
    
    # Auto-name output file if using default placeholder name
    final_output_file = output_file
    output_dir = output_file if os.path.isdir(output_file) else os.path.dirname(output_file)
    output_basename = os.path.basename(output_file)
    if output_basename == "converted_data.json" or os.path.isdir(output_file):
        date_tag = datetime.now().strftime("%Y%m%d")
        final_output_file = os.path.join(
            output_dir,
            f"{format_type}_converted_{len(all_conversations)}_{date_tag}.json"
        )

    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✨ 所有转换结果已保存到: {final_output_file}")
    print(f"📦 共 {len(all_conversations)} 条对话数据")
    
    return all_conversations

def convert_single_file(annotation_path, image_dir, output_dir=None, format_type="qwen"):
    """转换单个标注文件"""
    # This function is now deprecated/refactored into convert_annotations
    # Keeping it for backward compatibility if needed, but its logic is moved.
    raise NotImplementedError("convert_single_file has been refactored. Use convert_annotations directly.")

def batch_convert_all_files(annotation_dir, image_dir, output_file, format_type="qwen"):
    """批量转换所有标注文件并保存到一个JSON文件中"""
    # This function is now deprecated/refactored into convert_annotations
    # Keeping it for backward compatibility if needed, but its logic is moved.
    raise NotImplementedError("batch_convert_all_files has been refactored. Use convert_annotations directly.")

import argparse
import sys
import os
import json # Ensure json is imported

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Convert annotation JSON files to ChatTS conversation format")
    parser.add_argument("--input-dir", required=True, help="Input directory containing annotation JSON files")
    parser.add_argument("--image-dir", required=True, help="Directory containing corresponding images")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--file", help="Specific filename to convert (optional)")
    parser.add_argument("--format", default="qwen", choices=["qwen", "chatts"], help="Output format (default: qwen)")
    parser.add_argument("--csv-src", help="Fallback directory to look for CSVs if images missing (optional)")
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        sys.exit(1)
        
    if not os.path.exists(args.image_dir):
        print(f"❌ 图片目录不存在: {args.image_dir}")
        # sys.exit(1) # Allow proceeding if image_dir is missing, but conversion might fail for Qwen format
        # If image_dir is missing, we should still allow chatts format if csv_src is provided.
        # Or, if qwen, it will fail later.
        pass # Let convert_annotations handle the specific image/csv missing errors
        
    if args.file:
        # 单文件转换模式
        print(f"🔄 正在转换单个文件: {args.file}")
        annotation_path = os.path.join(args.input_dir, args.file)
        if not os.path.exists(annotation_path):
            print(f"❌ 文件不存在: {annotation_path}")
            sys.exit(1)
            
        placeholder_output = os.path.isdir(args.output) or os.path.basename(args.output) == "converted_data.json"
        output_dir = args.output if os.path.isdir(args.output) else os.path.dirname(args.output)
        base_output_path = None

        if placeholder_output:
            base_output_path = find_latest_auto_file(output_dir, args.format)
        else:
            base_output_path = args.output

        # 先读取已有输出，避免被转换过程覆盖
        all_conversations = []
        if base_output_path and os.path.exists(base_output_path):
            try:
                with open(base_output_path, 'r', encoding='utf-8') as f:
                    all_conversations = json.load(f)
            except:
                pass

        # 转换单个文件（使用临时输出避免覆盖目标文件）
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        result = convert_annotations(
            args.input_dir,
            tmp_path,
            args.image_dir,
            filename=args.file,
            format_type=args.format,
            csv_src_dir=args.csv_src,
        )
        try:
            os.unlink(tmp_path)
        except:
            pass
        conversation = result[0] if result else None

        if conversation:
            
            # Remove old entry for this image/file if exists
            # Determine key based on format
            if args.format == "chatts":
                # For ChatTS, we check "input" or "target" equality? Or maybe we can't easily dedup without ID?
                # The previous logic relied on image path.
                # We'll skip dedup for ChatTS or use input text as partial key.
                pass
            else:
                new_image_path = conversation.get("image")
                new_point_id = extract_point_id(new_image_path or args.file)
                if new_image_path or new_point_id:
                    def should_keep(c):
                        img = c.get("image", "")
                        if new_image_path and img == new_image_path:
                            return False
                        if new_point_id and extract_point_id(img) == new_point_id:
                            return False
                        return True
                    all_conversations = [c for c in all_conversations if should_keep(c)]
            
            all_conversations.append(conversation)

            # 仅写自动命名文件（或用户指定文件）
            if placeholder_output:
                date_tag = datetime.now().strftime("%Y%m%d")
                final_output_path = os.path.join(
                    output_dir,
                    f"{args.format}_converted_{len(all_conversations)}_{date_tag}.json"
                )
            else:
                final_output_path = args.output

            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_conversations, f, ensure_ascii=False, indent=2)
            print(f"✅ 单文件已更新至: {final_output_path}")
        else:
            print("❌ 单文件转换失败 (可能是找不到图片或CSV)")
            sys.exit(1)
            
    else:
        # 批量转换所有文件
        convert_annotations(
            args.input_dir,
            args.output,
            args.image_dir,
            format_type=args.format,
            csv_src_dir=args.csv_src,
        )

if __name__ == "__main__":
    main()
