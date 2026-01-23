#!/usr/bin/env python3
"""
将标注JSON文件从当前格式转换为对话格式
"""
import json
import os
import re
from pathlib import Path

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
    """
    # 移除 "annotations_数据集zhlh_100_" 前缀和 ".json" 后缀
    match = re.search(r'annotations_数据集zhlh_100_(.+)\.json$', annotation_filename)
    if match:
        return match.group(1)
    return None

def find_image_file(point_name, image_dir):
    """根据点位名称查找对应的图片文件"""
    # 图片文件格式: zhlh_100_{point_name}.jpg
    image_filename = f"zhlh_100_{point_name}.jpg"
    image_path = os.path.join(image_dir, image_filename)
    
    if os.path.exists(image_path):
        return image_path
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

def convert_single_file(annotation_path, image_dir, output_dir=None):
    """转换单个标注文件"""
    
    # 读取标注文件
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)
    
    # 提取点位名称
    filename = os.path.basename(annotation_path)
    point_name = extract_point_name(filename)
    
    if not point_name:
        print(f"⚠️  无法从文件名提取点位名称: {filename}")
        return None
    
    # 查找对应的图片文件
    image_path = find_image_file(point_name, image_dir)
    
    if not image_path:
        print(f"⚠️  找不到对应的图片文件: {point_name}")
        return None
    
    # 转换格式
    conversation = convert_annotation_to_conversation(annotation_data, image_path)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"converted_{point_name}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 转换成功: {filename} -> {output_filename}")
    
    return conversation

def batch_convert_all_files(annotation_dir, image_dir, output_file):
    """批量转换所有标注文件并保存到一个JSON文件中"""
    
    all_conversations = []
    success_count = 0
    failed_files = []
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
    total_files = len(json_files)
    
    print(f"🔄 开始批量转换，共 {total_files} 个文件...")
    print("=" * 80)
    
    for idx, filename in enumerate(json_files, 1):
        annotation_path = os.path.join(annotation_dir, filename)
        
        # 显示进度
        print(f"[{idx}/{total_files}] 处理: {filename}", end=" ... ")
        
        try:
            # 读取标注文件
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
            
            # 提取点位名称
            point_name = extract_point_name(filename)
            
            if not point_name:
                print("❌ 无法提取点位名称")
                failed_files.append((filename, "无法提取点位名称"))
                continue
            
            # 查找对应的图片文件
            image_path = find_image_file(point_name, image_dir)
            
            if not image_path:
                print("❌ 找不到对应图片")
                failed_files.append((filename, "找不到对应图片"))
                continue
            
            # 转换格式
            conversation = convert_annotation_to_conversation(annotation_data, image_path)
            all_conversations.append(conversation)
            
            success_count += 1
            print("✅")
            
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            failed_files.append((filename, str(e)))
    
    # 保存所有转换结果到一个JSON文件
    print("\n" + "=" * 80)
    print(f"📊 转换统计：成功 {success_count}/{total_files}")
    
    if failed_files:
        print(f"\n⚠️  失败的文件 ({len(failed_files)}):")
        for filename, reason in failed_files:
            print(f"  - {filename}: {reason}")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✨ 所有转换结果已保存到: {output_file}")
    print(f"📦 共 {len(all_conversations)} 条对话数据")
    
    return all_conversations

def main():
    """主函数"""
    # 定义路径
    annotation_dir = "/home/douff/ts/timeseries-annotator-v2/backend/annotations/douff"
    image_dir = "/home/douff/数据标注/data/picture_data"
    output_file = "/home/douff/converted_annotations/all_conversations.json"
    
    # 批量转换所有文件
    batch_convert_all_files(annotation_dir, image_dir, output_file)

if __name__ == "__main__":
    main()
