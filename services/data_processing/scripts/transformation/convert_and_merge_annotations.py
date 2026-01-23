#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：
1. 读取 douff 目录下的所有 JSON 文件
2. 调整数据格式：将全局属性信息从overall_attribute提取，添加到user的value中
3. 合并所有文件到一个 JSON
"""

import json
import os
from pathlib import Path


def parse_overall_attribute(overall_attr):
    """
    解析 overall_attribute 字典，生成文字描述
    """
    characteristics = []
    
    # 频率
    freq_map = {
        'low_freq': '低频',
        'mid_freq': '中频',
        'high_freq': '高频'
    }
    if overall_attr.get('frequency'):
        characteristics.append(freq_map.get(overall_attr['frequency'], '未知频率'))
    
    # 噪声
    noise_map = {
        'clean': '低噪声',
        'no_noise': '无噪声',
        'moderate_noise': '中等噪声',
        'noisy': '高噪声'
    }
    if overall_attr.get('noise'):
        characteristics.append(noise_map.get(overall_attr['noise'], '未知噪声'))
    
    # 周期性
    seasonal_map = {
        'periodic': '有周期性',
        'local_periodic': '局部有周期性',
        'no_periodic': '无周期性'
    }
    if overall_attr.get('seasonal'):
        characteristics.append(seasonal_map.get(overall_attr['seasonal'], '未知周期性'))
    
    return characteristics


def create_user_prompt(overall_attr):
    """
    根据全局属性创建新的user prompt
    """
    if not overall_attr:
        return "请分析图中的时间序列数据，并识别异常发生的区间"
    
    characteristics = parse_overall_attribute(overall_attr)
    
    if characteristics:
        description = '、'.join(characteristics)
        new_prompt = f"请分析图中的{description}的时间序列数据，并识别异常发生的区间"
    else:
        new_prompt = "请分析图中的时间序列数据，并识别异常发生的区间"
    
    return new_prompt


def convert_annotation_to_response(annotation):
    """
    将单个标注转换为响应格式
    """
    label_text = annotation['label']['text']
    responses = []
    
    for segment in annotation['segments']:
        start = segment['start']
        end = segment['end']
        
        # 构建reason
        if start == end:
            if label_text in ['点异常']:
                reason = f"索引位置{start}处存在孤立异常点,与周围数据明显不符"
            elif label_text in ['上行尖峰', '下行尖峰']:
                reason = f"索引位置{start}处数据出现{'向上' if label_text == '上行尖峰' else '向下'}尖峰,数值急剧{'上升' if label_text == '上行尖峰' else '下降'}后迅速回{'落' if label_text == '上行尖峰' else '升'}"
            else:
                reason = f"索引位置{start}处数据异常"
        else:
            if label_text == '点异常':
                reason = f"索引区间[{start},{end}]内存在多个孤立异常点"
            elif label_text in ['阶跃上升', '阶跃下降']:
                reason = f"索引区间[{start},{end}]内数据发生{label_text},数值快速跃迁到更{'高' if label_text == '阶跃上升' else '低'}水平"
            elif label_text in ['上行尖峰', '下行尖峰', '双向尖峰']:
                reason = f"索引区间[{start},{end}]内数据出现{'向上' if label_text == '上行尖峰' else ('向下' if label_text == '下行尖峰' else '双向')}尖峰,数值急剧变化后迅速回复"
            elif label_text in ['突然漂移', '缓慢漂移']:
                reason = f"索引区间[{start},{end}]内数据发生{label_text},基线水平{'突变' if label_text == '突然漂移' else '缓慢变化'}"
            elif label_text == '区间异常':
                reason = f"索引区间[{start},{end}]内数据整体异常,与正常模式存在显著差异"
            elif label_text == '震荡区间':
                reason = f"索引区间[{start},{end}]内数据呈现异常震荡,波动幅度明显增大"
            elif label_text == '上下文异常':
                reason = f"索引区间[{start},{end}]内数据与前后上下文不一致,存在模式突变"
            else:
                reason = f"索引区间[{start},{end}]内数据存在{label_text}"
        
        response = {
            "interval": [start, end],
            "type": label_text,
            "reason": reason
        }
        responses.append(response)
    
    return responses


def process_json_file(file_path):
    """
    处理单个JSON文件
    """
    print(f"处理文件: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取全局属性并生成user prompt
        overall_attr = data.get('overall_attribute', {})
        new_user_prompt = create_user_prompt(overall_attr)
        
        # 构建user message
        user_message = {
            "from": "user",
            "value": f"<image>\n你是一位时间序列异常检测专家。{new_user_prompt}。\n\n异常类型：\n1. 尖峰类：上行尖峰、下行尖峰、双向尖峰\n2. 阶跃类：阶跃上升、阶跃下降\n3. 漂移类：突然漂移、缓慢漂移\n4. 特殊异常段：点异常、区间异常、震荡区间、上下文异常\n\n请基于全局信号特征识别异常。输出必须是标准JSON格式：\n\n{{\"status\":\"success\",\"detected_anomalies\":[{{\"interval\":[start,end],\"type\":\"类型\",\"reason\":\"原因\"}}]}}\n\n若无异常：\n{{\"status\":\"success\",\"detected_anomalies\":[]}}\n\n请精确标注异常区间的起止索引。"
        }
        
        # 处理所有标注
        all_anomalies = []
        for annotation in data.get('annotations', []):
            anomalies = convert_annotation_to_response(annotation)
            all_anomalies.extend(anomalies)
        
        # 构建assistant responses
        assistant_responses = []
        for anomaly in all_anomalies:
            response = {
                "status": "success",
                "detected_anomalies": [anomaly]
            }
            assistant_responses.append(json.dumps(response, ensure_ascii=False))
        
        # 如果没有异常，添加无异常响应
        if not assistant_responses:
            assistant_responses.append(json.dumps({
                "status": "success",
                "detected_anomalies": [{
                    "type": "无异常",
                    "reason": "该数据不做异常检测"
                }]
            }, ensure_ascii=False))
        
        # 构建assistant message
        assistant_message = {
            "from": "assistant",
            "value": ','.join(assistant_responses)
        }
        
        # 构建图片路径
        # 从filename提取图片名
        filename = data.get('filename', '')
        if filename:
            # 替换.csv为.jpg
            image_name = filename.replace('.csv', '.jpg')
            image_path = f"/home/douff/数据标注/data/picture_data/{image_name}"
        else:
            image_path = ""
        
        return {
            "image": image_path,
            "conversations": [user_message, assistant_message]
        }
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 定义路径
    source_dir = Path('/home/douff/ts/timeseries-annotator-v2/backend/annotations/douff')
    output_file = Path('/home/douff/converted_annotations/merged_annotations.json')
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = list(source_dir.glob('*.json'))
    print(f"找到 {len(json_files)} 个JSON文件\n")
    
    # 处理所有文件
    all_conversations = []
    
    for json_file in sorted(json_files):
        result = process_json_file(json_file)
        if result:
            all_conversations.append(result)
    
    # 保存合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"共处理 {len(all_conversations)} 个文件")
    print(f"合并后的文件已保存到: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
