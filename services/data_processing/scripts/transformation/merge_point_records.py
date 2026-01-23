#!/usr/bin/env python3
"""
合并JSON数据中同一点位的多条异常记录

将 gdsh, hbsn, whlj 格式的数据（一个点位多条记录）
转换为 zhlh 格式（一个点位一条记录，包含所有异常）
"""

import json
import os
import re
from collections import defaultdict
from typing import List, Dict, Any


def extract_point_name(image_path: str, prefix: str) -> str:
    """从图片路径中提取点位名称"""
    import os
    
    # 获取文件名（不含扩展名）
    basename = os.path.basename(image_path)
    point_name = os.path.splitext(basename)[0]
    
    # 根据不同的前缀处理不同的格式
    if prefix == 'gdsh':
        # 格式: gdsh_second_xxx.PV.jpg -> second_xxx.PV
        if 'gdsh_' in point_name:
            point_name = point_name.split('gdsh_', 1)[1]
        return point_name
    
    elif prefix == 'hbsn':
        # hbsn 文件中 image 路径实际使用 gdsh_ 前缀
        # 格式: gdsh_second_xxx.PV.jpg -> second_xxx.PV
        if 'gdsh_' in point_name:
            point_name = point_name.split('gdsh_', 1)[1]
        return point_name
    
    elif prefix == 'whlj_ljsj':
        # 格式: NB.LJSJ.xxx.PV.jpg -> NB.LJSJ.xxx.PV
        return point_name
    
    elif prefix == 'zhlh':
        # 格式: zhlh_100_xxx.PV.jpg -> 100_xxx.PV
        if 'zhlh_' in point_name:
            point_name = point_name.split('zhlh_', 1)[1]
        return point_name
    
    # 默认：直接返回文件名
    return point_name


def parse_anomalies_from_response(response: str) -> List[Dict]:
    """从assistant响应中解析异常列表"""
    try:
        # 尝试解析JSON
        data = json.loads(response)
        if isinstance(data, dict) and 'detected_anomalies' in data:
            return data['detected_anomalies']
    except json.JSONDecodeError:
        pass
    return []


def merge_anomalies(anomalies_list: List[List[Dict]]) -> List[Dict]:
    """合并多个异常列表，去重并排序"""
    all_anomalies = []
    seen_intervals = set()
    
    for anomalies in anomalies_list:
        for anomaly in anomalies:
            interval = anomaly.get('interval', [])
            if len(interval) == 2:
                interval_key = (interval[0], interval[1])
                if interval_key not in seen_intervals:
                    seen_intervals.add(interval_key)
                    all_anomalies.append(anomaly)
    
    # 按起始索引排序
    all_anomalies.sort(key=lambda x: x.get('interval', [0, 0])[0])
    
    # 过滤掉 "无异常" 类型的记录
    all_anomalies = [a for a in all_anomalies if a.get('type', '') != '无异常']
    
    return all_anomalies


def merge_json_file(input_path: str, output_path: str, prefix: str) -> Dict:
    """合并单个JSON文件中的同点位记录"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按点位分组
    point_records = defaultdict(list)
    point_images = {}  # 保存每个点位的第一个image路径
    point_user_prompt = {}  # 保存每个点位的用户提示
    
    for record in data:
        image_path = record.get('image', '')
        point_name = extract_point_name(image_path, prefix)
        
        if point_name:
            point_records[point_name].append(record)
            if point_name not in point_images:
                point_images[point_name] = image_path
                # 保存用户提示
                conversations = record.get('conversations', [])
                for conv in conversations:
                    if conv.get('from') == 'user':
                        point_user_prompt[point_name] = conv.get('value', '')
                        break
    
    # 合并同一点位的异常
    merged_data = []
    stats = {
        'original_records': len(data),
        'unique_points': len(point_records),
        'points_with_multiple_records': 0
    }
    
    for point_name, records in point_records.items():
        if len(records) > 1:
            stats['points_with_multiple_records'] += 1
        
        # 收集所有异常
        all_anomalies_lists = []
        for record in records:
            conversations = record.get('conversations', [])
            for conv in conversations:
                if conv.get('from') == 'assistant':
                    anomalies = parse_anomalies_from_response(conv.get('value', ''))
                    all_anomalies_lists.append(anomalies)
        
        # 合并异常
        merged_anomalies = merge_anomalies(all_anomalies_lists)
        
        # 构建合并后的响应
        merged_response = {
            "status": "success",
            "detected_anomalies": merged_anomalies
        }
        
        # 构建合并后的记录
        merged_record = {
            "image": point_images[point_name],
            "conversations": [
                {
                    "from": "user",
                    "value": point_user_prompt.get(point_name, '')
                },
                {
                    "from": "assistant", 
                    "value": json.dumps(merged_response, ensure_ascii=False)
                }
            ]
        }
        
        merged_data.append(merged_record)
    
    # 按点位名排序
    merged_data.sort(key=lambda x: x.get('image', ''))
    
    # 保存合并后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    stats['merged_records'] = len(merged_data)
    
    return stats


def main():
    base_dir = '/home/dff652/TS-anomaly-detection/ChatTS-Training/data/chatts_tune/json'
    
    # 需要处理的文件及其前缀
    files_to_process = [
        ('gdsh.json', 'gdsh'),
        ('hbsn.json', 'hbsn'),
        ('whlj_ljsj.json', 'whlj_ljsj'),
    ]
    
    print("=" * 60)
    print("JSON 数据合并工具")
    print("=" * 60)
    
    for filename, prefix in files_to_process:
        input_path = os.path.join(base_dir, filename)
        output_filename = filename.replace('.json', '_merged.json')
        output_path = os.path.join(base_dir, output_filename)
        
        if not os.path.exists(input_path):
            print(f"\n[跳过] 文件不存在: {filename}")
            continue
        
        print(f"\n[处理] {filename}")
        print("-" * 40)
        
        try:
            stats = merge_json_file(input_path, output_path, prefix)
            print(f"  原始记录数: {stats['original_records']}")
            print(f"  唯一点位数: {stats['unique_points']}")
            print(f"  多记录点位数: {stats['points_with_multiple_records']}")
            print(f"  合并后记录数: {stats['merged_records']}")
            print(f"  输出文件: {output_filename}")
        except Exception as e:
            print(f"  [错误] 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
