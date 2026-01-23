#!/usr/bin/env python3
"""
验证转换后的数据一致性
"""
import json
import os
import sys

def verify_data_consistency():
    """验证转换后数据与源数据的一致性"""
    
    # 读取转换后的数据
    converted_file = '/home/douff/converted_annotations/all_conversations.json'
    with open(converted_file, 'r', encoding='utf-8') as f:
        converted_data = json.load(f)
    
    # 创建点位到转换数据的映射
    converted_map = {}
    for item in converted_data:
        point_name = item['image'].split('zhlh_100_')[-1].replace('.jpg', '')
        converted_map[point_name] = item
    
    # 源文件目录
    source_dir = '/home/douff/ts/timeseries-annotator-v2/backend/annotations/douff'
    
    # 随机抽查几个文件进行验证
    import random
    test_files = random.sample([f for f in os.listdir(source_dir) if f.startswith('annotations_')], 5)
    
    print("=" * 80)
    print("数据一致性验证报告")
    print("=" * 80)
    
    all_passed = True
    
    for filename in test_files:
        print(f"\n检查文件: {filename}")
        
        # 读取源文件
        source_path = os.path.join(source_dir, filename)
        with open(source_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        # 提取点位名称
        import re
        match = re.search(r'annotations_数据集zhlh_100_(.+)\.json$', filename)
        if not match:
            print("  ❌ 无法提取点位名称")
            continue
        
        point_name = match.group(1)
        
        if point_name not in converted_map:
            print(f"  ❌ 点位 {point_name} 未在转换后的数据中找到")
            all_passed = False
            continue
        
        converted_item = converted_map[point_name]
        
        # 验证异常数量
        source_annotations = source_data.get('annotations', [])
        assistant_value = converted_item['conversations'][1]['value']
        
        # 计算转换后的异常数量（每个 "detected_anomalies" 出现次数，减去全局属性）
        anomaly_count = assistant_value.count('"detected_anomalies"')
        has_overall = '全局属性' in assistant_value
        if has_overall:
            anomaly_count -= 1  # 减去全局属性
        
        # 计算源数据的segment总数
        source_segment_count = sum(len(ann.get('segments', [])) for ann in source_annotations)
        
        print(f"  源文件异常segment数: {source_segment_count}")
        print(f"  转换后异常数: {anomaly_count}")
        
        if source_segment_count == anomaly_count:
            print("  ✅ 异常数量一致")
        else:
            print("  ⚠️  异常数量不一致")
            all_passed = False
        
        # 验证overall_attribute
        source_overall = source_data.get('overall_attribute', {})
        has_source_overall = any(source_overall.values())
        
        if has_source_overall and not has_overall:
            print("  ❌ 源文件有overall_attribute，但转换后未包含")
            all_passed = False
        elif has_source_overall and has_overall:
            print("  ✅ overall_attribute已转换")
        elif not has_source_overall:
            print("  ℹ️  源文件无overall_attribute")
        
        # 检查具体异常类型
        source_types = set()
        for ann in source_annotations:
            source_types.add(ann['label']['text'])
        
        converted_types = set()
        for ann in source_annotations:
            if ann['label']['text'] in assistant_value:
                converted_types.add(ann['label']['text'])
        
        if source_types == converted_types:
            print(f"  ✅ 异常类型一致: {source_types}")
        else:
            print(f"  ⚠️  异常类型可能有差异")
            print(f"     源: {source_types}")
            print(f"     转换: {converted_types}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有检查通过！")
    else:
        print("⚠️  发现一些不一致！")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    verify_data_consistency()
