#!/usr/bin/env python3
"""
将微调数据集中的样本拆分：每个样本只包含一个异常区域

原始样本格式：
{
    "input": "<ts><ts/>\n用户提示...",
    "output": "{\"status\": \"success\", \"detected_anomalies\": [{异常1}, {异常2}, ...]}",
    "timeseries": [[...]]
}

拆分后的样本格式：
{
    "input": "<ts><ts/>\n用户提示...",
    "output": "{\"status\": \"success\", \"detected_anomalies\": [{单个异常}]}",
    "timeseries": [[...]]
}

对于没有异常的样本（detected_anomalies为空），保持原样。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def split_sample_by_anomaly(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将包含多个异常的样本拆分为多个单异常样本
    
    Args:
        sample: 原始样本字典，包含 input, output, timeseries
        
    Returns:
        拆分后的样本列表
    """
    try:
        output_data = json.loads(sample['output'])
    except json.JSONDecodeError as e:
        print(f"警告: 无法解析 output JSON: {e}")
        return [sample]  # 无法解析则保持原样
    
    anomalies = output_data.get('detected_anomalies', [])
    
    # 如果没有异常或只有一个异常，保持原样
    if len(anomalies) <= 1:
        return [sample]
    
    # 拆分为多个样本，每个样本一个异常
    split_samples = []
    for anomaly in anomalies:
        new_output = {
            'status': 'success',
            'detected_anomalies': [anomaly]
        }
        
        new_sample = {
            'input': sample['input'],
            'output': json.dumps(new_output, ensure_ascii=False),
            'timeseries': sample['timeseries']
        }
        split_samples.append(new_sample)
    
    return split_samples


def process_jsonl_file(input_path: str, output_path: str) -> Dict[str, int]:
    """
    处理整个 JSONL 文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        
    Returns:
        统计信息字典
    """
    stats = {
        'original_samples': 0,
        'split_samples': 0,
        'samples_without_anomaly': 0,
        'samples_with_single_anomaly': 0,
        'samples_with_multiple_anomalies': 0,
        'total_anomalies': 0
    }
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行无法解析: {e}")
                continue
            
            stats['original_samples'] += 1
            
            # 统计异常数量
            try:
                output_data = json.loads(sample['output'])
                num_anomalies = len(output_data.get('detected_anomalies', []))
                stats['total_anomalies'] += num_anomalies
                
                if num_anomalies == 0:
                    stats['samples_without_anomaly'] += 1
                elif num_anomalies == 1:
                    stats['samples_with_single_anomaly'] += 1
                else:
                    stats['samples_with_multiple_anomalies'] += 1
            except:
                pass
            
            # 拆分样本
            split_samples = split_sample_by_anomaly(sample)
            
            for new_sample in split_samples:
                fout.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                stats['split_samples'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='将微调数据集中的多异常样本拆分为单异常样本')
    parser.add_argument('input_file', help='输入的 JSONL 文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（默认为输入文件名_split.jsonl）')
    parser.add_argument('--dry-run', action='store_true', help='仅显示统计信息，不写入文件')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_split{input_path.suffix}"
    
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print()
    
    if args.dry_run:
        # 仅统计模式
        print("=== 干运行模式（仅统计）===")
        stats = {
            'original_samples': 0,
            'samples_without_anomaly': 0,
            'samples_with_single_anomaly': 0,
            'samples_with_multiple_anomalies': 0,
            'total_anomalies': 0,
            'expected_split_samples': 0
        }
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    output_data = json.loads(sample['output'])
                    num_anomalies = len(output_data.get('detected_anomalies', []))
                    
                    stats['original_samples'] += 1
                    stats['total_anomalies'] += num_anomalies
                    
                    if num_anomalies == 0:
                        stats['samples_without_anomaly'] += 1
                        stats['expected_split_samples'] += 1
                    elif num_anomalies == 1:
                        stats['samples_with_single_anomaly'] += 1
                        stats['expected_split_samples'] += 1
                    else:
                        stats['samples_with_multiple_anomalies'] += 1
                        stats['expected_split_samples'] += num_anomalies
                except:
                    pass
        
        print(f"原始样本数: {stats['original_samples']}")
        print(f"  - 无异常样本: {stats['samples_without_anomaly']}")
        print(f"  - 单异常样本: {stats['samples_with_single_anomaly']}")
        print(f"  - 多异常样本: {stats['samples_with_multiple_anomalies']}")
        print(f"总异常区域数: {stats['total_anomalies']}")
        print(f"拆分后预计样本数: {stats['expected_split_samples']}")
    else:
        # 实际处理
        stats = process_jsonl_file(str(input_path), str(output_path))
        
        print("=== 处理完成 ===")
        print(f"原始样本数: {stats['original_samples']}")
        print(f"  - 无异常样本: {stats['samples_without_anomaly']}")
        print(f"  - 单异常样本: {stats['samples_with_single_anomaly']}")
        print(f"  - 多异常样本: {stats['samples_with_multiple_anomalies']}")
        print(f"总异常区域数: {stats['total_anomalies']}")
        print(f"拆分后样本数: {stats['split_samples']}")
    
    return 0


if __name__ == '__main__':
    exit(main())
