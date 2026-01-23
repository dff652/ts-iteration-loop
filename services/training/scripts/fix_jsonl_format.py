#!/usr/bin/env python3
"""
修复 JSONL 数据格式，使其符合 ChatTS 训练要求

问题：
1. input 字段缺少 <ts><ts/> 占位符
2. output 字段类型不是字符串

解决方案：
1. 在 input 开头添加 <ts><ts/>\n
2. 将 output 转换为 JSON 字符串
"""

import json
import os
import sys


def fix_jsonl_format(input_file: str, output_file: str) -> dict:
    """
    修复 JSONL 文件格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        
    Returns:
        统计信息字典
    """
    stats = {
        'total': 0,
        'fixed_ts_tag': 0,
        'fixed_output': 0,
        'skipped': 0,
    }
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                stats['total'] += 1
                
                # 1. 修复 input 字段：添加 <ts><ts/> 占位符
                input_text = data.get('input', '')
                if '<ts><ts/>' not in input_text:
                    # 在开头添加占位符
                    data['input'] = '<ts><ts/>\n' + input_text
                    stats['fixed_ts_tag'] += 1
                
                # 2. 修复 output 字段：转换为 JSON 字符串
                output = data.get('output', '')
                if not isinstance(output, str):
                    data['output'] = json.dumps(output, ensure_ascii=False)
                    stats['fixed_output'] += 1
                
                # 3. 验证 timeseries 字段
                ts = data.get('timeseries')
                if not ts or not isinstance(ts, list):
                    print(f"警告: 行 {i} 缺少有效的 timeseries 字段")
                    stats['skipped'] += 1
                    continue
                
                # 确保 timeseries 是嵌套列表格式 [[...]]
                if ts and not isinstance(ts[0], list):
                    data['timeseries'] = [ts]
                
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"错误: 行 {i} JSON 解析失败: {e}")
                stats['skipped'] += 1
                continue
    
    return stats


def main():
    # 默认文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = os.path.join(base_dir, 'data/chatts_tune/gdsh_hbsn_data.jsonl')
        output_file = os.path.join(base_dir, 'data/chatts_tune/gdsh_hbsn_data_chatts.jsonl')
    
    print("=" * 60)
    print("ChatTS 数据格式修复工具")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    stats = fix_jsonl_format(input_file, output_file)
    
    print("处理完成!")
    print(f"  总记录数: {stats['total']}")
    print(f"  添加 <ts><ts/> 标签: {stats['fixed_ts_tag']}")
    print(f"  修复 output 类型: {stats['fixed_output']}")
    print(f"  跳过无效记录: {stats['skipped']}")
    print(f"  输出文件: {output_file}")


if __name__ == '__main__':
    main()
