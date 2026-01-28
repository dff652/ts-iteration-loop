#!/usr/bin/env python3
"""
数据修复工具
用于批量修复旧数据文件的命名规范和元数据

功能：
1. 扫描目录中的旧格式文件
2. 尝试从文件名/内容中提取元数据
3. 重命名为标准格式: {PointID}_{StartTime}_{EndTime}_{Algorithm}.csv

使用方法：
    python repair_data.py --source /path/to/data --dry-run  # 预览
    python repair_data.py --source /path/to/data            # 执行
"""

import os
import re
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple


def parse_old_filename(filename: str) -> Optional[Dict]:
    """
    尝试解析旧格式文件名
    
    支持的格式：
    - global_chatts_m4_0.1_1200_ZHLH_4C_1216_AI_20405E.PV_20230101_trend_resid.csv
    - {task}_{method}_{downsampler}_{ratio}_{points}_{column}_{date}_{components}.csv
    """
    # 格式1: global_chatts_m4_0.1_1200_{path}_{point}_{date}_{components}.csv
    pattern1 = r'^(\w+)_(\w+)_(\w+)_[\d.]+_\d+_(.+?)_(\d{8})_(.+)\.csv$'
    match = re.match(pattern1, filename)
    if match:
        return {
            'task_name': match.group(1),
            'method': match.group(2),
            'point_id': match.group(4),
            'date': match.group(5),
            'components': match.group(6),
            'original': filename
        }
    
    # 格式2: {Project}_{Date}_{Point}.csv (如 LHS2_20250322_20250325_H2S.csv)
    pattern2 = r'^(\w+)_(\d{8})_(\d{8})_(.+)\.csv$'
    match = re.match(pattern2, filename)
    if match:
        return {
            'project': match.group(1),
            'start_date': match.group(2),
            'end_date': match.group(3),
            'point_id': match.group(4),
            'original': filename
        }
    
    return None


def extract_time_range_from_csv(filepath: str) -> Optional[Tuple[datetime, datetime]]:
    """
    从 CSV 文件内容中提取时间范围
    """
    try:
        df = pd.read_csv(filepath, nrows=5)
        
        # 查找时间列
        time_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].iloc[0])
                    time_col = col
                    break
                except:
                    continue
        
        if time_col is None:
            return None
        
        # 读取首尾时间
        df_full = pd.read_csv(filepath, usecols=[time_col])
        times = pd.to_datetime(df_full[time_col])
        return times.min(), times.max()
        
    except Exception as e:
        print(f"  Warning: Failed to extract time range: {e}")
        return None


def generate_new_filename(metadata: Dict, time_range: Optional[Tuple[datetime, datetime]], algorithm: str = 'unknown') -> Optional[str]:
    """
    生成标准化文件名
    """
    point_id = metadata.get('point_id', '')
    if not point_id:
        return None
    
    # 清理测点名
    safe_point_id = point_id.replace('/', '_').replace('\\', '_')
    
    # 确定时间范围
    if time_range:
        start_str = time_range[0].strftime('%Y%m%d_%H%M%S')
        end_str = time_range[1].strftime('%Y%m%d_%H%M%S')
    elif 'start_date' in metadata and 'end_date' in metadata:
        start_str = f"{metadata['start_date']}_000000"
        end_str = f"{metadata['end_date']}_235959"
    elif 'date' in metadata:
        start_str = f"{metadata['date']}_000000"
        end_str = f"{metadata['date']}_235959"
    else:
        return None
    
    # 确定算法
    algo = metadata.get('method', algorithm)
    
    return f"{safe_point_id}_{start_str}_{end_str}_{algo}.csv"


def repair_directory(source_dir: str, dry_run: bool = True, target_dir: str = None):
    """
    修复目录中的文件
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir) if target_dir else source_path
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"数据修复工具 {'(预览模式)' if dry_run else '(执行模式)'}")
    print(f"{'='*60}")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_path}")
    print()
    
    csv_files = list(source_path.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个 CSV 文件\n")
    
    repaired = 0
    skipped = 0
    failed = 0
    
    for csv_file in csv_files:
        filename = csv_file.name
        print(f"处理: {filename}")
        
        # 检查是否已经是标准格式
        std_pattern = r'^.+_\d{8}_\d{6}_\d{8}_\d{6}_\w+\.csv$'
        if re.match(std_pattern, filename):
            print(f"  [跳过] 已是标准格式")
            skipped += 1
            continue
        
        # 解析旧格式
        metadata = parse_old_filename(filename)
        if not metadata:
            print(f"  [失败] 无法解析文件名格式")
            failed += 1
            continue
        
        # 提取时间范围
        time_range = extract_time_range_from_csv(str(csv_file))
        
        # 生成新文件名
        new_filename = generate_new_filename(metadata, time_range)
        if not new_filename:
            print(f"  [失败] 无法生成标准文件名")
            failed += 1
            continue
        
        print(f"  -> {new_filename}")
        
        if not dry_run:
            try:
                new_path = target_path / new_filename
                if source_path == target_path:
                    csv_file.rename(new_path)
                else:
                    import shutil
                    target_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(csv_file, new_path)
                print(f"  [成功] 已重命名")
                repaired += 1
            except Exception as e:
                print(f"  [错误] {e}")
                failed += 1
        else:
            repaired += 1
    
    print(f"\n{'='*60}")
    print(f"统计: 修复 {repaired}, 跳过 {skipped}, 失败 {failed}")
    print(f"{'='*60}")
    
    if dry_run:
        print("\n提示: 使用 --execute 参数执行实际修复")


def main():
    parser = argparse.ArgumentParser(description='数据修复工具 - 批量修复旧数据文件命名')
    parser.add_argument('--source', '-s', required=True, help='源数据目录')
    parser.add_argument('--target', '-t', help='目标目录（默认同源目录）')
    parser.add_argument('--execute', action='store_true', help='执行实际修复（默认为预览模式）')
    
    args = parser.parse_args()
    
    repair_directory(
        source_dir=args.source,
        target_dir=args.target,
        dry_run=not args.execute
    )


if __name__ == '__main__':
    main()
