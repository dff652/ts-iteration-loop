#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   compare_methods.py
@Time    :   2025/09/24
@Author  :   DouFengfeng
@Version :   1.0.0
@Contact :   ff.dou@cyber-insight.com
@License :   (C)Copyright 2019-2026, CyberInsight
@Desc    :   比较两种检测算法（如 stl_wavelet 与 adtk_hbos）的检测结果并可视化
"""

import os
import re
import argparse
import logging
from multiprocessing import Pool
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_result_file(data_dir: str, task_name: str, column: str, method: str) -> Optional[str]:
    """
    在 `data_dir/task_name` 下查找包含指定列与方法名的结果文件。

    Parameters:
        data_dir (str): 数据根目录
        task_name (str): 任务名（global/local）
        column (str): 点位列名
        method (str): 方法名关键字（如 'stl_wavelet'、'adtk_hbos'）

    Returns:
        Optional[str]: 匹配到的文件路径，未找到返回 None
    """
    # 1) 优先方法子目录: <data_dir>/<task_name>/<method>/
    method_dir = os.path.join(data_dir, task_name, method)
    short_col = _short_point_name(column)
    if os.path.exists(method_dir):
        for fname in os.listdir(method_dir):
            if not fname.endswith('.csv'):
                continue
            if column in fname or (short_col and short_col in fname):
                return os.path.join(method_dir, fname)
    # 2) 兼容旧结构: <data_dir>/<task_name>/ 下通过文件名包含 method + column 匹配
    dir_path = os.path.join(data_dir, task_name)
    if not os.path.exists(dir_path):
        return None
    for fname in os.listdir(dir_path):
        if not fname.endswith('.csv'):
            continue
        # 命名规则包含: method_downsampler_shortpoint_yyyymmdd.csv
        if method in fname and (column in fname or (short_col and short_col in fname)):
            return os.path.join(dir_path, fname)
    return None


def _strip_date_suffix(name: str) -> str:
    name = re.sub(r"_\d{8}(?:_\d{6})?_to_\d{8}(?:_\d{6})?$", "", name)
    name = re.sub(r"_\d{8}(?:_\d{6})?$", "", name)
    return name


def _short_point_name(raw: str) -> str:
    if not raw:
        return raw
    name = os.path.basename(str(raw))
    if name.lower().endswith(".csv"):
        name = name[:-4]
    name = _strip_date_suffix(name)

    match = re.search(r"(?:^|_)([A-Za-z]{1,3}_[A-Za-z0-9]+\.[A-Za-z0-9]+)$", name)
    if match:
        return match.group(1)

    parts = name.split("_")
    for i in range(len(parts) - 1, -1, -1):
        if "." in parts[i]:
            if i > 0:
                return f"{parts[i - 1]}_{parts[i]}"
            return parts[i]
    return name


def get_data_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not c.endswith('_mask') and not c.endswith('_mask_cluster') and c != 'outlier_mask']


def pick_mask_column(df: pd.DataFrame, task_name: str) -> Optional[str]:
    """
    优先返回 'global_mask'；其次返回与 task 相关的掩码；否则返回任意一个 *_mask。
    """
    candidates = []
    if 'global_mask' in df.columns:
        return 'global_mask'
    # 常见备选
    for key in [f'{task_name}_mask', 'outlier_mask']:
        if key in df.columns:
            return key
    # 兜底选择第一个 *_mask
    for c in df.columns:
        if c.endswith('_mask') and not c.endswith('_mask_cluster'):
            candidates.append(c)
    return candidates[0] if candidates else None


def plot_comparison(original: pd.Series,
                    mask_a: np.ndarray,
                    mask_b: np.ndarray,
                    label_a: str,
                    label_b: str,
                    save_dir: str,
                    filename_prefix: str) -> List[str]:
    os.makedirs(save_dir, exist_ok=True)
    paths = []

    data_len = len(original)
    x_axis = np.arange(data_len)
    max_y = original.max()
    min_y = original.min()
    height_step = (max_y - min_y) * 0.3 if max_y != min_y else 1.0

    # 原始数据 + 两算法掩码对比
    plt.figure(figsize=(20, 8))
    plt.plot(x_axis, original.values, label='Original Data', color='blue', linewidth=1)
    plt.fill_between(x_axis, min_y - height_step * 0.5, min_y + height_step * 0.5,
                     where=mask_a.astype(bool), color='red', alpha=0.5, label=label_a)
    plt.fill_between(x_axis, min_y + height_step * 1.2, min_y + height_step * 2.2,
                     where=mask_b.astype(bool), color='green', alpha=0.5, label=label_b)
    plt.title(f'{filename_prefix} - Methods Comparison on Original Data')
    plt.xlabel('Index')
    plt.ylabel(original.name)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(save_dir, f'{filename_prefix}_original_comparison.png')
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()
    paths.append(p1)

    # 异常数量对比
    plt.figure(figsize=(10, 6))
    methods = [label_a, label_b]
    counts = [int(mask_a.sum()), int(mask_b.sum())]
    x = np.arange(2)
    bars = plt.bar(x, counts, color=['red', 'green'], alpha=0.7)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2.0, count + max(counts) * 0.01 if max(counts) > 0 else 0.02,
                 f'{count}', ha='center', va='bottom')
    plt.xticks(x, methods)
    plt.ylabel('Anomaly Count')
    plt.title(f'{filename_prefix} - Anomaly Count Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(save_dir, f'{filename_prefix}_count_comparison.png')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    paths.append(p2)

    return paths


def process_point(column: str,
                  data_path: str,
                  fig_path: str,
                  task_name: str,
                  method_a: str,
                  method_b: str,
                  mode: str = 'downsample') -> List[str]:
    try:
        file_a = find_result_file(data_path, task_name, column, method_a)
        file_b = find_result_file(data_path, task_name, column, method_b)
        if not file_a or not file_b:
            logger.warning(f"{column}: 未找到对应文件 (A={file_a}, B={file_b})，跳过")
            return []

        df_a = pd.read_csv(file_a, index_col=0)
        df_b = pd.read_csv(file_b, index_col=0)

        # 选择原始数据列
        data_cols_a = get_data_columns(df_a)
        data_cols_b = get_data_columns(df_b)
        if not data_cols_a or not data_cols_b:
            logger.warning(f"{column}: 未找到原始数据列，跳过")
            return []
        base_col = data_cols_a[0]

        mask_col_a = pick_mask_column(df_a, task_name)
        mask_col_b = pick_mask_column(df_b, task_name)
        if mask_col_a is None or mask_col_b is None:
            logger.warning(f"{column}: 未找到掩码列，跳过 (A={mask_col_a}, B={mask_col_b})")
            return []

        if mode == 'raw':
            a_has_pos = 'orig_pos' in df_a.columns
            b_has_pos = 'orig_pos' in df_b.columns
            if not a_has_pos:
                # A 视为原始参考
                original_series = df_a[base_col]
                original_len = len(original_series)
                mask_a = df_a[mask_col_a].astype(int).values
                if b_has_pos:
                    mask_b = np.zeros(original_len, dtype=int)
                    pos_b = df_b['orig_pos'].astype(int).values
                    vals_b = df_b[mask_col_b].astype(int).values
                    valid = (pos_b >= 0) & (pos_b < original_len)
                    pos_b = pos_b[valid]
                    vals_b = vals_b[valid]
                    mask_b[pos_b] = vals_b
                else:
                    df_b_aligned = df_b.reindex(df_a.index)
                    mask_b = df_b_aligned[mask_col_b].fillna(0).astype(int).values
                save_dir = os.path.join(fig_path, task_name, 'methods_comparison_raw')
                filename_prefix = f'{column}_{method_a}_vs_{method_b}_raw'
                return plot_comparison(original_series, mask_a, mask_b, method_a, method_b, save_dir, filename_prefix)
            elif not b_has_pos:
                # B 视为原始参考
                original_series = df_b[base_col]
                original_len = len(original_series)
                mask_b = df_b[mask_col_b].astype(int).values
                if a_has_pos:
                    mask_a = np.zeros(original_len, dtype=int)
                    pos_a = df_a['orig_pos'].astype(int).values
                    vals_a = df_a[mask_col_a].astype(int).values
                    valid = (pos_a >= 0) & (pos_a < original_len)
                    pos_a = pos_a[valid]
                    vals_a = vals_a[valid]
                    mask_a[pos_a] = vals_a
                else:
                    df_a_aligned = df_a.reindex(df_b.index)
                    mask_a = df_a_aligned[mask_col_a].fillna(0).astype(int).values
                save_dir = os.path.join(fig_path, task_name, 'methods_comparison_raw')
                filename_prefix = f'{column}_{method_a}_vs_{method_b}_raw'
                return plot_comparison(original_series, mask_a, mask_b, method_a, method_b, save_dir, filename_prefix)
            else:
                logger.warning(f"{column}: 两个文件均为降采样保存（含 orig_pos），无法在 raw 模式绘制原始曲线，改为 downsample 模式")
                mode = 'downsample'

        # downsample 模式：按 A 的索引对齐 B
        df_a_aligned = df_a
        df_b_aligned = df_b.reindex(df_a.index)
        original_series = df_a_aligned[base_col]
        mask_a = df_a_aligned[mask_col_a].astype(int).values
        mask_b = df_b_aligned[mask_col_b].fillna(0).astype(int).values

        save_dir = os.path.join(fig_path, task_name, 'methods_comparison')
        filename_prefix = f'{column}_{method_a}_vs_{method_b}'
        return plot_comparison(original_series, mask_a, mask_b, method_a, method_b, save_dir, filename_prefix)
    except Exception as e:
        logger.error(f"{column}: 对比失败 - {e}")
        return []


def discover_columns(data_dir: str, task_name: str, methods: List[str]) -> List[str]:
    """从结果目录中提取所有列名集合，支持方法子目录与扁平目录。"""
    columns = set()
    # 1) 方法子目录优先扫描
    for m in methods:
        method_dir = os.path.join(data_dir, task_name, m)
        if os.path.exists(method_dir):
            for fname in os.listdir(method_dir):
                if not fname.endswith('.csv'):
                    continue
                name = fname[:-4]
                parts = name.split('_')
                if len(parts) >= 6:
                    column = parts[-3]
                    columns.add(column)
    # 2) 兼容旧结构：扁平目录
    dir_path = os.path.join(data_dir, task_name)
    if os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if not fname.endswith('.csv'):
                continue
            name = fname[:-4]
            parts = name.split('_')
            if len(parts) >= 6:
                column = parts[-3]
                columns.add(column)
    return sorted(columns)


def main():
    parser = argparse.ArgumentParser(description='对比两种检测算法（如 stl_wavelet 与 adtk_hbos）并可视化')
    parser.add_argument('--task_name', type=str, required=True, help='任务名称 (global/local)')
    parser.add_argument('--fig_path', type=str, required=True, help='图片保存路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据路径')
    parser.add_argument('--n_jobs', type=int, default=8, help='并行任务数')
    parser.add_argument('--methods', type=str, required=True, help='以逗号分隔的两个方法，例如 "stl_wavelet, adtk_hbos"')
    parser.add_argument('--mode', type=str, default='downsample', choices=['raw', 'downsample'], help='绘图模式：raw 使用原始长度对齐；downsample 使用保存的降采样结果对齐')

    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    if len(methods) < 2:
        logger.error('methods 至少需要包含两个方法进行对比，例如: "stl_wavelet, adtk_hbos"')
        return
    method_a, method_b = methods[0], methods[1]

    logger.info(f"开始算法对比: {method_a} vs {method_b}")
    columns = discover_columns(args.data_path, args.task_name, methods)
    if not columns:
        logger.error('未在结果目录中发现可用列，请先运行异常检测生成结果文件。')
        return

    logger.info(f"共发现 {len(columns)} 个点位，开始并行处理...")
    work = [(col, args.data_path, args.fig_path, args.task_name, method_a, method_b, args.mode) for col in columns]
    with Pool(processes=args.n_jobs) as pool:
        results = pool.starmap(process_point, work)

    total = sum(len(x) for x in results if x)
    success = sum(1 for x in results if x)
    subdir = 'methods_comparison_raw' if args.mode == 'raw' else 'methods_comparison'
    logger.info(f"对比完成！成功点位: {success}/{len(columns)}，生成图片总数: {total}")
    logger.info(f"图表目录: {os.path.join(args.fig_path, args.task_name, subdir)}")


if __name__ == '__main__':
    main()
