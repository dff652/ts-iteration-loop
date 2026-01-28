#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   compare_algorithms.py
@Time    :   2024/12/26
@Author  :   DouFengfeng
@Version :   1.0.0
@Contact :   ff.dou@cyber-insight.com
@License :   (C)Copyright 2019-2026, CyberInsight
@Desc    :   对比分析固定阈值方法和聚类算法的异常值分割结果
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from pathlib import Path
from joblib import Parallel, delayed
import logging
from multiprocessing import Pool
from wavelet import extract_outlier_features, split_continuous_outliers

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_comparison_files(data_path, task_name):
    """
    获取需要对比的文件列表。
    
    Parameters:
        data_path (str): 数据路径
        task_name (str): 任务名称 (global/local)
        
    Returns:
        list: 文件路径列表
    """
    task_dir = os.path.join(data_path, task_name)
    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"任务目录不存在: {task_dir}")
    
    # 查找所有包含聚类掩码的CSV文件
    compare_files = []
    for file in os.listdir(task_dir):
        if file.endswith('.csv') and 'cluster' in file:
            compare_files.append(os.path.join(task_dir, file))
    
    # 如果没有找到包含cluster的文件，查找任何CSV文件
    if not compare_files:
        for file in os.listdir(task_dir):
            if file.endswith('.csv'):
                compare_files.append(os.path.join(task_dir, file))
    
    if not compare_files:
        raise FileNotFoundError(f"未找到结果文件，请先运行异常检测")
    
    logger.info(f"找到 {len(compare_files)} 个结果文件")
    return compare_files


def calculate_metrics(data, threshold_col, cluster_col):
    """
    计算两种方法的性能指标。
    
    Parameters:
        data (pd.DataFrame): 数据
        threshold_col (str): 固定阈值方法的列名
        cluster_col (str): 聚类算法的列名
        
    Returns:
        dict: 性能指标字典
    """
    metrics = {}
    
    # 异常点数量
    metrics['threshold_outliers'] = data[threshold_col].sum()
    metrics['cluster_outliers'] = data[cluster_col].sum()
    
    # 异常点比例
    total_points = len(data)
    metrics['threshold_ratio'] = metrics['threshold_outliers'] / total_points
    metrics['cluster_ratio'] = metrics['cluster_outliers'] / total_points
    
    # 一致性指标
    agreement = (data[threshold_col] == data[cluster_col]).sum()
    metrics['agreement_ratio'] = agreement / total_points
    metrics['disagreement_ratio'] = 1 - metrics['agreement_ratio']
    
    # 差异分析
    threshold_only = (data[threshold_col] == 1) & (data[cluster_col] == 0)
    cluster_only = (data[threshold_col] == 0) & (data[cluster_col] == 1)
    
    metrics['threshold_only_count'] = threshold_only.sum()
    metrics['cluster_only_count'] = cluster_only.sum()
    metrics['threshold_only_ratio'] = threshold_only.sum() / total_points
    metrics['cluster_only_ratio'] = cluster_only.sum() / total_points
    
    return metrics


def create_comparison_plots(data, column_name, fig_path, task_name, clustering_method):
    """
    创建对比图表 - 分别保存原始数据对比和异常数量对比
    
    Parameters:
        data (pd.DataFrame): 数据
        column_name (str): 列名
        fig_path (str): 图片保存路径
        task_name (str): 任务名称 (global/local)
        clustering_method (str): 聚类方法
    """
    # 创建图片保存目录
    save_dir = os.path.join(fig_path, task_name, 'comparison')
    os.makedirs(save_dir, exist_ok=True)
    plot_paths = []
    
    # 确保数据索引是时间类型
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
    
    # 获取原始数据列名（排除掩码列）
    data_columns = [col for col in data.columns if not col.endswith('_mask') and not col.endswith('_cluster')]
    if not data_columns:
        logger.error(f"No data column found in {column_name}")
        return None
    
    data_column = data_columns[0]  # 使用第一个数据列
    
    # 1. 原始数据上的算法结果对比（参考combined方式）
    plt.figure(figsize=(20, 8))
    data_len = len(data)
    
    # 绘制原始数据
    plt.plot(range(data_len), data[data_column], label='Original Data', color='blue', linewidth=1)
    
    # 获取数据的最大和最小值
    max_y = data[data_column].max()
    min_y = data[data_column].min()
    height_step = (max_y - min_y) * 0.3
    
    # 绘制固定阈值方法的异常区域
    if f'{task_name}_mask' in data.columns:
        plt.fill_between(
            range(data_len),
            y1=min_y - height_step * 0.5,
            y2=min_y + height_step * 0.5,
            where=data[f'{task_name}_mask'] == 1,
            color='red',
            alpha=0.5,
            label='Fixed Threshold'
        )
    
    # 绘制聚类算法的异常区域
    if f'{task_name}_mask_cluster' in data.columns:
        plt.fill_between(
            range(data_len),
            y1=min_y + height_step * 1.2,
            y2=min_y + height_step * 2.2,
            where=data[f'{task_name}_mask_cluster'] == 1,
            color='green',
            alpha=0.5,
            label=f'{clustering_method.upper()} Clustering'
        )
    
    # 设置横坐标标签
    xticks = np.linspace(0, data_len - 1, num=10, dtype=int)
    xtick_labels = data.index[xticks].to_pydatetime()
    plt.xticks(xticks, [label.strftime('%Y-%m-%d %H:%M:%S') for label in xtick_labels], rotation=45, ha='right')
    
    plt.title(f'{column_name} - {task_name.upper()} Anomaly Detection Comparison on Original Data\n'
              f'Fixed Threshold vs {clustering_method.upper()} Clustering')
    plt.xlabel('Time')
    plt.ylabel(data_column)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存原始数据对比图
    save_path1 = os.path.join(save_dir, f'{column_name}_original_comparison.png')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"保存原始数据对比图: {save_path1}")
    plot_paths.append(save_path1)
    
    # 2. 异常点数量对比
    plt.figure(figsize=(12, 8))
    
    # 准备对比数据
    methods = ['Fixed Threshold', f'{clustering_method.upper()} Clustering']
    global_counts = []
    local_counts = []
    
    # 获取固定阈值方法的计数
    if 'global_mask' in data.columns and 'local_mask' in data.columns:
        global_counts.append(data['global_mask'].sum())
        local_counts.append(data['local_mask'].sum())
    
    # 获取聚类算法的计数
    if 'global_mask_cluster' in data.columns and 'local_mask_cluster' in data.columns:
        global_counts.append(data['global_mask_cluster'].sum())
        local_counts.append(data['local_mask_cluster'].sum())
    
    if global_counts and local_counts:
        x = np.arange(len(methods))
        width = 0.35
        
        # 绘制柱状图
        bars1 = plt.bar(x - width/2, global_counts, width, label='Global Anomalies', color='red', alpha=0.7)
        bars2 = plt.bar(x + width/2, local_counts, width, label='Local Anomalies', color='blue', alpha=0.7)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(global_counts + local_counts) * 0.01,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.title(f'{column_name} - Anomaly Count Comparison')
        plt.xlabel('Detection Method')
        plt.ylabel('Anomaly Count')
        plt.xticks(x, methods)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存异常数量对比图
        save_path2 = os.path.join(save_dir, f'{column_name}_count_comparison.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存异常数量对比图: {save_path2}")
        plot_paths.append(save_path2)
    
    # 3. 特征空间二维散点图（基于聚类标签），仅当聚类掩码存在时生成
    if 'global_mask_cluster' in data.columns and 'local_mask_cluster' in data.columns:
        def mask_to_segments(mask_series):
            idx_list = np.where(mask_series.values == 1)[0].tolist()
            return split_continuous_outliers(idx_list)

        # 基于聚类结果的全局/局部段
        global_segments = mask_to_segments(data['global_mask_cluster'])
        local_segments = mask_to_segments(data['local_mask_cluster'])

        if len(global_segments) + len(local_segments) > 0:
            # 计算每段的二维特征（mean_ratio, range_ratio）
            base_series_df = data[[data_column]]  # 以DataFrame形式传入以兼容特征函数
            features_global = extract_outlier_features(base_series_df, global_segments) if global_segments else np.empty((0, 2))
            features_local = extract_outlier_features(base_series_df, local_segments) if local_segments else np.empty((0, 2))

            plt.figure(figsize=(10, 8))
            if features_global.size > 0:
                plt.scatter(features_global[:, 0], features_global[:, 1], c='red', alpha=0.8, s=25, label='Global (cluster)')
            if features_local.size > 0:
                plt.scatter(features_local[:, 0], features_local[:, 1], c='green', alpha=0.8, s=25, label='Local (cluster)')

            plt.title(f'{column_name} - Feature Scatter by {clustering_method.upper()} Labels')
            plt.xlabel('mean_ratio')
            plt.ylabel('range_ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path3 = os.path.join(save_dir, f'{column_name}_feature_scatter.png')
            plt.savefig(save_path3, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"保存特征空间散点图: {save_path3}")
            plot_paths.append(save_path3)
    
    return plot_paths


def create_point_statistics(data, column_name, fig_path, task_name, clustering_method):
    """
    为单个点位创建统计图表
    
    Parameters:
        data (pd.DataFrame): 数据
        column_name (str): 列名
        fig_path (str): 图片保存路径
        task_name (str): 任务名称 (global/local)
        clustering_method (str): 聚类方法
    """
    # 创建统计目录
    stats_dir = os.path.join(fig_path, task_name, 'statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    # 根据task_name选择对应的掩码列
    threshold_mask_col = f'{task_name}_mask'
    cluster_mask_col = f'{task_name}_mask_cluster'
    
    # 计算指标
    metrics = calculate_metrics(data, threshold_mask_col, cluster_mask_col)
    
    # 创建统计图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{column_name} - {task_name.upper()} Anomaly Detection Statistics\n'
                 f'Fixed Threshold vs {clustering_method.upper()} Clustering', fontsize=16)
    
    # 1. 异常点数量对比
    ax1 = axes[0, 0]
    methods = ['Fixed Threshold', f'{clustering_method.upper()} Clustering']
    counts = [metrics['threshold_outliers'], metrics['cluster_outliers']]
    
    bars = ax1.bar(methods, counts, color=['red', 'green'], alpha=0.7)
    ax1.set_title(f'{task_name.upper()} Anomaly Count Comparison')
    ax1.set_ylabel('Anomaly Count')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                f'{int(count)}', ha='center', va='bottom')
    
    # 2. 检测一致性分析
    ax2 = axes[0, 1]
    agreement_data = [metrics['agreement_ratio'], metrics['disagreement_ratio']]
    labels = ['Detection Agreement', 'Detection Disagreement']
    colors = ['lightgreen', 'lightcoral']
    
    ax2.pie(agreement_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{task_name.upper()} Detection Agreement')
    
    # 3. 异常点比例对比
    ax3 = axes[1, 0]
    ratios = [metrics['threshold_ratio'], metrics['cluster_ratio']]
    
    bars = ax3.bar(methods, ratios, color=['red', 'green'], alpha=0.7)
    ax3.set_title(f'{task_name.upper()} Anomaly Ratio Comparison')
    ax3.set_ylabel('Anomaly Ratio')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(ratios) * 0.01,
                f'{ratio:.3f}', ha='center', va='bottom')
    
    # 4. 差异分析
    ax4 = axes[1, 1]
    diff_data = [metrics['threshold_only_count'], metrics['cluster_only_count']]
    diff_labels = ['Fixed Threshold Only', f'{clustering_method.upper()} Only']
    colors = ['red', 'green']
    
    ax4.bar(diff_labels, diff_data, color=colors, alpha=0.7)
    ax4.set_title(f'{task_name.upper()} Detection Differences')
    ax4.set_ylabel('Point Count')
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, count in enumerate(diff_data):
        ax4.text(i, count + max(diff_data) * 0.01, f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存统计图表
    stats_path = os.path.join(stats_dir, f'{column_name}_statistics.png')
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计数据
    stats_csv_path = os.path.join(stats_dir, f'{column_name}_statistics.csv')
    stats_df = pd.DataFrame([{'column': column_name, **metrics}])
    stats_df.to_csv(stats_csv_path, index=False)
    
    logger.info(f"保存点位统计图表: {stats_path}")
    logger.info(f"保存点位统计数据: {stats_csv_path}")
    
    return stats_path, stats_csv_path


def process_single_point_comparison(file_path, fig_path, task_name, clustering_method):
    """
    处理单个点位的对比分析（在单个进程中完成）
    
    Parameters:
        file_path (str): 数据文件路径
        fig_path (str): 图片保存路径
        task_name (str): 任务名称
        clustering_method (str): 聚类方法
        
    Returns:
        list: 生成的图片路径列表
    """
    try:
        # 加载数据
        data = pd.read_csv(file_path, index_col=0)
        
        # 检查是否包含聚类算法的掩码列
        has_cluster_columns = ('global_mask_cluster' in data.columns and 'local_mask_cluster' in data.columns)
        
        if not has_cluster_columns:
            logger.warning(f"文件 {file_path} 不包含聚类算法掩码列，跳过")
            return []
        
        # 解析文件名获取点位信息
        file_name = os.path.basename(file_path)
        
        column = data.columns[0]
        
        # parts = file_name.replace('.csv', '').split('_')
        # if len(parts) >= 6:
        #     column = parts[5]  # 列名
        # else:
        #     column = "unknown"
        
        logger.info(f"处理点位: {column}, 文件: {file_name}, 形状: {data.shape}")
        
        # 创建对比图表
        plot_paths = create_comparison_plots(data, column, fig_path, task_name, clustering_method)
        
        return plot_paths
        
    except Exception as e:
        logger.error(f"处理文件 {file_path} 失败: {e}")
        return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='对比分析固定阈值方法和聚类算法的异常值分割结果')
    parser.add_argument('--task_name', type=str, required=True, help='任务名称 (global/local)')
    parser.add_argument('--fig_type', type=str, default='combined', help='图表类型')
    parser.add_argument('--fig_path', type=str, required=True, help='图片保存路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据路径')
    parser.add_argument('--n_jobs', type=int, default=8, help='并行任务数')
    parser.add_argument('--clustering_method', type=str, default='kmeans', help='聚类方法')
    
    args = parser.parse_args()
    
    logger.info(f"开始对比分析任务: {args.task_name}")
    logger.info(f"聚类方法: {args.clustering_method}")
    
    try:
        # 1. 获取需要处理的文件列表
        comparison_files = get_comparison_files(args.data_path, args.task_name)
        
        if not comparison_files:
            logger.error("未找到对比数据文件")
            return
        
        # 2. 并行处理每个点位的对比分析
        logger.info(f"开始并行处理 {len(comparison_files)} 个点位的对比分析...")
        
        # 准备并行处理的参数
        process_args = [(file_path, args.fig_path, args.task_name, args.clustering_method) 
                       for file_path in comparison_files]
        
        # 使用并行处理
        with Pool(processes=args.n_jobs) as pool:
            results = pool.starmap(process_single_point_comparison, process_args)
        
        # 统计结果
        total_plots = 0
        successful_points = 0
        
        for i, plot_paths in enumerate(results):
            if plot_paths:
                successful_points += 1
                total_plots += len(plot_paths)
                logger.info(f"点位 {i+1} 成功生成 {len(plot_paths)} 个图表")
            else:
                logger.warning(f"点位 {i+1} 处理失败")
        
        logger.info(f"对比分析完成！")
        logger.info(f"成功处理 {successful_points}/{len(comparison_files)} 个点位")
        logger.info(f"总共生成 {total_plots} 个图表")
        logger.info(f"对比图表保存在: {os.path.join(args.fig_path, args.task_name, 'comparison')}")
        
    except Exception as e:
        logger.error(f"对比分析过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main() 