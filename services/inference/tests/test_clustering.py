#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   test_clustering.py
@Time    :   2024/12/26
@Author  :   DouFengfeng
@Version :   1.0.0
@Contact :   ff.dou@cyber-insight.com
@License :   (C)Copyright 2019-2026, CyberInsight
@Desc    :   测试聚类算法异常值分割功能
"""

import numpy as np
import pandas as pd
from wavelet import (
    range_split_outliers, 
    extract_outlier_features, 
    cluster_based_outlier_split, 
    adaptive_outlier_split
)


def test_clustering_functions():
    """测试聚类算法相关函数"""
    print("=" * 60)
    print("测试聚类算法异常值分割功能")
    print("=" * 60)
    
    # 1. 生成测试数据
    print("\n1. 生成测试数据...")
    np.random.seed(42)
    n_points = 1000
    
    # 基础信号
    t = np.linspace(0, 10, n_points)
    signal = 2 * np.sin(2 * np.pi * t) + 0.5 * np.random.randn(n_points)
    
    # 添加异常值
    outliers = []
    
    # 局部异常：短时间脉冲
    local_outlier_indices = [200, 201, 202]
    signal[local_outlier_indices] += 3
    outliers.append(local_outlier_indices)
    
    # 全局异常：长时间偏移
    global_outlier_indices = list(range(400, 450))
    signal[global_outlier_indices] += 2
    outliers.append(global_outlier_indices)
    
    # 另一个局部异常
    local_outlier_indices2 = [600, 601, 602, 603]
    signal[local_outlier_indices2] += 2.5
    outliers.append(local_outlier_indices2)
    
    print(f"数据长度: {len(signal)}")
    print(f"异常值段数量: {len(outliers)}")
    
    # 2. 测试特征提取
    print("\n2. 测试特征提取...")
    features = extract_outlier_features(signal, outliers)
    print(f"特征矩阵形状: {features.shape}")
    print(f"特征维度: {features.shape[1]}")
    
    # 3. 测试原始阈值方法
    print("\n3. 测试原始阈值方法...")
    global_indices_threshold, local_indices_threshold = range_split_outliers(signal, outliers, range_th=0.1)
    print(f"阈值方法 - 全局异常: {len(global_indices_threshold)} 个点")
    print(f"阈值方法 - 局部异常: {len(local_indices_threshold)} 个点")
    
    # 4. 测试聚类方法
    print("\n4. 测试聚类方法...")
    
    # KMeans聚类
    print("\n4.1 KMeans聚类:")
    global_indices_kmeans, local_indices_kmeans = cluster_based_outlier_split(
        signal, outliers, method='kmeans', random_state=42
    )
    print(f"KMeans - 全局异常: {len(global_indices_kmeans)} 个点")
    print(f"KMeans - 局部异常: {len(local_indices_kmeans)} 个点")
    
    # DBSCAN聚类
    print("\n4.2 DBSCAN聚类:")
    global_indices_dbscan, local_indices_dbscan = cluster_based_outlier_split(
        signal, outliers, method='dbscan'
    )
    print(f"DBSCAN - 全局异常: {len(global_indices_dbscan)} 个点")
    print(f"DBSCAN - 局部异常: {len(local_indices_dbscan)} 个点")
    
    # 5. 测试自适应方法
    print("\n5. 测试自适应方法...")
    global_indices_auto, local_indices_auto = adaptive_outlier_split(signal, outliers, method='auto')
    print(f"自适应方法 - 全局异常: {len(global_indices_auto)} 个点")
    print(f"自适应方法 - 局部异常: {len(local_indices_auto)} 个点")
    
    # 6. 结果对比
    print("\n6. 结果对比分析...")
    print(f"{'方法':<15} {'全局异常':<10} {'局部异常':<10}")
    print("-" * 40)
    print(f"{'阈值方法':<15} {len(global_indices_threshold):<10} {len(local_indices_threshold):<10}")
    print(f"{'KMeans':<15} {len(global_indices_kmeans):<10} {len(local_indices_kmeans):<10}")
    print(f"{'DBSCAN':<15} {len(global_indices_dbscan):<10} {len(local_indices_dbscan):<10}")
    print(f"{'自适应':<15} {len(global_indices_auto):<10} {len(local_indices_auto):<10}")
    
    # 7. 一致性分析
    print("\n7. 一致性分析...")
    
    # 计算与阈值方法的一致性
    def calculate_agreement(indices1, indices2):
        if len(indices1) == 0 and len(indices2) == 0:
            return 1.0
        if len(indices1) == 0 or len(indices2) == 0:
            return 0.0
        
        # 创建掩码
        mask1 = np.zeros(len(signal), dtype=bool)
        mask2 = np.zeros(len(signal), dtype=bool)
        mask1[indices1] = True
        mask2[indices2] = True
        
        # 计算一致性
        agreement = (mask1 == mask2).sum()
        return agreement / len(signal)
    
    kmeans_agreement = calculate_agreement(global_indices_threshold, global_indices_kmeans)
    dbscan_agreement = calculate_agreement(global_indices_threshold, global_indices_dbscan)
    auto_agreement = calculate_agreement(global_indices_threshold, global_indices_auto)
    
    print(f"KMeans与阈值方法一致性: {kmeans_agreement:.3f}")
    print(f"DBSCAN与阈值方法一致性: {dbscan_agreement:.3f}")
    print(f"自适应方法与阈值方法一致性: {auto_agreement:.3f}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_clustering_functions() 