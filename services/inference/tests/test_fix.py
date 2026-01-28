#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   test_fix.py
@Time    :   2024/12/26
@Author  :   DouFengfeng
@Version :   1.0.0
@Contact :   ff.dou@cyber-insight.com
@License :   (C)Copyright 2019-2026, CyberInsight
@Desc    :   测试修复后的逻辑
"""

import numpy as np
import pandas as pd
from wavelet import create_outlier_mask


def test_mask_creation():
    """测试掩码创建逻辑"""
    print("=" * 60)
    print("测试掩码创建逻辑")
    print("=" * 60)
    
    # 创建测试数据
    data_length = 1000
    data = pd.DataFrame({
        'test_column': np.random.randn(data_length)
    })
    
    # 测试情况1: 有异常索引
    print("\n1. 测试有异常索引的情况:")
    global_indices = [100, 101, 102, 500, 501]
    local_indices = [200, 201, 300, 301]
    
    global_mask = create_outlier_mask(data, global_indices)
    local_mask = create_outlier_mask(data, local_indices)
    
    print(f"global_indices: {global_indices}")
    print(f"global_mask sum: {global_mask.sum()}")
    print(f"local_mask sum: {local_mask.sum()}")
    
    # 测试情况2: 空异常索引
    print("\n2. 测试空异常索引的情况:")
    empty_indices = []
    
    empty_mask = create_outlier_mask(data, empty_indices)
    print(f"empty_indices: {empty_indices}")
    print(f"empty_mask sum: {empty_mask.sum()}")
    print(f"empty_mask shape: {empty_mask.shape}")
    
    # 测试情况3: None异常索引
    print("\n3. 测试None异常索引的情况:")
    none_indices = None
    
    try:
        none_mask = create_outlier_mask(data, none_indices)
        print(f"none_indices: {none_indices}")
        print(f"none_mask sum: {none_mask.sum()}")
    except Exception as e:
        print(f"处理None索引时出错: {e}")
    
    print("\n测试完成！")


def test_dataframe_operations():
    """测试DataFrame操作"""
    print("\n" + "=" * 60)
    print("测试DataFrame操作")
    print("=" * 60)
    
    # 创建测试数据
    data = pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randn(100)
    })
    
    # 添加掩码列
    data['global_mask'] = np.zeros(100, dtype=int)
    data['local_mask'] = np.zeros(100, dtype=int)
    data['outlier_mask'] = np.zeros(100, dtype=int)
    
    print(f"DataFrame shape: {data.shape}")
    print(f"DataFrame columns: {list(data.columns)}")
    
    # 检查列是否存在
    required_columns = ['global_mask', 'local_mask', 'outlier_mask']
    for col in required_columns:
        if col in data.columns:
            print(f"✓ {col} 列存在")
        else:
            print(f"✗ {col} 列不存在")
    
    # 测试访问列
    try:
        global_mask_sum = data['global_mask'].sum()
        print(f"global_mask sum: {global_mask_sum}")
    except KeyError as e:
        print(f"访问global_mask列时出错: {e}")
    
    print("\nDataFrame操作测试完成！")


if __name__ == "__main__":
    test_mask_creation()
    test_dataframe_operations() 