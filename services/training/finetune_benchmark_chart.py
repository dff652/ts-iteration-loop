#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成微调结果基准测试柱状图
仿照图1样式
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
algorithms = [
    'ChatTS-14B\n基准版',
    'ChatTS-14B\n微调版',
    'ChatTS-8B\n微调版',
    'Adtk_Hbos'
]
scores = [33.6, 41.5, 43, 67]

# 颜色设置 - 仿照图1
colors = ['#D4A574', '#D4A574', '#D4A574', '#7BA3C9']  # 前三个橙色调，Adtk_Hbos蓝色调

# 创建图形
plt.figure(figsize=(10, 6))
plt.switch_backend('agg')

# 创建柱状图
bars = plt.bar(algorithms, scores, color=colors, edgecolor='black', linewidth=0.8, width=0.6)

# 在柱子上方添加数值标签
for bar, score in zip(bars, scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{score}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 设置标题和标签
plt.title('Benchmark Analysis: Average Score (Fine-tuned)', fontsize=14, fontweight='bold')
plt.xlabel('Algorithms', fontsize=12)
plt.ylabel('Average Score', fontsize=12)

# 设置Y轴范围
plt.ylim(0, 80)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = '/home/dff652/TS-anomaly-detection/ChatTS-Training/finetune_benchmark_chart.png'
plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
print(f"图表已保存至: {output_path}")

# 显示数据摘要
print("\n=== 微调结果数据摘要 ===")
for algo, score in zip(algorithms, scores):
    print(f"{algo.replace(chr(10), ' ')}: {score}")
