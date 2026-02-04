"""
统一时序图像生成工具

与 /home/douff/ilabel/qwen3-vl-8B-test/reasoning/Qwen3-VL_test.py 保持一致的图像生成逻辑
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image
from typing import Tuple, List, Optional


# ==================== 图像配置 ====================
class PlotConfig:
    """图像生成配置，与测试脚本保持一致"""
    FIGURE_WIDTH: int = 20
    FIGURE_HEIGHT: int = 4
    FIGURE_DPI: int = 200
    PLOT_LINEWIDTH: float = 0.5
    X_TICKS_COUNT: int = 150
    THUMBNAIL_DPI: int = 200  # 缩略图 DPI


# ==================== 核心图像生成函数 ====================
def create_ts_image(
    data: pd.Series,
    y_range: Optional[Tuple[float, float]] = None,
    start_index: int = 0,
    highlight_regions: Optional[List[Tuple[int, int]]] = None,
    dpi: int = PlotConfig.FIGURE_DPI,
    figsize: Tuple[int, int] = (PlotConfig.FIGURE_WIDTH, PlotConfig.FIGURE_HEIGHT),
    linewidth: float = PlotConfig.PLOT_LINEWIDTH,
    x_ticks_count: int = PlotConfig.X_TICKS_COUNT,
    show_grid: bool = True,
) -> Image.Image:
    """
    创建时间序列图像，与测试脚本 ImageGenerator.create_single_image 保持一致
    
    Args:
        data: 时序数据
        y_range: Y轴范围 (min, max)，如果为 None 则自动计算
        start_index: X轴起始索引
        highlight_regions: 高亮区域列表 [(start, end), ...]
        dpi: 图像 DPI
        figsize: 图像尺寸
        linewidth: 线条宽度
        x_ticks_count: X轴刻度数量
        show_grid: 是否显示网格线
        
    Returns:
        PIL.Image.Image 对象
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # X轴使用索引
    x_indices = np.arange(start_index, start_index + len(data))
    
    # 绘制曲线
    ax.plot(x_indices, data.values, color='black', linewidth=linewidth, alpha=1)
    
    # 禁用科学计数法
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    
    # X轴刻度设置
    if len(data) > 1:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=min(x_ticks_count, len(data)), integer=True))
        ax.tick_params(axis='x', rotation=90, labelsize=6)
    
    # 添加网格线
    if show_grid:
        ax.figure.canvas.draw()  # 确保刻度位置已计算
        xticks = ax.get_xticks()
        for tick in xticks:
            if start_index <= tick <= start_index + len(data):
                ax.axvline(x=tick, color='lightblue', linewidth=0.5, alpha=0.3, zorder=0)
    
    # 高亮异常区域
    if highlight_regions:
        for start, end in highlight_regions:
            if start < end <= start_index + len(data):
                ax.axvspan(start, end, color='#e74c3c', alpha=0.3)
    
    # Y轴范围设置
    if y_range is None:
        y_min, y_max = data.min(), data.max()
    else:
        y_min, y_max = y_range
    
    if abs(y_max - y_min) > 1e-10:
        pad = (y_max - y_min) * 0.1  # 10% 边距
        ax.set_ylim(y_min - pad, y_max + pad)
    
    # 保存到内存
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf).convert("RGB")
    return img


def generate_ts_thumbnail(data, save_path: str) -> bool:
    """
    Generate a thumbnail image for time series data with consistent styling.
    与测试脚本保持一致的图像生成逻辑。
    
    Args:
        data (pd.Series or pd.DataFrame): Time series data.
        save_path (str): Path to save the image.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine values to plot
        if isinstance(data, pd.DataFrame):
            series = data.iloc[:, 0]
        else:
            series = data
        
        # 使用统一的图像生成函数
        img = create_ts_image(
            data=series,
            y_range=None,
            start_index=0,
            highlight_regions=None,
            dpi=PlotConfig.THUMBNAIL_DPI,
            figsize=(PlotConfig.FIGURE_WIDTH, PlotConfig.FIGURE_HEIGHT),
            linewidth=PlotConfig.PLOT_LINEWIDTH,
            x_ticks_count=PlotConfig.X_TICKS_COUNT,
            show_grid=True,
        )
        
        # 保存为 JPG
        img.save(save_path, format='JPEG', quality=95)
        return True
        
    except Exception as e:
        print(f"Failed to generate plot: {save_path} | Error: {e}")
        return False
