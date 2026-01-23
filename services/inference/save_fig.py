from iotdb.Session import Session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.dates as mdates
import argparse
import os
from tqdm import tqdm
import time


import json
from multiprocessing import Pool, cpu_count

# import datashader as ds
# import datashader.transfer_functions as tf
from typing import Union, List
from signal_processing import ts_downsample, stl_decompose, read_iotdb, get_fulldata
from config import load_args, save_args_to_file


def load_config(config_filename):
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的绝对路径
    config_path = os.path.join(script_dir, 'configs', config_filename)

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_local_mask(directory,sensor_id,method='piecewise_linear'):
    # 遍历目录及其子目录，查找包含指定点名的CSV文件
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and sensor_id in file:
                matching_files.append(os.path.join(root, file))
    
    # 输出匹配的文件路径
    if matching_files:
        print("找到的文件:")
        for file_path in matching_files:
            if method in file_path:
                print(file_path)
                data = pd.read_csv(file_path,index_col=0)
                return data[['outlier_mask']]
    else:
        print(f"{sensor_id}未找到包含指定点名的CSV文件。")    

def get_global_data(directory, sensor_id, method='wavelet'):
    """
    从指定目录读取包含指定传感器ID的CSV文件
    
    Parameters:
    - directory: 目录路径（应该是方法子目录，如 /opt/results/global/stl_wavelet/）
    - sensor_id: 传感器ID
    - method: 方法名（用于验证文件路径中是否包含该方法名）
    
    Returns:
    - DataFrame 或 None
    """
    # 遍历目录及其子目录，查找包含指定点名的CSV文件
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and sensor_id in file:
                matching_files.append(os.path.join(root, file))
    
    # 输出匹配的文件路径
    if matching_files:
        print(f"找到的文件 (sensor_id={sensor_id}, method={method}, directory={directory}):")
        for file_path in matching_files:
            # 如果指定了method，检查文件路径或文件名中是否包含该方法名
            # 如果没有指定method或method为None，直接返回第一个匹配的文件
            if method is None or method == '':
                print(f"  使用文件（无方法过滤）: {file_path}")
                data = pd.read_csv(file_path, index_col=0)
                return data
            elif method in file_path or method in os.path.basename(file_path):
                print(f"  使用文件（方法匹配）: {file_path}")
                # 验证：确保文件路径中确实包含该方法名
                if method not in file_path and method not in os.path.basename(file_path):
                    print(f"  警告：文件路径 {file_path} 中不包含方法名 {method}，但通过了匹配检查")
                data = pd.read_csv(file_path, index_col=0)
                return data
            else:
                print(f"  跳过文件（方法不匹配）: {file_path} (路径中不包含方法名: {method})")
    else:
        print(f"{sensor_id}未找到包含指定点名的CSV文件（目录: {directory}）。")
        # continue
    return None

def convert_mask_to_label_data(data, column, mask='global_mask', threshold=1):
    """
    将算法结果CSV文件中的mask数据转换为标签格式（开始时间、结束时间）
    
    Parameters:
    - data: DataFrame，包含原始数据和mask列
    - column: 传感器列名
    - mask: 掩码类型 ('global_mask' 或 'local_mask')
    - threshold: 阈值，大于等于此值的mask视为异常
    
    Returns:
    - DataFrame: 包含 '测点', '开始时间', '结束时间', '标签' 列的标签数据
    """
    from dateutil import parser
    from datetime import datetime
    
    def time_convert(time_str):
        """将时间字符串转换为标准格式"""
        if isinstance(time_str, str):
            parsed_time = parser.isoparse(time_str)
            return parsed_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        return time_str
    
    if mask not in data.columns:
        return pd.DataFrame(columns=['测点', '开始时间', '结束时间', '标签'])
    
    combine_masks = data.copy()
    combine_masks['label'] = combine_masks[mask].apply(lambda x: 1 if x >= threshold else 0)
    combine_masks['Group'] = (combine_masks['label'] != combine_masks['label'].shift()).cumsum()
    label_data = combine_masks[combine_masks['label'] == 1]
    
    if label_data.empty:
        return pd.DataFrame(columns=['测点', '开始时间', '结束时间', '标签'])
    
    grouped = label_data.groupby('Group')
    target_data = pd.DataFrame(columns=['测点', '开始时间', '结束时间', '标签'])
    
    for name, group in grouped:
        start_time = group.index.min()
        end_time = group.index.max()
        target_data = pd.concat([target_data, pd.DataFrame({
            '测点': [column],
            '开始时间': [start_time],
            '结束时间': [end_time],
            '标签': [mask],
        })], ignore_index=True)
    
    target_data['开始时间'] = target_data['开始时间'].apply(lambda x: time_convert(x))
    target_data['结束时间'] = target_data['结束时间'].apply(lambda x: time_convert(x))
    return target_data


def load_manual_label_data(manual_label_path):
    """
    加载人工标签数据
    
    Parameters:
    - manual_label_path: 人工标签CSV文件路径
    
    Returns:
    - DataFrame: 包含人工标签的 DataFrame，如果文件不存在则返回空 DataFrame
    """
    if not manual_label_path or not os.path.exists(manual_label_path):
        return pd.DataFrame(columns=['测点', '开始时间', '结束时间', '标签'])
    
    try:
        df_label = pd.read_csv(manual_label_path)
        # 过滤出标注为 '标注' 的行
        manual_label_data = df_label[(df_label['标签'] == '标注')] if '标签' in df_label.columns else df_label
        print(f"已加载人工标签文件: {manual_label_path} (共 {len(manual_label_data)} 条记录)")
        return manual_label_data
    except Exception as e:
        print(f"加载人工标签文件失败: {manual_label_path}, 错误: {e}")
        return pd.DataFrame(columns=['测点', '开始时间', '结束时间', '标签'])


def get_label_data(directory, sensor_id, mask, method=None):
    """
    获取标签数据
    
    Parameters:
    - directory: 目录路径
    - sensor_id: 传感器ID
    - mask: 掩码类型 ('global_mask' 或 'local_mask')
    - method: 可选的方法名，用于过滤包含该方法名的文件
    """
    # print('directory:', directory)
    print('='*50)
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and sensor_id in file:
                # print(file)
                matching_files.append(os.path.join(root, file))
               
    if matching_files:
        # 首先尝试查找包含方法名的文件
        if method is not None:
            method_variants = [method, method.replace('_', '-'), method.replace('-', '_')]
            for file_path in matching_files:
                if mask in file_path:
                    # 检查文件路径是否包含该方法名
                    if any(var in file_path for var in method_variants):
                        print(f"{sensor_id} 找到包含指定点名{mask}和方法{method}的CSV文件{file_path}。")
                        data = pd.read_csv(file_path)
                        return data
        
        # 如果指定了方法名但没找到包含方法名的文件，尝试不指定方法名查找
        # 这是因为 combine_data.py 生成的文件名可能不包含方法名
        for file_path in matching_files:
            if mask in file_path:
                # 如果指定了方法名，但文件名中不包含方法名，也尝试读取
                # 这种情况下，文件可能是 combine_data.py 生成的统一标签文件
                if method is not None:
                    # 检查文件名中是否明确包含其他方法名（避免误匹配）
                    # 如果文件名中包含其他方法名，跳过
                    other_methods = ['stl_wavelet', 'adtk_hbos', 'piecewise_linear', 'standardized', 'iforest', 'wavelet']
                    other_methods = [m for m in other_methods if m != method]
                    if any(m.replace('_', '-') in file_path or m.replace('-', '_') in file_path for m in other_methods):
                        continue
                
                print(f"{sensor_id} 找到包含指定点名{mask}的CSV文件{file_path}。")
                data = pd.read_csv(file_path)
                return data
    else:
        print(f"{directory}未找到包含指定点名{sensor_id}的CSV文件。")
    return None
    
    
      
# def plot_global_and_local_data(global_data, column, fig_path, method='1'):
#     # 创建一个大的画布
#     fig, axes = plt.subplots(5, 1, figsize=(15, 30), gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
#     fig.tight_layout(pad=5)
    
#     # 父图：全量数据使用 outlier_mask
#     axes[0].plot(global_data.index, global_data[column], label='global_data', color='blue')
#     outliers = global_data[global_data['outlier_mask'] == 1]
#     axes[0].scatter(outliers.index, outliers[column], color='red', label='outlier', s=10)
#     axes[0].set_title(f'global_mask_{method}')
#     axes[0].set_xlabel('Time')
#     axes[0].set_ylabel(column)
#     axes[0].legend()
    
#     # 父图：全量数据使用 local_mask
#     axes[1].plot(global_data.index, global_data[column], label='global_data', color='blue')
#     outliers = global_data[global_data['local_mask'] == 1]
#     axes[1].scatter(outliers.index, outliers[column], color='red', label='outlier', s=10)
#     axes[1].set_title('local_mask')
#     axes[1].set_xlabel('Time')
#     axes[1].set_ylabel(column)
#     axes[1].legend()
    
#     # 随机选择3个7天的数据切片并绘制子图
#     for i in range(3):
#         start_index = np.random.randint(0, len(global_data) - 7 * 24 * 60 * 60)
#         end_index = start_index + 7 * 24 * 60 * 60
#         slice_data = global_data.iloc[start_index:end_index]
        
#         # 子图
#         axes[i + 2].plot(slice_data.index, slice_data[column], label='7D', color='green')
#         slice_outliers = slice_data[slice_data['local_mask'] == 1]
#         axes[i + 2].scatter(slice_outliers.index, slice_outliers[column], color='red', label='outlier', s=10)
#         axes[i + 2].set_title(f'7D {i+1}')
#         axes[i + 2].set_xlabel('Time')
#         axes[i + 2].set_ylabel(column)
#         axes[i + 2].legend()
    
#     # 保存图像
#     if not os.path.exists(os.path.dirname(fig_path)):
#         os.makedirs(os.path.dirname(fig_path))  # 创建保存目录
#     plt.savefig(fig_path, dpi=50)
#     plt.close()  # 关闭图像以释放内存



def plot_global_and_local_data(global_data, column, fig_path, method='1'):
    # Ensure global_data index is datetime and aligned properly
    if not isinstance(global_data.index, pd.DatetimeIndex):
        global_data.index = pd.to_datetime(global_data.index, errors='coerce')
    
    # Remove NaT values
    if global_data.index.isna().any():
        global_data = global_data[global_data.index.notna()]
        
    # 检查并转换数据列为数值型
    global_data[column] = pd.to_numeric(global_data[column], errors='coerce')

    # Create a large figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    fig.tight_layout(pad=5)

    # Plot the global data with the global mask
    ax_global = fig.add_subplot(3, 3, (1, 3))  # Merge first row (3 subplots)
    ax_global.plot(global_data.index, global_data[column], label='global_data', color='blue')
    ax_global.fill_between(global_data.index, 
                           global_data[column].min() - 2, 
                           global_data[column].max() + 2, 
                           where=global_data['outlier_mask'] == 1, 
                           color='purple', alpha=0.5, label='Global Mask Anomalies')
    
    ax_global.set_title(f'Global Mask Anomalies - Method {method}')
    ax_global.set_xlabel('Time')
    ax_global.set_ylabel(column)
    ax_global.legend(loc='upper right')
    
    # Set date format
    ax_global.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax_global.xaxis.set_major_locator(mdates.DayLocator(interval=30))  # 7-day interval for ticks
    plt.setp(ax_global.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot the global data with the local mask
    ax_local = fig.add_subplot(3, 3, (4, 6))  # Merge second row (3 subplots)
    ax_local.plot(global_data.index, global_data[column], label='global_data', color='blue')
    ax_local.fill_between(global_data.index, 
                          global_data[column].min() - 2, 
                          global_data[column].max() + 2, 
                          where=global_data['local_mask'] == 1, 
                          color='orange', alpha=0.5, label='Local Mask Anomalies')
    
    ax_local.set_title(f'Local Mask Anomalies - Method {method}')
    ax_local.set_xlabel('Time')
    ax_local.set_ylabel(column)
    ax_local.legend(loc='upper right')
    
    # Set date format
    ax_local.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax_local.xaxis.set_major_locator(mdates.DayLocator(interval=30))  # 7-day interval for ticks
    plt.setp(ax_local.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Divide the global data into three equal segments and plot them in the subplots
    total_duration = (global_data.index[-1] - global_data.index[0]).total_seconds()
    segment_duration = total_duration / 3
    
    for i in range(3):
        start_time = global_data.index[0] + pd.Timedelta(seconds=i * segment_duration)
        end_time = start_time + pd.Timedelta(days=7)
        
        # Ensure the end time is within bounds
        if end_time > global_data.index[-1]:
            end_time = global_data.index[-1]
        
        slice_data = global_data[(global_data.index >= start_time) & (global_data.index <= end_time)]
        
        while slice_data.empty and end_time < global_data.index[-1]:
            start_time += pd.Timedelta(days=3)  # 向后移动3天
            end_time += pd.Timedelta(days=3)
            slice_data = global_data[(global_data.index >= start_time) & (global_data.index <= end_time)]
        
        ax = fig.add_subplot(3, 3, 7 + i)  # Plot in the third row
        ax.plot(slice_data.index, slice_data[column], label='7D', color='green')
        ax.fill_between(slice_data.index, 
                        slice_data[column].min() - 2, 
                        slice_data[column].max() + 2, 
                        where=slice_data['local_mask'] == 1, 
                        color='orange', alpha=0.5, label='Local Mask Anomalies')

        ax.set_title(f'7D Segment {i+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel(column)
        ax.legend(loc='upper right')
        
        # Set date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))  # 1-day interval for ticks
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Highlight corresponding time range in the larger plots
        ax_global.axvspan(slice_data.index[0], slice_data.index[-1], color='red', alpha=0.3, linestyle='--')
        ax_local.axvspan(slice_data.index[0], slice_data.index[-1], color='red', alpha=0.3, linestyle='--')

    # Save the figure
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
        
    
    plt.savefig(fig_path, dpi=50)
    plt.close()

    print(f'保存图像到: {fig_path}')
    print(f'图像标题: Global and Local Mask Anomalies - Method {method}')
    print(f'数据形状: {global_data.shape}')
    
    
def plot_global(data, col, fig_path, method='1', mask_col='outlier_mask'):

    
    plt.figure(figsize=(28, 11))
    
    # 确保数据索引是时间类型
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
        
    # 绘制原始数据曲线，使用索引序列号作为横坐标
    data_len = data.shape[0]
    plt.plot(range(data_len) ,data[col], label='Original Data')
    
    # 根据 mask 列的值填充异常点的区域
    if mask_col in data.columns:
        plt.fill_between(range(data_len), data[col].min()*1.02, data[col].max()*1.02, 
                         where=data[mask_col] == 1, color='red', alpha=0.5, label=f'Anomalies ({mask_col})')
    else:
        # 回退到 outlier_mask（兼容旧结果）
        if 'outlier_mask' in data.columns:
            plt.fill_between(range(data_len), data[col].min()*1.02, data[col].max()*1.02, 
                             where=data['outlier_mask'] == 1, color='red', alpha=0.5, label='Anomalies (outlier_mask)')
    
    # 设置日期格式
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    
    # 手动设置横坐标
    # xticks = np.linspace(0, data_len - 1, num=10, dtype=int)  # 取 10 个间隔均匀的点
    # xtick_labels = data.index[xticks].to_pydatetime()  # 对应的时间标签

    # plt.xticks(xticks, [label.strftime('%Y-%m-%d %H:%M:%S') for label in xtick_labels] ,rotation=45, ha='right')

   
    plt.xticks(rotation=45)
    plt.legend(loc="upper right")
    plt.title(f'Sensor ID: {col} Method {method}')  # 可以添加标题显示sensor_id
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # 保存图像

    
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
        
    
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()   


def plot_combined_global(data_list, col, fig_path, methods):
    """
    将三种方法的结果标签绘制在同一个图上。
    
    参数:
    - data_list: 包含三种方法数据的列表，每个元素都是一个 DataFrame。
    - col: 需要绘制的列名。
    - fig_path: 图像保存路径。
    - methods: 一个包含方法名的列表，例如 ['wavelet', 'piecewise_linear', 'method3']。
    """
    
    plt.figure(figsize=(28, 11))
    
    # 确保数据索引是时间类型
    for data in data_list:
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, errors='coerce')
    
    # 绘制原始数据曲线，使用索引序列号作为横坐标
    data_len = data_list[0].shape[0]  # 假设所有方法的数据长度相同
    plt.plot(range(data_len), data_list[0][col], label='Original Data', color='blue')
    
    # 根据不同方法的 mask 列绘制不同的异常区域
    fill_colors = ['red', 'green', 'purple']

    # 填充的高度设置，分别设置 y1 和 y2，保证每种方法填充在不同高度上
    max_y = data_list[0][col].max()  # 获取数据的最大值
    min_y = data_list[0][col].min()  # 获取数据的最小值
    
    height_step = (max_y - min_y) * 0.36  # 设置每个填充区域之间的高度间隔
    
    # 确保每个填充区域在原始曲线的不同高度
    for i, data in enumerate(data_list):
        if 'outlier_mask' in data.columns:
            # y1 和 y2 的高度基于原始数据的范围设置，使得它们在原始曲线的不同高度上
            # y1 = -0.1*min_y + i * height_step  # 每种方法在原始数据下方按不同高度分布
            # y2 = y1 + height_step  # 填充区域的高度范围

            if i == 0:
                # 第一组方法的填充区域低于最小值
                y1 = min_y -height_step*0.1  # 填充区域起点低于最小值
                y2 = y1 + height_step  # 填充到最小值的位置
            else:
                # 其余方法的填充区域在原始数据曲线的不同高度
                y1 = min_y  + (i-0.1) * height_step  # 填充区域在原始数据的上方
                y2 = y1 + height_step

            plt.fill_between(
                range(data_len), 
                y1=y1,
                y2=y2,
                where=data['outlier_mask'] == 1,
                color=fill_colors[i],
                alpha=0.5,  # 设置透明度以便更明显地看到填充颜色
                label=f'Method {i}'
            )


    # 手动设置横坐标标签（可选）
    xticks = np.linspace(0, data_len - 1, num=10, dtype=int)  # 取 10 个间隔均匀的点
    xtick_labels = data_list[0].index[xticks].to_pydatetime()  # 对应的时间标签

    plt.xticks(xticks, [label.strftime('%Y-%m-%d %H:%M:%S') for label in xtick_labels], rotation=45, ha='right')

    plt.legend(loc="upper right")
    plt.title(f'Sensor ID: {col} Combined Methods')  # 可以添加标题显示 sensor_id
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # 保存图像
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f'图像已保存到: {fig_path}')



def plot_local(data, label_data, fig_fig_path='test_anomaly', mask = 'global_mask'):
    
    data.index = pd.to_datetime(data.index)
    
    # data.index = data.index.tz_convert('Asia/Shanghai')
    # data.index = data.index.floor('S') 
    column = label_data.iloc[0, 0]
    
    for i in range(label_data.shape[0]):
        start_time = pd.to_datetime(label_data.iloc[i, 1])
        end_time = pd.to_datetime(label_data.iloc[i, 2])
        
        start_time = start_time.tz_localize('Asia/Shanghai')
        end_time = end_time.tz_localize('Asia/Shanghai')

        if start_time in data.index and end_time in data.index:
            start_idx = data.index.get_loc(start_time)
            end_idx = data.index.get_loc(end_time)
        else:
            print(f"时间点 {start_time} 或 {end_time} 不在索引中，{column}跳过")
            continue
        
        fig_path = os.path.join(fig_fig_path, f'{label_data.shape[0]}_{mask}_{column}_{start_time}_{end_time}_local_plot.png')
        if os.path.exists(fig_path):
            print(f"图像已存在，跳过生成: {fig_path}")
            continue

        # 前后5小时和20小时的时间范围
        five_hours_before = start_time - pd.Timedelta(hours=5)
        five_hours_after = end_time + pd.Timedelta(hours=5)
        twenty_hours_before = start_time - pd.Timedelta(hours=20)
        twenty_hours_after = end_time + pd.Timedelta(hours=20)
        
        five_hours_before = five_hours_before.tz_convert('Asia/Shanghai')
        five_hours_after = five_hours_after.tz_convert('Asia/Shanghai')
        
        twenty_hours_before = twenty_hours_before.tz_convert('Asia/Shanghai')
        twenty_hours_after = twenty_hours_after.tz_convert('Asia/Shanghai')

        # 获取全局和局部图的数据片段
        df_part1 = data[(data.index >= five_hours_before) & (data.index <= five_hours_after)]
        df_part2 = data[(data.index >= twenty_hours_before) & (data.index <= twenty_hours_after)]

        # 设置绘图布局
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(3, 1, figure=fig)
        ax0 = fig.add_subplot(gs[0, :])  # 全局图
        ax1 = fig.add_subplot(gs[1, :])  # 局部图1（前后5小时）
        ax2 = fig.add_subplot(gs[2, :])  # 局部图2（前后20小时）

        # 设置图的边距和坐标轴刻度
        for ax in [ax0, ax1, ax2]:
            ax.margins(x=0.01, y=0.01)
            ax.tick_params(labelsize=10)

        # 全局图
        ax0.plot(range(len(data)), data[column], label='Data', color='b')  # 使用时间索引作为横坐标
        
        
        # start_idx = data.index.get_indexer([start_time], method='nearest')[0]
        # end_idx = data.index.get_indexer([end_time], method='nearest')[0]

        # 将 range 转换为 numpy 数组
        x_range = np.arange(len(data))
        ax0.fill_between(x_range, data[column].min(), data[column].max(),
                        where=(x_range >= start_idx) & (x_range <= end_idx),
                        color='r', alpha=0.5)
        ax0.set_title('Global View', fontsize=12)

        # 局部图1（前后5小时）
        x_range_part1 = np.arange(len(df_part1))
        ax1.plot(x_range_part1, df_part1[column], label='Data', color='b')
        
        start_idx_1 = df_part1.index.get_loc(start_time) if start_time in df_part1.index else 0
        end_idx_1 = df_part1.index.get_loc(end_time) if end_time in df_part1.index else len(df_part1) - 1
        
        ax1.fill_between(x_range_part1, df_part1[column].min(), df_part1[column].max(),
                        where=(x_range_part1 >= start_idx_1) & (x_range_part1 <= end_idx_1),
                        color='r', alpha=0.5)
        ax1.set_title('Local View: ±5 Hours', fontsize=12)

        # 局部图2（前后20小时）
        x_range_part2 = np.arange(len(df_part2))
        ax2.plot(x_range_part2, df_part2[column], label='Data', color='b')
        
        start_idx_2 = df_part2.index.get_loc(start_time) if start_time in df_part2.index else 0
        end_idx_2 = df_part2.index.get_loc(end_time) if end_time in df_part2.index else len(df_part2) - 1
        
        ax2.fill_between(x_range_part2, df_part2[column].min(), df_part2[column].max(),
                        where=(x_range_part2 >= start_idx_2) & (x_range_part2 <= end_idx_2),
                        color='r', alpha=0.5)
        ax2.set_title('Local View: ±20 Hours', fontsize=12)

        # 使 ax2 的 y 轴刻度和 ax0 保持一致
        ax2.set_ylim(ax0.get_ylim())  # 设置 ax2 的 y 轴范围与 ax0 一致
        ax2.set_yticks(ax0.get_yticks())  # 设置 ax2 的 y 轴刻度与 ax0 一致

        # 设置统一的横坐标格式
        for ax in [ax0, ax1, ax2]:
            ax.set_xticks(ax.get_xticks())  # 保持横坐标时间刻度
            ax.tick_params(axis='x', rotation=45)  # 旋转标签以便于阅读

        # 设置标题和保存图像
        plt.suptitle(f'{mask}_{column}_{start_time} to {end_time}', fontsize=15)
       
        plt.savefig(fig_path, dpi=100)
        print(f'图像已保存到: {fig_path}')
        plt.close()

def plot_combined_labels(raw_data, 
                         algorithm_label, 
                         manual_label,
                         column, 
                         fig_path,
                         mask,
                         method_name=None
                         ):
    """
    将原始数据与算法标签和人工标签绘制在同一个图上。

    参数:
    - raw_data: 原始数据 DataFrame，索引为时间，列为传感器数据。
    - algorithm_label: 算法标签数据 DataFrame，包含 start_time、end_time 列。
    - manual_label: 人工标签数据 DataFrame，包含 开始时间、结束时间 列。
    - column: 绘制的列名。
    - fig_path: 图像保存路径。
    - method_name: 算法名称，用于图例显示。
    """
    plt.figure(figsize=(28, 11))

    # 确保原始数据索引是时间类型
    if not isinstance(raw_data.index, pd.DatetimeIndex):
        raw_data.index = pd.to_datetime(raw_data.index, errors='coerce')
    
    # 确保索引是唯一的（处理重复索引问题）
    if raw_data.index.duplicated().any():
        print(f"警告: 检测到重复的时间索引，将删除重复项 (点位: {column})")
        raw_data = raw_data[~raw_data.index.duplicated(keep='first')]

    # 将索引时区信息统一为UTC，确保兼容性
    raw_data.index = raw_data.index.tz_localize(None)

    # 绘制原始数据曲线
    data_len = raw_data.shape[0]
    plt.plot(range(data_len), raw_data[column], label='Original Data', color='blue')

    # 获取原始数据的最大和最小值
    max_y = raw_data[column].max()
    min_y = raw_data[column].min()
    height_step = (max_y - min_y) * 0.6  # 设置填充区域的高度间隔

    # 标志位：确保只绘制一次图例
    algorithm_label_drawn = False
    manual_label_drawn = False

    # 绘制算法标签
    # 确定算法标签的显示名称
    algorithm_label_name = method_name if method_name else 'Algorithm Label'
    
    for idx, row in algorithm_label.iterrows():
        # 找到最接近的索引位置
        start_time = pd.to_datetime(row['开始时间']).tz_localize(None)
        end_time = pd.to_datetime(row['结束时间']).tz_localize(None)
        start_idx = raw_data.index.get_indexer([start_time], method='nearest')[0]
        end_idx = raw_data.index.get_indexer([end_time], method='nearest')[0]
        plt.fill_between(
            range(start_idx, end_idx + 1),
            y1=min_y - height_step * 0.2,
            y2=min_y + height_step,
            color='red',
            alpha=0.5,
            label=algorithm_label_name if not algorithm_label_drawn else ""
        )
        algorithm_label_drawn = True
    
    # 如果没有算法标签数据，但指定了算法名称，也要在图例中显示（用空数据）
    if algorithm_label.empty and method_name:
        # 创建一个不可见的填充区域，只是为了在图例中显示算法名称
        plt.fill_between([], [], [], color='red', alpha=0.5, label=method_name)

    # 绘制人工标签
    for idx, row in manual_label.iterrows():
        # 找到最接近的索引位置
        start_time = pd.to_datetime(row['开始时间']).tz_localize(None)
        end_time = pd.to_datetime(row['结束时间']).tz_localize(None)
        start_idx = raw_data.index.get_indexer([start_time], method='nearest')[0]
        end_idx = raw_data.index.get_indexer([end_time], method='nearest')[0]
        plt.fill_between(
            range(start_idx, end_idx + 1),
            y1=min_y + height_step * 1.1,
            y2=min_y + height_step * 2.5,
            color='green',
            alpha=0.5,
            label='Manual Label' if not manual_label_drawn else ""
        )
        manual_label_drawn = True

    # 手动设置横坐标标签
    xticks = np.linspace(0, data_len - 1, num=10, dtype=int)
    xtick_labels = raw_data.index[xticks].to_pydatetime()
    plt.xticks(xticks, [label.strftime('%Y-%m-%d %H:%M:%S') for label in xtick_labels], rotation=45, ha='right')

    plt.legend(loc="upper right")
    # 根据是否有算法名称来设置标题
    if method_name:
        plt.title(f'Sensor ID: {column} Algorithm ({method_name}) vs Manual Labels')
    else:
        plt.title(f'Sensor ID: {column} Algorithm vs Manual Labels')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    # 保存图像
    # 确保 fig_path 是目录路径，如果不是则创建 combined 子目录
    if os.path.isfile(fig_path):
        fig_dir = os.path.dirname(fig_path)
    else:
        fig_dir = os.path.join(fig_path, 'combined')
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    fig_file_path = os.path.join(fig_dir, f"{algorithm_label.shape[0]}_{mask}_{raw_data.shape[0]}_{column}_{algorithm_label.shape[0]}_{manual_label.shape[0]}.png")
    plt.savefig(fig_file_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f'图像已保存到: {fig_file_path}')


def process_column(column, global_path, local_path, method):
    try:
        # 获取该点位的全局数据和标签
        global_data = get_global_data(global_path, column, method)
        
        fig_path = f'./test/{global_data.shape[0]}_{column}_{method}_plot.png'
        
        # 检查图像是否已存在
        if os.path.exists(fig_path):
            print(f"图像已存在，跳过生成: {fig_path}")
            return
        
        if global_data is None or global_data.empty:
            print(f"未找到全局数据文件或数据为空: {column}")
            return
        
        print(f'global_data shape: {global_data.shape}')
        
        local_mask = get_local_mask(local_path, column, method)
        
        if local_mask is None or local_mask.empty:
            print(f"未找到本地标签文件或数据为空: {column}")
            return
        
        print(f'local_mask shape: {local_mask.shape}')
        
        global_data['local_mask'] = local_mask
        
        # 绘制图形
        if method == 'wavelet':
            plot_global_and_local_data(global_data, column, fig_path, '1')
        else:
            plot_global_and_local_data(global_data, column, fig_path, '0')
    
    except Exception as e:
        print(f"处理点位 {column} 时出错: {e}")




def process_column_global(column, global_path, local, method):        
    
    global_data = get_global_data(global_path, column, method)
        
    
    
    if global_data is None or global_data.empty:
        print(f"未找到全局数据文件或数据为空: {column}")
        return
    
    
    
    fig_path = f'./global_figs/{global_data.shape[0]}_{column}_{method}_plot.png'
    
    # 检查图像是否已存在
    if os.path.exists(fig_path):
        print(f"图像已存在，跳过生成: {fig_path}")
        return
    
    if method == 'wavelet':
        plot_global(global_data, column, fig_path, '1')
    else:
        plot_global(global_data, column, fig_path, '0')
        
        
def process_combined_column(column, global_path, methods):
    """
    处理每个列的数据，使用三种方法的结果绘制在同一个图上。
    
    参数:
    - column: 需要处理的列名。
    - global_path: 全局数据路径。
    - methods: 一个包含方法名的列表，例如 ['wavelet', 'piecewise_linear', 'standardized']。
    """
    
    data_list = []
    for method in methods:
        data = get_global_data(global_path, column, method)
        if data is None or data.empty:
            print(f"未找到全局数据文件或数据为空: {column} (Method: {method})")
            return
        data_list.append(data)

    fig_path = f'./combined_figs/{data_list[0].shape[0]}_{column}_combined_plot.png'

    # 检查图像是否已存在
    if os.path.exists(fig_path):
        print(f"图像已存在，跳过生成: {fig_path}")
        return
    
    plot_combined_global(data_list, column, fig_path, methods) 
 
 
def process_column_local(info,  label_path, mask_to_use='global_mask'):
    # try:
    # 获取全局数据
    path, column, st, et  = info
    # global_data = get_global_data(global_path, column, 'wavelet')
    data = read_iotdb(
        target_column=column,
        path=path,
        st=st,
        et=et)
    global_data = get_fulldata(data, column)
    if global_data is None or global_data.empty:
        print('执行 process_column_local')
        print(f"未找到全局数据文件或数据为空: {column}")
        print('开始从iotdb读取数据')
        global_data = read_iotdb(path = path, target_column=column, st=st, et=et)
        print(f'数据大小：{global_data.shape}')
    # 仅根据 task_name 选择的 mask 绘制
    masks = [mask_to_use]
    for mask in masks:
        # 获取标签数据
        label_data = get_label_data(label_path, column, mask)
        if label_data is None or label_data.empty:
            print(f"未找到标签数据文件或数据为空: {column} {mask}")
            continue 

    # fig_path = '/opt/results/test'
    
        # 检查并创建保存路径目录
        if not os.path.exists(label_path):
            print(f"创建目录: {label_path}")
            os.makedirs(label_path)
        
        # 绘制局部图像
        plot_local(global_data, label_data, label_path, mask)

    # except Exception as e:
    #     print(f"plot_local 处理点位 {column} 时出错: {e}")


def process_column_normal(info, data_path, method, fig_path, mask_col='outlier_mask'):
    """
    处理普通视图（单算法结果图）
    
    Parameters:
    - info: (path, column, st, et) 传感器信息
    - data_path: 数据路径
    - method: 算法方法名
    - fig_path: 图像保存路径
    """
    path, column, st, et = info
    
    # 检查是否已有图像结果（使用点位匹配，更高效）
    save_dir = os.path.join(fig_path, 'normal')
    if os.path.exists(save_dir):
        # 查找包含该点位名称和方法名的图像文件
        existing_files = [f for f in os.listdir(save_dir) if f.endswith('.png') and column in f and method in f]
        if existing_files:
            print(f"点位 {column} (方法: {method}) 已有图像结果，跳过生成。现有文件: {existing_files[0]}")
            return
    
    global_path = os.path.join(data_path, 'global')
    
    # 获取全局数据
    global_data = get_global_data(global_path, column, method)
    if global_data is None or global_data.empty:
        print(f"未找到全局数据文件或数据为空: {column} (Method: {method})")
        return
    
    # 构建保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig_file_path = os.path.join(save_dir, f'{global_data.shape[0]}_{column}_{method}_plot.png')
    
    # 根据方法名判断显示方式
    if method in ['wavelet', 'stl_wavelet']:
        plot_global(global_data, column, fig_file_path, '1', mask_col=mask_col)
    else:
        plot_global(global_data, column, fig_file_path, '0', mask_col=mask_col)


def process_column_with_labels(info, 
                               algorithm_label_path, 
                               manual_label_data,
                               fig_path,
                               mask='global_mask',
                               methods=None,
                               data_path=None,
                               use_direct_data=False):
    """
    处理算法标签和人工标签对比图，支持多算法对比
    
    Parameters:
    - info: (path, column, st, et) 传感器信息
    - algorithm_label_path: 算法标签路径
    - manual_label_data: 人工标签数据
    - fig_path: 图像保存路径
    - mask: 掩码类型 ('global_mask' 或 'local_mask')
    - methods: 算法方法列表，如果提供多个方法，将绘制多算法对比图
    """
    path, column, st, et = info
    
    # 检查是否已有图像结果（使用点位匹配，更高效）
    if methods and len(methods) > 0:
        # 构建图像保存目录：根据算法名称创建子目录
        method_subdir = '_vs_'.join(sorted(methods))
        fig_dir = os.path.join(fig_path, 'combined', method_subdir)
        if os.path.exists(fig_dir):
            # 查找包含该点位名称的图像文件
            existing_files = [f for f in os.listdir(fig_dir) if f.endswith('.png') and column in f]
            if existing_files:
                print(f"点位 {column} 已有图像结果，跳过生成。现有文件: {existing_files[0]}")
                return
    else:
        # 单算法情况
        fig_dir = os.path.join(fig_path, 'combined')
        if os.path.exists(fig_dir):
            existing_files = [f for f in os.listdir(fig_dir) if f.endswith('.png') and column in f]
            if existing_files:
                print(f"点位 {column} 已有图像结果，跳过生成。现有文件: {existing_files[0]}")
                return
    
    # 优先使用算法结果文件中的数据，如果不可用再读取原始数据
    raw_data = None
    if use_direct_data and data_path and methods and len(methods) > 0:
        # 遍历所有算法，选择数据量最大的结果文件作为 raw_data
        # 这样可以避免因第一个算法数据量较少（如 chatts 只有 768 行）导致其他算法标签映射异常
        best_method = None
        best_data = None
        best_data_len = 0
        
        for method in methods:
            method_data_path = os.path.join(data_path, method)
            if os.path.exists(method_data_path):
                global_data = get_global_data(method_data_path, column, method)
                if global_data is not None and not global_data.empty and column in global_data.columns:
                    current_len = len(global_data)
                    if current_len > best_data_len:
                        best_data_len = current_len
                        best_method = method
                        best_data = global_data
        
        # 使用数据量最大的算法结果
        if best_data is not None:
            raw_data = pd.DataFrame()
            raw_data.index = best_data.index
            raw_data[column] = best_data[column]
            print(f"使用算法结果文件中的数据作为原始数据 (方法: {best_method}, 数据点: {len(raw_data)}, 选自 {len(methods)} 个算法中数据量最大者)")
    
    # 如果无法从算法结果文件获取数据，则从IoTDB读取
    if raw_data is None or raw_data.empty:
        raw_data = read_iotdb(target_column=column, path=path, st=st, et=et)
        if raw_data is None or raw_data.empty:
            print(f"未找到数据或数据为空: {column}")
            return
        print(f"从IoTDB读取原始数据 (数据点: {len(raw_data)})")
    
    # 确保索引是唯一的（处理重复索引问题）
    if raw_data.index.duplicated().any():
        print(f"警告: 检测到重复的时间索引，将删除重复项 (点位: {column}, 原始数据点: {len(raw_data)})")
        raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
        print(f"处理后的数据点: {len(raw_data)}")

    # 统一处理：支持单个或多个方法
    if methods and len(methods) > 0:
        algorithm_labels = []
        for method in methods:
            label_data = None
            # 如果 use_direct_data=True，跳过标签文件查找，直接读取算法结果CSV文件
            if not use_direct_data:
                # 首先尝试从标签文件中读取
                label_data = get_label_data(algorithm_label_path, column, mask, method=method)
            
            # 如果找不到标签文件或 use_direct_data=True，尝试直接从算法结果CSV文件中读取
            if label_data is None or label_data.empty:
                # 从 data_path 中读取算法结果CSV文件
                # data_path 应该是 /opt/results/global 这样的路径，方法子目录在下面
                method_data_path = None
                if data_path is not None:
                    # 构建方法子目录路径：/opt/results/global/{method}/
                    method_data_path = os.path.join(data_path, method)
                    if not os.path.exists(method_data_path):
                        method_data_path = None
                
                # 如果 data_path 不存在或未提供，尝试从 algorithm_label_path 的上级目录查找
                if method_data_path is None:
                    # 尝试从 /opt/results/global/{method}/ 目录读取
                    # algorithm_label_path 可能是 /opt/results/figs，需要找到 /opt/results/global/{method}/
                    base_dir = os.path.dirname(algorithm_label_path)  # /opt/results
                    data_path_guess = os.path.join(base_dir, 'global', method)
                    if os.path.exists(data_path_guess):
                        method_data_path = data_path_guess
                
                # 如果找到了方法子目录，尝试读取算法结果CSV文件
                if method_data_path is not None and os.path.exists(method_data_path):
                    print(f"尝试从算法结果CSV文件读取 {method} 的数据 (路径: {method_data_path})")
                    global_data = get_global_data(method_data_path, column, method)
                    if global_data is not None and not global_data.empty:
                        # 添加调试信息：打印实际读取的文件路径和数据形状
                        print(f"  -> {method} 成功读取数据，形状: {global_data.shape}, 列: {list(global_data.columns)}")
                        if mask in global_data.columns:
                            # 统计 global_mask 的值
                            mask_sum = global_data[mask].sum() if mask in global_data.columns else 0
                            print(f"  -> {method} {mask} 列中异常点总数: {mask_sum}")
                            print(f"从算法结果CSV文件直接读取 {method} 的数据并转换为标签格式 (路径: {method_data_path})")
                            label_data = convert_mask_to_label_data(global_data, column, mask, threshold=1)
                            print(f"  -> {method} 转换后标签数量: {len(label_data)}")
                        else:
                            print(f"  -> {method} 在路径 {method_data_path} 中未找到包含 {mask} 列的数据，可用列: {list(global_data.columns)}")
                    else:
                        print(f"  -> {method} 未能读取数据或数据为空")
                else:
                    print(f"  -> {method} 未找到算法结果目录，尝试的路径: {method_data_path if method_data_path else 'None'}")
            
            if label_data is not None and not label_data.empty:
                print(f"  -> {method} 最终标签数量: {len(label_data)}")
                algorithm_labels.append((method, label_data))
            else:
                print(f"  -> {method} 未找到标签数据")
        
        # 构建完整的算法标签列表（包括未找到数据的算法）
        # 确保图例中显示所有选项中的算法名称
        all_method_labels = []
        for method in methods:
            # 查找是否找到了该算法的标签数据
            found_label = None
            for method_name, label_data in algorithm_labels:
                if method_name == method:
                    found_label = label_data
                    break
            
            if found_label is not None:
                all_method_labels.append((method, found_label))
            else:
                # 即使没有找到数据，也要添加到列表中（用空DataFrame）
                all_method_labels.append((method, pd.DataFrame(columns=['测点', '开始时间', '结束时间', '标签'])))
        
        if len(all_method_labels) > 1:
            # 多个算法对比（使用所有算法，包括未找到数据的）
            plot_multi_algorithm_labels(raw_data, all_method_labels, manual_label_data, column, fig_path, mask)
        elif len(all_method_labels) == 1:
            # 只有一个算法，使用单算法对比
            method_name, label_data = all_method_labels[0]
            manual_label_col = manual_label_data[(manual_label_data['测点'] == column)]
            if not manual_label_col.empty:
                plot_combined_labels(raw_data, label_data, manual_label_col, column, fig_path, mask, method_name=method_name)
            else:
                # 如果没有人工标签，也绘制单算法结果
                plot_combined_labels(raw_data, label_data, pd.DataFrame(columns=['测点', '开始时间', '结束时间', '标签']), column, fig_path, mask, method_name=method_name)
        else:
            print(f"未找到任何算法标签数据: {column}")
    else:
        # 原有的单算法对比逻辑（兼容旧代码）
        algorithm_label_data = get_label_data(algorithm_label_path, column, mask)
        manual_label_col = manual_label_data[(manual_label_data['测点'] == column)]
        
        if algorithm_label_data is None or manual_label_col.empty:
            print(f"未找到相应的标签数据: {column}")
            return

        plot_combined_labels(raw_data, algorithm_label_data, manual_label_col, column, fig_path, mask)


def plot_multi_algorithm_labels(raw_data, 
                                algorithm_labels, 
                                manual_label,
                                column, 
                                fig_path,
                                mask):
    """
    将原始数据与多个算法标签和人工标签绘制在同一个图上。
    
    Parameters:
    - raw_data: 原始数据 DataFrame，索引为时间，列为传感器数据。
    - algorithm_labels: 算法标签列表，每个元素为 (method_name, label_data) 元组。
    - manual_label: 人工标签数据 DataFrame，包含 开始时间、结束时间 列。
    - column: 绘制的列名。
    - fig_path: 图像保存路径。
    - mask: 掩码类型。
    """
    plt.figure(figsize=(28, 11))

    # 确保原始数据索引是时间类型
    if not isinstance(raw_data.index, pd.DatetimeIndex):
        raw_data.index = pd.to_datetime(raw_data.index, errors='coerce')
    
    # 确保索引是唯一的（处理重复索引问题）
    if raw_data.index.duplicated().any():
        print(f"警告: 检测到重复的时间索引，将删除重复项 (点位: {column})")
        raw_data = raw_data[~raw_data.index.duplicated(keep='first')]

    # 将索引时区信息统一为UTC，确保兼容性
    raw_data.index = raw_data.index.tz_localize(None)

    # 绘制原始数据曲线
    data_len = raw_data.shape[0]
    
    # 获取原始数据的最大和最小值
    max_y = raw_data[column].max()
    min_y = raw_data[column].min()
    data_range = max_y - min_y
    
    # 计算标签区域的高度（基于数据范围）
    label_height = data_range * 0.15  # 每个标签区域的高度
    
    # 检查是否有人工标签
    manual_label_col = manual_label[(manual_label['测点'] == column)]
    has_manual_label = not manual_label_col.empty
    
    # 统一计算所有标签（算法标签 + 人工标签）
    total_labels = len(algorithm_labels)
    if has_manual_label:
        total_labels += 1
    
    # 计算需要的上下空间
    # 上方：第一个标签 + 后续偶数索引标签
    num_upper_labels = (total_labels + 1) // 2  # 向上取整
    upper_space = label_height * 0.3 + num_upper_labels * label_height
    
    # 下方：第二个标签 + 后续奇数索引标签
    num_lower_labels = total_labels // 2
    lower_space = label_height * 1.3 + num_lower_labels * label_height
    
    # 设置y轴范围，为上下标签留出空间
    plt.ylim(min_y - lower_space, max_y + upper_space)
    
    # 绘制原始数据曲线（在中间）
    plt.plot(range(data_len), raw_data[column], label='Original Data', color='blue', linewidth=1.5)

    # 统一处理所有标签（算法标签 + 人工标签）
    # 第一个算法：绿色，在上方
    # 第二个算法：红色，在下方
    # 人工标签：如果有，作为第三个标签处理
    
    # 构建统一的标签列表
    all_labels = []
    for method_name, algorithm_label in algorithm_labels:
        all_labels.append((method_name, algorithm_label))
    
    # 如果有人工标签，也加入列表
    if has_manual_label:
        all_labels.append(('Manual Label', manual_label_col))
    
    # 定义颜色：第一个绿色，第二个红色，后续使用其他颜色
    label_colors = ['green', 'red', 'orange', 'purple', 'cyan', 'magenta']
    
    # 绘制所有标签
    for i, (label_name, label_data) in enumerate(all_labels):
        label_drawn = False
        color = label_colors[i % len(label_colors)]
        
        # 计算标签位置：第一个在上方，第二个在下方，后续交替
        if i == 0:
            # 第一个标签：在原始数据上方（绿色）
            y1 = max_y + label_height * 0.3
            y2 = y1 + label_height
        elif i == 1:
            # 第二个标签：在原始数据下方（红色）
            y1 = min_y - label_height * 1.3
            y2 = y1 + label_height
        else:
            # 后续标签：交替放置
            if i % 2 == 0:
                # 偶数索引在上方
                y1 = max_y + label_height * 0.3 + ((i - 2) // 2 + 1) * label_height
                y2 = y1 + label_height
            else:
                # 奇数索引在下方
                y1 = min_y - label_height * 1.3 - ((i - 3) // 2 + 1) * label_height
                y2 = y1 + label_height
        
        # 如果标签数据为空，仍然在图例中显示算法名称（用空数据）
        if label_data.empty:
            # 创建一个不可见的填充区域，只是为了在图例中显示算法名称
            plt.fill_between([], [], [], color=color, alpha=0.5, label=label_name)
            label_drawn = True
        else:
            # 有数据时正常绘制
            for idx, row in label_data.iterrows():
                # 找到最接近的索引位置
                start_time = pd.to_datetime(row['开始时间']).tz_localize(None)
                end_time = pd.to_datetime(row['结束时间']).tz_localize(None)
                start_idx = raw_data.index.get_indexer([start_time], method='nearest')[0]
                end_idx = raw_data.index.get_indexer([end_time], method='nearest')[0]
                
                # 扩展异常区域：前后各扩展数据长度的0.1%，最小扩展1个点
                data_len = len(raw_data)
                expand_size = max(1, int(data_len * 0.001))  # 0.1%的数据长度，最小1个点
                start_idx_expanded = max(0, start_idx - expand_size)
                end_idx_expanded = min(data_len - 1, end_idx + expand_size)
                
                # 增加y轴方向的宽度，使异常区域更明显
                y_height = y2 - y1
                y_expand = y_height * 0.2  # 增加20%的高度
                y1_expanded = y1 - y_expand
                y2_expanded = y2 + y_expand
                
                plt.fill_between(
                    range(start_idx_expanded, end_idx_expanded + 1),
                    y1=y1_expanded,
                    y2=y2_expanded,
                    color=color,
                    alpha=0.5,
                    label=label_name if not label_drawn else ""
                )
                label_drawn = True

    # 手动设置横坐标标签
    xticks = np.linspace(0, data_len - 1, num=10, dtype=int)
    xtick_labels = raw_data.index[xticks].to_pydatetime()
    plt.xticks(xticks, [label.strftime('%Y-%m-%d %H:%M:%S') for label in xtick_labels], rotation=45, ha='right')

    plt.legend(loc="upper right")
    method_names = ', '.join([m[0] for m in algorithm_labels])
    plt.title(f'Sensor ID: {column} Multi-Algorithm ({method_names}) vs Manual Labels')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    # 保存图像
    # 确保 fig_path 是目录路径，根据对比算法名称创建子目录
    if os.path.isfile(fig_path):
        fig_dir = os.path.dirname(fig_path)
    else:
        # 根据算法名称创建子目录，如 combined/adtk_hbos_vs_chatts
        method_names_sorted = sorted([m[0] for m in algorithm_labels])
        method_subdir = '_vs_'.join(method_names_sorted)
        fig_dir = os.path.join(fig_path, 'combined', method_subdir)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    method_str = '_'.join([m[0] for m in algorithm_labels])
    total_alg_labels = sum([len(m[1]) for m in algorithm_labels])
    manual_count = len(manual_label_col) if not manual_label_col.empty else 0
    
    # 新增：统计每个算法的标签数量，并按照算法顺序拼接
    method_counts_parts = []
    for method_name, label_df in algorithm_labels:
        method_counts_parts.append(f"{method_name}{len(label_df)}")
    method_counts_str = '_'.join(method_counts_parts)
    
    # 手工标签计数信息
    manual_part = f"_manual{manual_count}" if has_manual_label else ""
    
    fig_file_path = os.path.join(fig_dir, f"{total_alg_labels}_{mask}_{raw_data.shape[0]}_{column}_{method_counts_str}{manual_part}.png")
    plt.savefig(fig_file_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f'图像已保存到: {fig_file_path}')   
    
    
    
    
def main(
    # point_config_path : str = 'test_points_config.json',
    #      methods: Union[List[str], str, None] = 'wavelet',
    #      label_path = '/opt/results/test',
    #      global_path = '/opt/results/global',
         ):
    
    # point_config = load_config(point_config_path)
    
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Combine outlier labels remain start and end time')
    parser.add_argument('--task_name', type=str, default='local', help='global or local')
    parser.add_argument('--fig_type', type=str, default='focus', help='type of fig: combined, focus, or normal')
    parser.add_argument('--config_file', type=str, default='test_points_config_gold.json', help='config file')
    parser.add_argument('--fig_path', type=str, default='./figs', help='save path')
    parser.add_argument('--label_path', type=str, default=None, help='label path for combined type')
    parser.add_argument('--data_path', type=str, default='./data', help='data path for global data')
    parser.add_argument('--method', type=str, default='stl_wavelet', help='method name for normal type')
    parser.add_argument('--methods', type=str, default=None, help='comma-separated methods for combined type')
    parser.add_argument('--n_jobs', type=int, default=12, help='number of processes')
    parser.add_argument('--manual_label_path', type=str, default=None, help='path to manual label CSV file')
    parser.add_argument('--use_direct_data', action='store_true', help='直接使用算法结果数据，跳过标签文件查找（combined类型）')
    
    args = parser.parse_args()
    
    print(args)
    
    # save_args_to_file(args, script_name=os.path.basename(__file__), save_dir= args.fig_path,  file_prefix=f"{args.task_name}_params")
    save_args_to_file(args, script_name = os.path.basename(__file__), save_dir= './params', file_prefix=f"{args.task_name}_params")   
    
    point_config = load_config(args.config_file)
    n_jobs = args.n_jobs
    print(f"使用 {n_jobs} 个进程并行处理任务...")
    
    columns_to_process = []
    sensor_infos = []
    for path, sensor_variables in point_config.items():
        print(f"Processing path: {path}")
        print(f"Sensor variables: {sensor_variables}")
        columns_to_process.extend(sensor_variables['columns'])
        columns = sensor_variables['columns']
        st = sensor_variables['st']
        et = sensor_variables['et']
        for column in columns:
            sensor_infos.append((path, column, st, et))
    
    fig_path = os.path.join(args.fig_path, args.task_name)
    
    # 解析 methods 参数（如果是字符串）
    methods_list = None
    if args.methods:
        methods_list = [m.strip() for m in args.methods.split(',')]
    
    if args.fig_type == 'combined':    
        # 加载人工标签数据（使用新的统一加载函数）
        manual_label_data = load_manual_label_data(args.manual_label_path)
        if manual_label_data.empty and args.manual_label_path:
            print(f"警告: 未能加载人工标签文件或文件为空: {args.manual_label_path}")
        elif manual_label_data.empty:
            print("未提供人工标签文件路径，将仅使用算法标签进行对比")
        
        # 使用 label_path 或默认使用 fig_path
        algorithm_label_path = args.label_path if args.label_path else fig_path
    
        # 构建 data_path，用于直接从算法结果CSV文件读取数据
        global_data_path = os.path.join(args.data_path, args.task_name) if args.data_path else None
        
        # 根据 task_name 选择 mask
        selected_mask = 'global_mask' if args.task_name == 'global' else 'local_mask'
        with Pool(processes=n_jobs) as pool:
            print(f'type of fig: {args.fig_type}')
            if methods_list and len(methods_list) > 1:
                print(f"使用多算法对比模式: {methods_list}")
            print(f"算法标签路径: {algorithm_label_path}")
            print(f"数据路径: {global_data_path}")
            print(f"直接使用算法结果数据: {args.use_direct_data}")
            print(f"Processing {len(columns_to_process)} columns...")
            pool.starmap(process_column_with_labels, [(info, algorithm_label_path, manual_label_data, fig_path, selected_mask, methods_list, global_data_path, args.use_direct_data) for info in sensor_infos])
            
    elif args.fig_type == 'focus':
        print('执行 process_column_local')
        # 根据 task_name 选择 mask
        selected_mask = 'global_mask' if args.task_name == 'global' else 'local_mask'
        with Pool(processes=n_jobs) as pool:
            print(f"Processing {len(columns_to_process)} columns...")
            pool.starmap(process_column_local, [(info, fig_path, selected_mask) for info in sensor_infos])
    
    elif args.fig_type == 'normal':
        print('执行 process_column_normal (单算法结果图)')
        if not args.data_path:
            print("错误: normal 类型需要提供 data_path 参数")
            return
        
        # 根据 task_name 选择 mask，normal 直接从算法结果 CSV 使用对应掩码列
        selected_mask = 'global_mask' if args.task_name == 'global' else 'local_mask'
        with Pool(processes=n_jobs) as pool:
            print(f"Processing {len(columns_to_process)} columns with method: {args.method}")
            pool.starmap(process_column_normal, [(info, args.data_path, args.method, fig_path, selected_mask) for info in sensor_infos])
        
    
    
    
    # if isinstance(methods, list):
        
    #     if len(methods) >1:
    #         # 多标签绘图
    #         with Pool(processes=n_jobs) as pool:
    #             pool.starmap(process_combined_column, [(col, global_path, methods) for col in columns_to_process])
                
    #     else:
    #         # 单标签绘图
    #         with Pool(processes=n_jobs) as pool:
    #             pool.starmap(process_column_global, [(col, global_path, None, methods) for col in columns_to_process])
    # # elif isinstance(methods, str):
    # #     # 单标签绘图
    # #     with Pool(processes=n_jobs) as pool:
    # #         pool.starmap(process_column_global, [(col, global_path, None, methods) for col in columns_to_process])
            
    # elif methods is None:
    #     print('执行 process_column_local')
    #     with Pool(processes=n_jobs) as pool:
    #         print(f"Processing {len(columns_to_process)} columns...")
    #         pool.starmap(process_column_local, [(info, global_path, label_path) for info in sensor_infos])
            
    # elif method == 'combined':
    #     with Pool(processes=n_jobs) as pool:
    #         print(f'method: {method}')
    #         print(f"Processing {len(columns_to_process)} columns...")
    #         pool.starmap(process_column_with_labels, [(info, label_path, manual_label_data, label_path) for info in sensor_infos])
    
    print(f"任务完成，总用时: {time.time() - start_time} 秒")
    
    
if __name__ == '__main__':
    # import os
    # excute_time = time.strftime("%Y-%m-%d", time.localtime())
    
    # point_config_path = 'test_points_config.json'
    
    # methods = ['combined']  
    # for method in methods: 
    #     label_path = f'/opt/results/test_{method}_{excute_time}'
    #     main(point_config_path, method, label_path, n_jobs=12)
    # main(method = 'wavelet',n_jobs=4)
    
    main()
