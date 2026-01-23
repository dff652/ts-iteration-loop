"""
@File    :   run.py
@Time    :   2024/10/30
@Author  :   DouFengfeng
@Version :   1.0.0
@Contact :   ff.dou@cyber-insight.com
@License :   (C)Copyright 2019-2026, CyberInsight
@Desc    :   数据质量指标计算
@Update  :   DouFengfeng, 2024/12/26

"""

# ========== CUDA 环境变量处理（必须在所有 import 之前） ==========
# 只清除可能存在的空字符串值，不要显式设置 CUDA_VISIBLE_DEVICES
# 设置 CUDA_VISIBLE_DEVICES=0,1 反而会在某些 PyTorch 版本中触发 bug
import os
_cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if _cuda_env is not None and _cuda_env == '':
    # 删除空字符串值，让 PyTorch 自动检测所有 GPU
    del os.environ['CUDA_VISIBLE_DEVICES']
# ================================================================

from iotdb.Session import Session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from tqdm import tqdm
from scipy.ndimage import label
from signal_processing import ts_downsample, stl_decompose, read_iotdb, check_time_continuity, \
    get_fulldata,min_max_scaling,calculate_group_median_variance,variance_filter, is_noisy_data, is_step_data, \
    adaptive_downsample,calculate_sampling_rate
from ensemble import replace_outliers_by_3sigma, create_outlier_mask, iqr_find_anomaly_indices, \
    nsigma_find_anomaly_indices, piecewise_linear, standardized_find_anomaly_indices
from wavelet import reconstruct_residuals, split_continuous_outliers, range_split_outliers, cv_sort_local_outlier, \
    refine_local_outliers, combine_local_outliers, exclude_indices, fit_and_replace_outliers
from morphological import morphological_gradient
from isolation_forest import detect_isolation_forest
from scipy import signal
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
import argparse
import time
import json
from multiprocessing import Pool
import os
import functools
from tsdownsample import MinMaxLTTBDownsampler, M4Downsampler
from statsmodels.tsa.seasonal import STL

# 引入基于 ADTK + HBOS 的检测方法
from jm_detect import adtk_hbos_detect

# 引入 ChatTS 检测方法（懒加载，避免无 GPU 时报错）
def get_chatts_detect():
    from chatts_detect import chatts_detect
    return chatts_detect

# 引入 Timer 检测方法（懒加载，避免无 GPU 时报错）
def get_timer_detect():
    from timer_detect import timer_detect
    return timer_detect

from config import load_args, save_args_to_file

# 导入性能跟踪日志模块
from perf_logger import PerfLogger, PerfMetrics, Timer, get_perf_logger


print("当前工作目录:", os.getcwd())
import logging

logging.basicConfig(
    filename="./function_timer.log",  # 日志文件名
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(message)s"  # 日志格式
)

def log_data_info(title, data, log_level=logging.INFO):
    """
    记录数据的基本信息到日志中，支持自定义日志级别。
    
    参数：
        title (str): 日志标题。
        data (pd.DataFrame): 要记录的信息数据。
        log_level (int): 日志级别（默认：logging.INFO）。
    """
    if data.empty:
        logging.log(log_level, '\n' + '=' * 30 + f' {title} ' + '=' * 30)
        logging.log(log_level, '数据量：0')
        logging.log(log_level, '数据时间范围：无数据')
    else:
        logging.log(log_level, '\n' + '=' * 30 + f' {title} {data.columns[0]} ' + '=' * 30)
        logging.log(log_level, f'数据量：{data.shape}')
        logging.log(log_level, f'数据时间范围：{data.index[0]} - {data.index[-1]}')


def timer_decorator(func):
    """装饰器用于记录函数运行时间"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"Function {func.__name__!r} took {run_time:.4f} seconds")
        return result

    return wrapper_timer


def load_config(config_filename):
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的绝对路径
    config_path = os.path.join(script_dir, 'configs', config_filename)

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def fetch_data(sensor_info):
    path, columns, st, et = sensor_info
    data = read_iotdb(path=path, target_column=columns, st=st, et=et)
    return data


def dead_value_detection(data, duration_threshold=3600, distinct_threshold=10):
    import numpy as np
    import pandas as pd

    columns_to_check = data.columns
    result_dfs = []
    constant_indices = {}

    for col in columns_to_check:
        # 检查列的唯一值数量
        # value_len = data[col].nunique()
        value_len = check_state(data, col)
        if value_len < distinct_threshold:
            temp_df = pd.DataFrame({
                'constant_rate': [0.0],
                'constant_count': [0]
            }).transpose()
            result_dfs.append(temp_df)
            constant_indices[col] = np.zeros(len(data), dtype=bool)
            continue

        # 计算变化点分组
        df = data.copy()
        df['Group'] = (df[col] != df[col].shift()).cumsum()

        df['t'] = df.index
        result = df.groupby('Group').agg(Start=('t', 'first'), End=('t', 'last'), Value=(col, 'first'))
        result['Duration'] = (result['End'] - result['Start']).dt.total_seconds()
        filtered_result = result[result['Duration'] > duration_threshold]
        
        # 计算每组的持续时间
        # durations = df.groupby('Group').size() * (data.index[1] - data.index[0]).total_seconds()
        # filtered_groups = durations[durations > duration_threshold].index

        # 直接生成掩码
        mask = df['Group'].isin(filtered_result.index).values
        # 从死值中选择0值
        mask = mask & (data[col].values == 0)

        count_constant = mask.sum()
        count_total = len(data)
        data_quality_rate = count_constant / count_total

        # 结果存储
        temp_df = pd.DataFrame({
            'constant_rate': [data_quality_rate],
            'constant_count': [count_constant]
        }).transpose()
        result_dfs.append(temp_df)

        constant_indices[col] = mask

    # 合并结果
    result_df = pd.concat(result_dfs, axis=1)
    result_df.columns = columns_to_check

    # 常值索引布尔 DataFrame
    constant_indices = pd.DataFrame(constant_indices, index=data.index)
    print(f'sum of constant_indices={constant_indices.sum()}')
    # 提取常值数据
    constant_data = data.loc[constant_indices.any(axis=1), columns_to_check]

    return result_df, constant_indices, constant_data

def check_state(data, col):
    values_unique = data[col].unique()
    return len(values_unique)


def get_true_indices(data):

    # 重置索引，确保索引是连续的自然索引
    df_reset = data.reset_index(drop=True)
    
    # 找到值为 True 的行
    true_rows = df_reset[df_reset[df_reset.columns[0]] == True]
    
    # 获取自然索引
    natural_indices = true_rows.index.tolist()
    
    return natural_indices


def sigma_filtered_and_anomaly_detection(data, method='piecewise_linear', th=1.5, N=0.01):
    # 合并的数据长度取中数据长度的1/100或1/200
    if data.empty:
        print("Received empty data for processing.")
        return None, None, None, None
    data_length = len(data)
    if N >= 1:
        merge_len = N
    else:
        print('----测试合并长度用百分比，计算合并长度')
        merge_len = int(N * data_length)

    if method == 'piecewise_linear':
        # 预处理
        # pre_data = replace_outliers_by_3sigma(data)

        window_length = min(601, data_length if data_length % 2 == 1 else data_length - 1)
        polyorder = min(3, window_length - 1)
        pre_data = savgol_filter(data.values.ravel(), window_length=window_length,
                                 polyorder=polyorder)  # 使用SG滤波，输入需要是array

        # 回归
        result = piecewise_linear(pre_data)
        if result is None or len(result) != 2:
            print("回归后结果为空，返回对应的mask全为0。")
            return np.zeros(data.shape[0]), []
        
        reconstruct_data, residuals = result
        
        if reconstruct_data is None or residuals is None:
            print("回归后结果为空，返回对应的mask全为0。")
            return np.zeros(data.shape[0]), []

        anomaly_indices = iqr_find_anomaly_indices(residuals, th, merge_len)  # 默认th=1.5

    elif method == 'wavelet':
        # 小波包
        min_max_data = min_max_scaling(data.values.ravel())
        # window_length = min(601, data_length if data_length % 2 == 1 else data_length - 1)
        # polyorder = min(3, window_length - 1)

        # pre_data = savgol_filter(min_max_data, window_length=window_length,
        #                         polyorder=polyorder)  # 使用SG滤波，输入需要是array
        reconstruct_data = reconstruct_residuals(min_max_data.ravel(), wavelet='db1', level=3)
        anomaly_indices = nsigma_find_anomaly_indices(reconstruct_data, th, merge_len)  # 默认th=5

        pre_data = None

    elif method == 'cv':
        min_max_data = min_max_scaling(data.values.ravel())
        reconstruct_data = morphological_gradient(min_max_data.ravel(), 4)
        anomaly_indices = nsigma_find_anomaly_indices(reconstruct_data, th, merge_len)
        pre_data = None


    elif method == 'standardized':
        detrended_data = signal.detrend(data.values.ravel(), type='linear')
        anomaly_indices = standardized_find_anomaly_indices(detrended_data, th, merge_len)
        reconstruct_data = None
        pre_data = None

    elif method == 'iforest':
        min_max_data = min_max_scaling(data.values.ravel())
        window_length = min(61, data_length if data_length % 2 == 1 else data_length - 1)
        polyorder = min(1, window_length - 1)

        pre_data = savgol_filter(min_max_data, window_length=window_length,
                                 polyorder=polyorder)  # 使用SG滤波，输入需要是array
        reconstruct_data = reconstruct_residuals(pre_data.ravel(), wavelet='db1', level=3)
        anomaly_indices = detect_isolation_forest(data, outlier_ratio=th, N=merge_len)
        reconstruct_data = None
        pre_data = None

    # 确保anomaly_indices是有效的
    if anomaly_indices is None or len(anomaly_indices) == 0:
        anomaly_indices = []
    
    outlier_mask = create_outlier_mask(data, anomaly_indices)

    # 异常点拆分
    # anomaly_group = split_continuous_outliers(anomaly_indices)
    # global_indices, local_indices = range_split_outliers(data, anomaly_group, 0.1) # 返回的是全局异常和局部异常
    # global_mask = create_outlier_mask(data, global_indices)
    # local_mask = create_outlier_mask(data, local_indices)
    print(f'---***---异常点数:{len(anomaly_indices)},数据长度:{len(data)},合并长度:{merge_len},阈值:{th} ---***---')
    return outlier_mask, anomaly_indices


# def recursive_detect_outliers(data, epoch=3, method='piecewise_linear', th=1.5, N=600):
#     data.index = pd.to_datetime(data.index, errors='coerce')
#     data = data[data.index.notnull()]

#     raw = pd.DataFrame()
#     outlier_mask, anomaly_indices, reconstruct_data, pre_data = sigma_filtered_and_anomaly_detection(data, method, th,
#                                                                                                      N)
#     adj_thres = 1 / 500 * data.shape[0]

#     data['mask'] = outlier_mask
#     data['Group'] = (data['mask'] != data['mask'].shift()).cumsum()

#     pass

@timer_decorator
def calculate_outliers(df_trend, df_seasonal, df_resid, data, method, threshold, num_points, ratio_classify,
                       use_trend=False, use_seasonal=False, use_resid=False, use_clustering=False,
                       clustering_method='kmeans', clustering_n_clusters=2):
    """
    根据用户选择的分量（趋势、季节性、残差）来计算异常。

    参数:
    - df_trend: 趋势数据
    - df_seasonal: 季节性数据
    - df_resid: 残差数据
    - data: 原始数据
    - method: 用于异常检测的方法
    - threshold: 异常检测的阈值
    - num_points: 计算的点数
    - ratio_classify: 分类比例
    - use_trend: 是否使用趋势分量
    - use_seasonal: 是否使用季节性分量
    - use_resid: 是否使用残差分量
    - use_clustering: 是否使用聚类算法进行异常值分割
    - clustering_method: 聚类方法 ('kmeans', 'dbscan', 'hierarchical')
    - clustering_n_clusters: 聚类数量

    返回:
    - global_indices: 全局异常索引（固定阈值方法）
    - local_indices: 局部异常索引（固定阈值方法）
    - outlier_mask: 异常掩码
    - global_indices_cluster: 全局异常索引（聚类算法方法）
    - local_indices_cluster: 局部异常索引（聚类算法方法）
    """
    outlier_mask = None
    combined_indices = []
    data_length = len(data)

    if use_trend:
        trend_outlier_mask, trend_indices = sigma_filtered_and_anomaly_detection(df_trend, method, threshold,
                                                                                 num_points)
        if outlier_mask is None:
            outlier_mask = trend_outlier_mask
        else:
            # # 确保两个掩码都是数组且形状相同
            # if isinstance(trend_outlier_mask, np.ndarray) and isinstance(outlier_mask, np.ndarray):
            #     outlier_mask += trend_outlier_mask
            # else:
            #     # 如果其中一个不是数组，重新创建
            #     outlier_mask = np.zeros(data_length, dtype=int)
            #     if isinstance(trend_outlier_mask, np.ndarray):
            #         outlier_mask += trend_outlier_mask
            
            outlier_mask += trend_outlier_mask       
         
        combined_indices.extend(trend_indices)

    if use_seasonal:
        seasonal_outlier_mask, seasonal_indices = sigma_filtered_and_anomaly_detection(df_seasonal, method, threshold,
                                                                                       num_points)
        if outlier_mask is None:
            outlier_mask = seasonal_outlier_mask
        else:
            # # 确保两个掩码都是数组且形状相同
            # if isinstance(seasonal_outlier_mask, np.ndarray) and isinstance(outlier_mask, np.ndarray):
            #     outlier_mask += seasonal_outlier_mask
            # else:
            #     # 如果其中一个不是数组，重新创建
            #     outlier_mask = np.zeros(data_length, dtype=int)
            #     if isinstance(seasonal_outlier_mask, np.ndarray):
            #         outlier_mask += seasonal_outlier_mask
            
            outlier_mask += seasonal_outlier_mask
            
        combined_indices.extend(seasonal_indices)

    if use_resid:
        resid_outlier_mask, resid_indices = sigma_filtered_and_anomaly_detection(df_resid, method, threshold,
                                                                                 num_points)
        if outlier_mask is None:
            outlier_mask = resid_outlier_mask
        else:
            # # 确保两个掩码都是数组且形状相同
            # if isinstance(resid_outlier_mask, np.ndarray) and isinstance(outlier_mask, np.ndarray):
            #     outlier_mask += resid_outlier_mask
            # else:
            #     # 如果其中一个不是数组，重新创建
            #     outlier_mask = np.zeros(data_length, dtype=int)
            #     if isinstance(resid_outlier_mask, np.ndarray):
            #         outlier_mask += resid_outlier_mask
            
            outlier_mask += resid_outlier_mask           
                    
        combined_indices.extend(resid_indices)

    # # 确保outlier_mask是正确形状的数组
    # if outlier_mask is None:
    #     outlier_mask = np.zeros(data_length, dtype=int)
    # elif not isinstance(outlier_mask, np.ndarray):
    #     outlier_mask = np.zeros(data_length, dtype=int)
    # elif outlier_mask.shape[0] != data_length:
    #     outlier_mask = np.zeros(data_length, dtype=int)
    
    # 合并死值检测的结果
    _,dead_df,_ = dead_value_detection(data,72000)
    
    natural_indices = get_true_indices(dead_df)

    unique_indices = np.unique(combined_indices)
    anomaly_group = split_continuous_outliers(unique_indices)

    # 使用 range_split_outliers 方法将异常分为全局异常和局部异常（固定阈值方法）
    global_indices, local_indices = range_split_outliers(data, anomaly_group, ratio_classify)
    
    # 将所有异常索引合并，并去重
    global_list = global_indices.tolist()
    global_list.extend(natural_indices)
    unique_global = np.unique(global_list)
    
    # 如果启用聚类算法，也使用聚类方法进行分割
    global_indices_cluster = unique_global.copy()
    local_indices_cluster = local_indices.copy()
    
    if use_clustering and len(anomaly_group) > 1:
        try:
            from wavelet import adaptive_outlier_split
            global_indices_cluster, local_indices_cluster = adaptive_outlier_split(
                data, anomaly_group, method='cluster',
                n_clusters=clustering_n_clusters,
                cluster_method=clustering_method
            )
            # 同样合并死值检测结果
            global_list_cluster = global_indices_cluster.tolist()
            global_list_cluster.extend(natural_indices)
            global_indices_cluster = np.unique(global_list_cluster)
        except Exception as e:
            print(f"聚类算法执行失败，使用固定阈值方法: {e}")
            global_indices_cluster = unique_global
            local_indices_cluster = local_indices
    
    return unique_global, local_indices, outlier_mask, global_indices_cluster, local_indices_cluster


def save_results(results, sensor_info, args, position_index=None):
    # 创建对应的文件夹
    print("当前工作目录:", os.getcwd())
    # 支持按方法名保存到子目录
    if getattr(args, 'save_by_method', False):
        folder_path = os.path.join(args.data_path, args.task_name, args.method)
    else:
        folder_path = os.path.join(args.data_path, args.task_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 获取点位名称和日期信息
    path, column, st, et = sensor_info
    date_str = pd.to_datetime(st).strftime('%Y%m%d')  # 格式化日期

    print(f"Saving results to: {folder_path}")

    components = []

    # 添加各个条件下需要的组件
    if args.use_trend:
        components.append("trend")
    if args.use_seasonal:
        components.append("seasonal")
    if args.use_resid:
        components.append("resid")

    # 构建文件名
    if components:
        component_str = "_".join(components)  # 将使用的分量组合起来
    else:
        component_str = "results"  # 如果没有任何分量，使用默认名称

    # 根据保存模式处理数据
    df_to_save = results
    if getattr(args, 'save_mode', 'raw') == 'downsample':
        if position_index is None:
            print("[保存警告] save_mode=downsample 但未提供 position_index，按原始数据保存。")
        else:
            # 如果传入的结果已是降采样后的（长度与 position_index 一致），则仅附加原始位置
            if len(results) == len(position_index):
                df_to_save = results.copy()
                df_to_save['orig_pos'] = position_index
            else:
                # 否则按原始数据长度进行截取
                df_to_save = results.iloc[position_index].copy()
                df_to_save['orig_pos'] = position_index

    # 压缩数据类型以减少存储体积（可节省约 80% 空间）
    # 数值列转换为 float32，掩码列转换为 uint8
    mask_columns = ['outlier_mask', 'global_mask', 'local_mask', 'global_mask_cluster', 'local_mask_cluster']
    for col in df_to_save.columns:
        if col in mask_columns:
            try:
                df_to_save[col] = df_to_save[col].astype(np.uint8)
            except Exception:
                pass
        elif col not in ['orig_pos'] and df_to_save[col].dtype in ['float64', 'int64']:
            try:
                df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce').astype(np.float32)
            except Exception:
                pass

    # 只保存一个完整的CSV文件，包含所有信息
    file_name = f"{args.task_name}_{args.method}_{args.downsampler}_{args.ratio}_{args.num_points}_{column}_{date_str}_{component_str}.csv"
    df_to_save.to_csv(os.path.join(folder_path, file_name))
    print(f"保存结果: {file_name}")
    
    # 如果启用了聚类算法，在文件名中标识
    if args.use_clustering:
        print(f"注意: 结果包含固定阈值方法和{args.clustering_method}聚类方法的对比数据")
        print(f"列名说明:")
        print(f"  - global_mask: 固定阈值方法的全局异常掩码")
        print(f"  - local_mask: 固定阈值方法的局部异常掩码")
        print(f"  - global_mask_cluster: {args.clustering_method}聚类方法的全局异常掩码")
        print(f"  - local_mask_cluster: {args.clustering_method}聚类方法的局部异常掩码")


def post_process(data, indices, threshold=0.2):
    split_refine = split_continuous_outliers(indices)
    cv_values, cv_indices = cv_sort_local_outlier(data.values.ravel(), split_refine, threshold)
    print('---异常合并---')
    group = split_continuous_outliers(cv_indices)
    refine = combine_local_outliers(data.values.ravel(), group)

    mask = create_outlier_mask(data, refine)

    return mask


@timer_decorator
def process_sensor(sensor_info, args):
    """
    处理单个传感器点位的异常检测，带有全面的性能跟踪日志
    """
    # === 初始化性能日志器和计时 ===
    perf_logger = get_perf_logger()
    metrics = PerfMetrics()
    total_start_time = time.perf_counter()
    
    # 1. 初始化和检查是否已存在结果
    path, column, st, et = sensor_info
    print(f"Processing in process ID: {os.getpid()} for {sensor_info}")
    logging.info(f"=== Start processing sensor: {sensor_info} ===")
    
    # 记录点位信息
    metrics.sensor_path = path
    metrics.sensor_column = column
    metrics.start_time = st
    metrics.end_time = et
    metrics.process_id = os.getpid()
    metrics.method = args.method
    metrics.downsampler = args.downsampler
    
    # 确定使用的设备
    if args.method == 'chatts':
        metrics.device = args.chatts_device
    elif args.method == 'timer':
        metrics.device = args.timer_device
    else:
        metrics.device = getattr(args, 'device', 'cpu')
    
    # 检查结果文件是否已存在（支持方法子目录）
    if getattr(args, 'save_by_method', False):
        folder_path = os.path.join(args.data_path, args.task_name, args.method)
    else:
        folder_path = os.path.join(args.data_path, args.task_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 构建文件名
    date_str = pd.to_datetime(st).strftime('%Y%m%d')
    components = []
    if args.use_trend: components.append("trend")
    if args.use_seasonal: components.append("seasonal")
    if args.use_resid: components.append("resid")
    component_str = "_".join(components) if components else "results"
    file_name = f"{args.task_name}_{args.method}_{args.downsampler}_{args.ratio}_{args.num_points}_{column}_{date_str}_{component_str}.csv"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        logging.info(f"Results already exist for {sensor_info}, skipping processing")
        print(f"Results already exist for {sensor_info}, skipping processing")
        metrics.status = "skipped"
        metrics.total_time = time.perf_counter() - total_start_time
        perf_logger.log_metrics(metrics)
        return None

    # 2. 数据读取与预处理
    with Timer("data_read") as t_read:
        if hasattr(args, 'input') and args.input:
            print(f"Reading from CSV: {args.input}")
            try:
                raw_data = pd.read_csv(args.input)
                # 尝试解析时间列
                time_col = None
                for col in ['date', 'time', 'timestamp', 'Time', 'Date']:
                    if col in raw_data.columns:
                        time_col = col
                        break
                
                if time_col:
                    raw_data[time_col] = pd.to_datetime(raw_data[time_col])
                    raw_data.set_index(time_col, inplace=True)
                    # 确保索引排序
                    raw_data.sort_index(inplace=True)
                else:
                    # 如果没有时间列，尝试使用推断频率或生成索引
                    print("Warning: No time column found, using auto-generated index")
                
                # 如果指定了列名但不在数据中，尝试使用第一列
                if column not in raw_data.columns and not raw_data.empty:
                    # 排除时间列（如果它是列而不是索引）
                    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        print(f"Column '{column}' not found, using '{numeric_cols[0]}'")
                        column = numeric_cols[0]
                        # 更新 sensor_info 以反映实际使用的列名
                        sensor_info = (path, column, st, et)
            
            except Exception as e:
                print(f"Error reading CSV: {e}")
                metrics.status = "failed"
                metrics.error_message = f"CSV read error: {str(e)}"
                perf_logger.log_metrics(metrics)
                return None
        else:
            raw_data = read_iotdb(path=path, target_column=column, st=st, et=et)
    metrics.data_read_time = t_read.elapsed
    
    print(f"Data shape: {raw_data.shape}")
    metrics.raw_data_length = len(raw_data)
    
    if raw_data.empty:
        print(f"No data for {sensor_info} - column: {column}")
        metrics.status = "failed"
        metrics.error_message = "No data available"
        metrics.total_time = time.perf_counter() - total_start_time
        perf_logger.log_metrics(metrics)
        return None

    # 分支：使用 adtk_hbos 方法时，直接调用 jm_detect.adtk_hbos_detect
    if args.method == 'adtk_hbos':
        # adtk_hbos_detect 内部完成降采样、死值检测与区段映射，返回与原始数据对齐的布尔列
        try:
            point_name = column
            input_df = pd.DataFrame({point_name: raw_data[point_name]})
            
            with Timer("anomaly_detect") as t_detect:
                outlier_result, position_index_ds = adtk_hbos_detect(
                    data=input_df,
                    downsampler=args.downsampler,
                    sample_param=args.ratio,
                    bin_nums=args.bin_nums,
                    min_threshold=args.min_threshold,
                    ratio=args.hbos_ratio if hasattr(args, 'hbos_ratio') else None
                )
            metrics.anomaly_detect_time = t_detect.elapsed

            # 将检测结果（原始长度掩码）叠加回原始数据，并按需要的保存模式落盘
            data = raw_data.copy()
            global_mask = outlier_result[point_name].astype(int).values
            local_mask = np.zeros_like(global_mask)
            global_mask_cluster = np.zeros_like(global_mask)
            local_mask_cluster = np.zeros_like(global_mask)
            outlier_mask = global_mask

            data['outlier_mask'] = outlier_mask
            data['global_mask'] = global_mask
            data['local_mask'] = local_mask
            data['global_mask_cluster'] = global_mask_cluster
            data['local_mask_cluster'] = local_mask_cluster

            # 记录检测结果统计
            metrics.downsampled_data_length = len(position_index_ds) if position_index_ds is not None else metrics.raw_data_length
            metrics.downsample_ratio = metrics.downsampled_data_length / metrics.raw_data_length if metrics.raw_data_length > 0 else 0
            metrics.anomaly_count = int(np.sum(global_mask))
            metrics.anomaly_ratio = (metrics.anomaly_count / len(global_mask) * 100) if len(global_mask) > 0 else 0
            metrics.global_anomaly_count = int(np.sum(global_mask))
            metrics.local_anomaly_count = 0

            # 保存结果
            with Timer("save") as t_save:
                print(f"Saving results for {sensor_info} with shape {data.shape}")
                save_results(data, sensor_info, args, position_index=position_index_ds)
            metrics.save_time = t_save.elapsed
            
            # 获取系统资源
            sys_metrics = perf_logger.get_system_metrics()
            metrics.cpu_percent = sys_metrics["cpu_percent"]
            metrics.memory_used_gb = sys_metrics["memory_used_gb"]
            metrics.memory_percent = sys_metrics["memory_percent"]
            
            metrics.status = "success"
            metrics.total_time = time.perf_counter() - total_start_time
            perf_logger.log_metrics(metrics)
            
            logging.info(f"=== Finished processing sensor (adtk_hbos): {sensor_info} ===")
            return data
        except Exception as e:
            print(f"adtk_hbos 方法执行失败，回退到常规流程: {e}")
            metrics.status = "failed"
            metrics.error_message = str(e)

    elif args.method == 'chatts':
        # ChatTS 大模型异常检测
        try:
            chatts_detect = get_chatts_detect()
            point_name = column
            input_df = pd.DataFrame({point_name: raw_data[point_name]})
            
            # 获取执行前的 GPU 状态
            gpu_metrics_before = perf_logger.get_gpu_metrics(args.chatts_device)
            
            with Timer("model_inference") as t_inference:
                # 解析 use_cache 参数
                use_cache = None
                if hasattr(args, 'chatts_use_cache') and args.chatts_use_cache is not None:
                    val = str(args.chatts_use_cache).lower()
                    if val in ['true', '1', 'yes']:
                        use_cache = True
                    elif val in ['false', '0', 'no']:
                        use_cache = False
                
                # 获取 LoRA 适配器路径（如果配置）
                lora_adapter_path = getattr(args, 'chatts_lora_adapter_path', None)
                
                # 解析 load_in_4bit 参数
                load_in_4bit_str = getattr(args, 'chatts_load_in_4bit', 'auto')
                if load_in_4bit_str == 'auto':
                    load_in_4bit = True  # 默认启用，ChatTSAnalyzer 会针对 8B 模型自动禁用
                elif load_in_4bit_str.lower() in ['true', '1', 'yes']:
                    load_in_4bit = True
                else:
                    load_in_4bit = False
                
                global_mask, anomalies, position_index_ds = chatts_detect(
                    data=input_df,
                    model_path=args.chatts_model_path,
                    device=args.chatts_device,
                    n_downsample=args.n_downsample,
                    downsampler=args.downsampler,
                    use_cache=use_cache,
                    lora_adapter_path=lora_adapter_path,
                    max_new_tokens=getattr(args, 'chatts_max_new_tokens', 4096),
                    prompt_template_name=getattr(args, 'chatts_prompt_template', 'default'),
                    load_in_4bit=load_in_4bit,
                )
            metrics.model_inference_time = t_inference.elapsed
            
            # 获取执行后的 GPU 状态
            gpu_metrics_after = perf_logger.get_gpu_metrics(args.chatts_device)
            metrics.gpu_id = gpu_metrics_after["gpu_id"]
            metrics.gpu_name = gpu_metrics_after["gpu_name"]
            metrics.gpu_utilization_percent = gpu_metrics_after["gpu_utilization_percent"]
            metrics.gpu_memory_used_mb = gpu_metrics_after["gpu_memory_used_mb"]
            metrics.gpu_memory_total_mb = gpu_metrics_after["gpu_memory_total_mb"]
            metrics.gpu_memory_percent = gpu_metrics_after["gpu_memory_percent"]
            metrics.gpu_temperature_c = gpu_metrics_after["gpu_temperature_c"]
            metrics.gpu_power_w = gpu_metrics_after["gpu_power_w"]
            
            data = raw_data.copy()
            local_mask = np.zeros_like(global_mask)
            global_mask_cluster = np.zeros_like(global_mask)
            local_mask_cluster = np.zeros_like(global_mask)
            outlier_mask = global_mask
            
            data['outlier_mask'] = outlier_mask
            data['global_mask'] = global_mask
            data['local_mask'] = local_mask
            data['global_mask_cluster'] = global_mask_cluster
            data['local_mask_cluster'] = local_mask_cluster
            
            # 记录检测结果统计
            metrics.downsampled_data_length = len(position_index_ds) if position_index_ds is not None else args.n_downsample
            metrics.downsample_ratio = metrics.downsampled_data_length / metrics.raw_data_length if metrics.raw_data_length > 0 else 0
            metrics.anomaly_count = int(np.sum(global_mask))
            metrics.anomaly_ratio = (metrics.anomaly_count / len(global_mask) * 100) if len(global_mask) > 0 else 0
            metrics.global_anomaly_count = int(np.sum(global_mask))
            metrics.local_anomaly_count = 0
            
            # 保存结果
            with Timer("save") as t_save:
                print(f"Saving results for {sensor_info} with shape {data.shape}")
                save_results(data, sensor_info, args, position_index=position_index_ds)
            metrics.save_time = t_save.elapsed
            
            # 获取系统资源
            sys_metrics = perf_logger.get_system_metrics()
            metrics.cpu_percent = sys_metrics["cpu_percent"]
            metrics.memory_used_gb = sys_metrics["memory_used_gb"]
            metrics.memory_percent = sys_metrics["memory_percent"]
            
            metrics.status = "success"
            metrics.total_time = time.perf_counter() - total_start_time
            perf_logger.log_metrics(metrics)
            
            logging.info(f"=== Finished processing sensor (chatts): {sensor_info} ===")
            return data
        except Exception as e:
            print(f"ChatTS 方法执行失败: {e}")
            import traceback
            traceback.print_exc()
            metrics.status = "failed"
            metrics.error_message = str(e)
            metrics.total_time = time.perf_counter() - total_start_time
            perf_logger.log_metrics(metrics)
            return None

    elif args.method == 'timer':
        # Timer 大模型残差异常检测
        try:
            timer_detect = get_timer_detect()
            point_name = column
            input_df = pd.DataFrame({point_name: raw_data[point_name]})
            
            # 获取执行前的 GPU 状态
            gpu_metrics_before = perf_logger.get_gpu_metrics(args.timer_device)
            
            with Timer("model_inference") as t_inference:
                global_mask, anomalies, position_index_ds = timer_detect(
                    data=input_df,
                    model_path=args.timer_model_path,
                    device=args.timer_device,
                    n_downsample=args.n_downsample,
                    downsampler=args.downsampler,
                    lookback_length=args.timer_lookback_length,
                    threshold_k=args.timer_threshold_k,
                    method=args.timer_method,
                    min_run=args.timer_min_run,
                    streaming=args.timer_streaming,
                    reset_interval=args.timer_reset_interval,
                )
            metrics.model_inference_time = t_inference.elapsed
            
            # 获取执行后的 GPU 状态
            gpu_metrics_after = perf_logger.get_gpu_metrics(args.timer_device)
            metrics.gpu_id = gpu_metrics_after["gpu_id"]
            metrics.gpu_name = gpu_metrics_after["gpu_name"]
            metrics.gpu_utilization_percent = gpu_metrics_after["gpu_utilization_percent"]
            metrics.gpu_memory_used_mb = gpu_metrics_after["gpu_memory_used_mb"]
            metrics.gpu_memory_total_mb = gpu_metrics_after["gpu_memory_total_mb"]
            metrics.gpu_memory_percent = gpu_metrics_after["gpu_memory_percent"]
            metrics.gpu_temperature_c = gpu_metrics_after["gpu_temperature_c"]
            metrics.gpu_power_w = gpu_metrics_after["gpu_power_w"]
            
            data = raw_data.copy()
            local_mask = np.zeros_like(global_mask)
            global_mask_cluster = np.zeros_like(global_mask)
            local_mask_cluster = np.zeros_like(global_mask)
            outlier_mask = global_mask
            
            data['outlier_mask'] = outlier_mask
            data['global_mask'] = global_mask
            data['local_mask'] = local_mask
            data['global_mask_cluster'] = global_mask_cluster
            data['local_mask_cluster'] = local_mask_cluster
            
            # 记录检测结果统计
            metrics.downsampled_data_length = len(position_index_ds) if position_index_ds is not None else args.n_downsample
            metrics.downsample_ratio = metrics.downsampled_data_length / metrics.raw_data_length if metrics.raw_data_length > 0 else 0
            metrics.anomaly_count = int(np.sum(global_mask))
            metrics.anomaly_ratio = (metrics.anomaly_count / len(global_mask) * 100) if len(global_mask) > 0 else 0
            metrics.global_anomaly_count = int(np.sum(global_mask))
            metrics.local_anomaly_count = 0
            
            # 保存结果
            with Timer("save") as t_save:
                print(f"Saving results for {sensor_info} with shape {data.shape}")
                save_results(data, sensor_info, args, position_index=position_index_ds)
            metrics.save_time = t_save.elapsed
            
            # 获取系统资源
            sys_metrics = perf_logger.get_system_metrics()
            metrics.cpu_percent = sys_metrics["cpu_percent"]
            metrics.memory_used_gb = sys_metrics["memory_used_gb"]
            metrics.memory_percent = sys_metrics["memory_percent"]
            
            metrics.status = "success"
            metrics.total_time = time.perf_counter() - total_start_time
            perf_logger.log_metrics(metrics)
            
            logging.info(f"=== Finished processing sensor (timer): {sensor_info} ===")
            return data
        except Exception as e:
            print(f"Timer 方法执行失败: {e}")
            import traceback
            traceback.print_exc()
            metrics.status = "failed"
            metrics.error_message = str(e)
            metrics.total_time = time.perf_counter() - total_start_time
            perf_logger.log_metrics(metrics)
            return None

    elif args.method == 'stl_wavelet':
        try:
            # 预处理阶段
            with Timer("preprocess") as t_preprocess:
                # 插入缺失值
                if args.insert_missing:
                    data = get_fulldata(raw_data, column)
                else:
                    data = raw_data.copy()
                
                # 计算采样率
                sampling_rate = calculate_sampling_rate(data[column])
                print(f"Sampling rate for {sensor_info}: {sampling_rate} Hz")
                metrics.sampling_rate_hz = sampling_rate
            metrics.preprocess_time = t_preprocess.elapsed
            
            # 3. 降采样处理 - 使用新的自适应降采样函数
            with Timer("downsample") as t_downsample:
                downsampled_data, ts, position_index = adaptive_downsample(
                    data[column], 
                    downsampler=args.downsampler,
                    sample_param=args.ratio, 
                    min_threshold=args.min_threshold
                )
            metrics.downsample_time = t_downsample.elapsed
            
            print(f"Downsampled data shape: {downsampled_data.shape}")
            metrics.downsampled_data_length = len(downsampled_data)
            metrics.downsample_ratio = metrics.downsampled_data_length / metrics.raw_data_length if metrics.raw_data_length > 0 else 0
            
            data = pd.DataFrame()
            data.index = ts
            data[column] = downsampled_data
            
            # 4. 数据类型判断 - 统一进行
            step_type = is_step_data(raw_data)
            noise_type = is_noisy_data(data[column], fs=sampling_rate)
            data_type = []
            if step_type: data_type.append("step")
            if noise_type: data_type.append("noise")
            metrics.is_step_data = step_type
            metrics.is_noisy_data = noise_type
            metrics.data_type = data_type
            del raw_data
            print(f"Data type for {sensor_info}: {data_type}")

            # 5. 初始化结果变量
            outlier_mask = np.zeros(data.shape[0])
            global_indices = []
            local_indices = []
            global_indices_cluster = []
            local_indices_cluster = []
            
            # 6. 根据数据类型决定是否进行异常检测
            if "step" not in data_type and "noise" not in data_type:
                # 6.1 STL分解(如果需要)
                if args.decompose:
                    with Timer("stl_decompose") as t_stl:
                        df_trend, df_seasonal, df_resid = stl_decompose(data[column], period=60, device=args.device)
                    metrics.stl_decompose_time = t_stl.elapsed
                    
                    # 6.2 计算异常
                    with Timer("anomaly_detect") as t_detect:
                        global_indices, local_indices, outlier_mask, global_indices_cluster, local_indices_cluster = calculate_outliers(
                            df_trend=df_trend,
                            df_seasonal=df_seasonal,
                            df_resid=df_resid,
                            data=data,
                            method=args.method,
                            threshold=args.threshold,
                            num_points=args.num_points,
                            ratio_classify=args.ratio_classify,
                            use_trend=args.use_trend,
                            use_seasonal=args.use_seasonal,
                            use_resid=args.use_resid,
                            use_clustering=args.use_clustering,
                            clustering_method=args.clustering_method,
                            clustering_n_clusters=args.clustering_n_clusters
                        )
                    metrics.anomaly_detect_time = t_detect.elapsed
                else:
                    # 不使用STL分解的情况
                    with Timer("anomaly_detect") as t_detect:
                        outlier_mask, anomaly_indices = sigma_filtered_and_anomaly_detection(
                            data, args.method, args.threshold, args.num_points)
                        anomaly_group = split_continuous_outliers(anomaly_indices)
                        
                        # 固定阈值方法
                        global_indices, local_indices = range_split_outliers(data, anomaly_group, args.ratio_classify)
                        
                        # 聚类算法方法
                        global_indices_cluster = global_indices.copy()
                        local_indices_cluster = local_indices.copy()
                        
                        if args.use_clustering and len(anomaly_group) > 1:
                            try:
                                from wavelet import adaptive_outlier_split
                                global_indices_cluster, local_indices_cluster = adaptive_outlier_split(
                                    data, anomaly_group, method='cluster',
                                    n_clusters=args.clustering_n_clusters,
                                    cluster_method=args.clustering_method
                                )
                            except Exception as e:
                                print(f"聚类算法执行失败，使用固定阈值方法: {e}")
                                global_indices_cluster = global_indices
                                local_indices_cluster = local_indices
                    metrics.anomaly_detect_time = t_detect.elapsed
                    
            # 7. 创建掩码和结果
            with Timer("postprocess") as t_postprocess:
                # 创建掩码(无论是否有异常)
                global_mask = create_outlier_mask(data, global_indices)
                local_mask = create_outlier_mask(data, local_indices)
                
                # 聚类算法的掩码
                global_mask_cluster = create_outlier_mask(data, global_indices_cluster)
                local_mask_cluster = create_outlier_mask(data, local_indices_cluster)
                
                # 添加掩码到数据
                data['outlier_mask'] = outlier_mask
                data['global_mask'] = global_mask 
                data['local_mask'] = local_mask
                
                # 聚类算法的掩码列
                data['global_mask_cluster'] = global_mask_cluster
                data['local_mask_cluster'] = local_mask_cluster
                
                # 8. 后处理(如果需要)
                if args.task_name == 'global':
                    # 检查global_mask列是否存在
                    if 'global_mask' in data.columns:
                        data = variance_filter(data, 'global_mask', args.post_filter, args.threhold_filter)
                    else:
                        print(f"警告: {sensor_info} 的 global_mask 列不存在，跳过 variance_filter")
            metrics.postprocess_time = t_postprocess.elapsed
            
            # 记录检测结果统计
            metrics.anomaly_count = int(np.sum(outlier_mask > 0)) if isinstance(outlier_mask, np.ndarray) else 0
            metrics.anomaly_ratio = (metrics.anomaly_count / len(data) * 100) if len(data) > 0 else 0
            metrics.global_anomaly_count = int(np.sum(global_mask)) if isinstance(global_mask, np.ndarray) else 0
            metrics.local_anomaly_count = int(np.sum(local_mask)) if isinstance(local_mask, np.ndarray) else 0
            
            # 9. 保存结果
            with Timer("save") as t_save:
                print(f"Saving results for {sensor_info} with shape {data.shape}")
                save_results(data, sensor_info, args, position_index=position_index)
            metrics.save_time = t_save.elapsed
            
            # 获取系统资源
            sys_metrics = perf_logger.get_system_metrics()
            metrics.cpu_percent = sys_metrics["cpu_percent"]
            metrics.memory_used_gb = sys_metrics["memory_used_gb"]
            metrics.memory_percent = sys_metrics["memory_percent"]
            
            # 如果使用了 GPU 进行 STL 分解
            if args.device != 'cpu':
                gpu_metrics = perf_logger.get_gpu_metrics(args.device)
                metrics.gpu_id = gpu_metrics["gpu_id"]
                metrics.gpu_name = gpu_metrics["gpu_name"]
                metrics.gpu_utilization_percent = gpu_metrics["gpu_utilization_percent"]
                metrics.gpu_memory_used_mb = gpu_metrics["gpu_memory_used_mb"]
                metrics.gpu_memory_total_mb = gpu_metrics["gpu_memory_total_mb"]
                metrics.gpu_memory_percent = gpu_metrics["gpu_memory_percent"]
                metrics.gpu_temperature_c = gpu_metrics["gpu_temperature_c"]
                metrics.gpu_power_w = gpu_metrics["gpu_power_w"]
            
            metrics.status = "success"
            metrics.total_time = time.perf_counter() - total_start_time
            perf_logger.log_metrics(metrics)
            
            logging.info(f"=== Finished processing sensor: {sensor_info} ===")
            
            return data
        except Exception as e:
            print(f"stl_wavelet 方法执行失败: {e}")
            import traceback
            traceback.print_exc()
            metrics.status = "failed"
            metrics.error_message = str(e)
            metrics.total_time = time.perf_counter() - total_start_time
            perf_logger.log_metrics(metrics)
            return None
    
    # 如果没有匹配到任何方法
    metrics.status = "failed"
    metrics.error_message = f"Unknown method: {args.method}"
    metrics.total_time = time.perf_counter() - total_start_time
    perf_logger.log_metrics(metrics)
    return None


def main():
    # point_config = load_config('test_points_config.json')

    start_time = time.time()
    parser = argparse.ArgumentParser(description='Check Outlier')
    parser.add_argument('--path_iotdb', type=str, default='all', help='whlj, hbsn, gdsh')
    parser.add_argument('--input', type=str, default=None, help='Input CSV file path (overrides IoTDB)')
    parser.add_argument('--task_name', type=str, default='global', help='global or local')
    parser.add_argument('--processing_interval', type=str, default='7D', help='7D or 1M')
    parser.add_argument('--method', type=str, default='chatts',
                        help='piecewise_linear, stl_wavelet, standardized, iforest, cv, adtk_hbos, chatts, timer')
    parser.add_argument('--threshold', type=float, default=8, help='threshold')
    parser.add_argument('--num_points', type=float, default=1200, help='number of points for merge in a window')
    parser.add_argument('--n_jobs', type=int, default=2, help='number of processes')
    parser.add_argument('--data_path', type=str, default='/home/share/results/data', help='path to save results')
    parser.add_argument('--downsampler', type=str, default='m4', help='m4, minmax, none')
    parser.add_argument('--decompose', type=bool, default=True, help='decompose or not')
    parser.add_argument('--insert_missing', type=bool, default=True, help='insert missing or not')
    # parser.add_argument('--decompose', action='store_true', help='whether to decompose or not (default is False)')
    # parser.add_argument('--insert_missing', action='store_true', help='whether to insert missing values or not (default is False)')

    parser.add_argument('--config_file', type=str, default='test_points_config_gold.json', help='config file (relative to configs folder)')
    parser.add_argument('--ratio', type=float, default=0.1, 
                        help='降采样参数: 0-1之间为按比例降采样(例如0.1表示保留10%), 大于1时作为固定降采样点数')
    parser.add_argument('--n_downsample', type=int, default=10000,
                        help='LLM方法(chatts/timer)的固定降采样点数，优先于ratio参数')
    parser.add_argument('--ratio_classify', type=float, default=0.2, help=' ratio for classify global and local')
    parser.add_argument('--post_filter', type=str, default='mean', help='mean or median')
    parser.add_argument('--threhold_filter', type=float, default=0.5, help='threhold for filter')
    # adtk_hbos 相关参数
    parser.add_argument('--bin_nums', type=int, default=20, help='adtk_hbos 直方图分箱数量')
    parser.add_argument('--hbos_ratio', type=float, default=None, help='adtk_hbos 跳变过滤阈值比例 [0,1]')

    # parser.add_argument('--use_trend', action='store_true', help='whether to use trend component')
    # parser.add_argument('--use_seasonal', action='store_true', help='whether to use seasonal component')
    # parser.add_argument('--use_resid', action='store_true', help='whether to use residual component')
    
    parser.add_argument('--use_trend', type=bool, default=True, help='whether to use trend component')
    parser.add_argument('--use_seasonal', type=bool, default=False, help='whether to use seasonal component')
    parser.add_argument('--use_resid', type=bool, default=True, help='whether to use residual component')
    parser.add_argument('--device', type=str, default='cpu', help='device for stl decompose, auto cpu gpu')
    parser.add_argument('--min_threshold', type=int, default=200000, help='最小数据点阈值，数据量小于此值时不进行降采样')
    parser.add_argument('--use_clustering', type=bool, default=True, help='whether to use clustering method')
    parser.add_argument('--clustering_method', type=str, default='kmeans', help='clustering method for adaptive outlier split')
    parser.add_argument('--clustering_n_clusters', type=int, default=2, help='number of clusters for adaptive outlier split')
    parser.add_argument('--save_by_method', type=bool, default=True, help='是否按方法名保存到子目录')
    parser.add_argument('--save_mode', type=str, default='downsample', help='保存模式: raw 或 downsample')
    
    # ChatTS 相关参数
    parser.add_argument('--chatts_model_path', type=str, 
                        default='/home/share/llm_models/bytedance-research/ChatTS-8B',
                        help='ChatTS 模型路径')
    parser.add_argument('--chatts_device', type=str, default='cuda:1',
                        help='ChatTS 使用的 GPU 设备')
    parser.add_argument('--chatts_use_cache', type=str, default=None,
                        help='ChatTS 是否使用 KV cache (true/false)，默认None表示自动检测')
    parser.add_argument('--chatts_lora_adapter_path', type=str, default='/home/douff/ts/ChatTS-Training/saves/chatts-8b',
                        help='ChatTS LoRA 微调适配器路径，默认None表示使用原始模型')
    parser.add_argument('--chatts_max_new_tokens', type=int, default=4096,
                        help='ChatTS 最大生成token数，默认4096')
    parser.add_argument('--chatts_prompt_template', type=str, default='default',
                        help='ChatTS prompt模板名称: default, detailed, minimal, industrial, english')
    parser.add_argument('--chatts_load_in_4bit', type=str, default='auto',
                        help='ChatTS 是否使用 4-bit 量化 (true/false/auto)。auto=8B模型自动禁用，14B模型启用')
    
    # Timer 相关参数
    parser.add_argument('--timer_model_path', type=str,
                        default='/home/share/llm_models/thuml/timer-base-84m',
                        help='Timer 模型路径')
    parser.add_argument('--timer_device', type=str, default='cuda:0',
                        help='Timer 使用的 GPU 设备')
    parser.add_argument('--timer_lookback_length', type=int, default=256,
                        help='Timer 滚动预测窗口长度')
    parser.add_argument('--timer_threshold_k', type=float, default=3.5,
                        help='Timer 异常检测阈值系数')
    parser.add_argument('--timer_method', type=str, default='mad',
                        help='Timer 残差检测方法: mad, sigma')
    parser.add_argument('--timer_min_run', type=int, default=1,
                        help='Timer 最小连续异常点数')
    parser.add_argument('--timer_streaming', type=bool, default=False,
                        help='Timer 是否使用流式模式')
    parser.add_argument('--timer_reset_interval', type=int, default=256,
                        help='Timer 流式模式下的上下文重置周期')
    
    args = parser.parse_args()
    print(args)

    sensor_infos = []
    
    if args.input:
        print(f"Running in single file mode: {args.input}")
        # 在单文件模式下，创建一个虚拟的 sensor_info
        # (path, column, st, et)
        # 具体的列名和时间会在 process_sensor 中从 CSV 读取并更新
        sensor_infos.append(('manual', 'value', '2023-01-01', '2024-01-01'))
    else:
        point_config = load_config(args.config_file)
        
        point_summary = {}
        for path, sensor_variables in point_config.items():
            columns = sensor_variables['columns']
            st = sensor_variables['st']
            et = sensor_variables['et']

            for column in columns:
                sensor_infos.append((path, column, st, et))
                
            point_summary[path] = {
                # "device": path,  # 假设设备名是路径第二段
                "num_points": len(columns),
                "start_time": st,
                "end_time": et
            }
        
        # 保存参数到文件，文件名包含时间戳
        # save_args_to_file(args, point_summary, os.path.basename(__file__), save_dir= '/opt/results/figs', file_prefix=f"{args.task_name}_params")
        save_args_to_file(args, point_summary, os.path.basename(__file__), save_dir= './params', file_prefix=f"{args.task_name}_params")
    
    # ChatTS 和 Timer 方法需要串行处理以避免 GPU OOM
    if args.method == 'chatts':
        # 预加载 ChatTS 模型，保证整个运行过程只加载一次（单进程内复用）
        try:
            from chatts_detect import get_analyzer
            print("[ChatTS] 预加载模型（仅加载一次，后续复用）...")
            _ = get_analyzer(
                model_path=args.chatts_model_path,
                device=args.chatts_device,
                load_in_4bit=True,
                lora_adapter_path=args.chatts_lora_adapter_path,
            )
        except Exception as e:
            print(f"[ChatTS] 预加载失败：{e}，将在首次调用时再尝试加载")
        
        # 支持 n_jobs 并行处理（48G 显存可支持适量并行）
        if args.n_jobs == 1:
            print("[ChatTS] 使用串行处理（n_jobs=1）")
            results = []
            for info in tqdm(sensor_infos, desc='ChatTS 处理进度'):
                result = process_sensor(info, args)
                results.append(result)
        else:
            print(f"[ChatTS] 使用并行处理（n_jobs={args.n_jobs}），注意 GPU 显存占用")
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_sensor)(info, args) for info in tqdm(sensor_infos, desc='ChatTS 处理进度')
            )
    elif args.method == 'timer':
        # 预加载 Timer 模型
        try:
            from timer_detect import get_timer_pipeline
            print("[Timer] 预加载模型（仅加载一次，后续复用）...")
            _ = get_timer_pipeline(
                model_path=args.timer_model_path,
                device=args.timer_device,
            )
        except Exception as e:
            print(f"[Timer] 预加载失败：{e}，将在首次调用时再尝试加载")
        
        # 支持 n_jobs 并行处理（48G 显存可支持适量并行）
        if args.n_jobs == 1:
            print("[Timer] 使用串行处理（n_jobs=1）")
            results = []
            for info in tqdm(sensor_infos, desc='Timer 处理进度'):
                result = process_sensor(info, args)
                results.append(result)
        else:
            print(f"[Timer] 使用并行处理（n_jobs={args.n_jobs}），注意 GPU 显存占用")
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_sensor)(info, args) for info in tqdm(sensor_infos, desc='Timer 处理进度')
            )
    else:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_sensor)(info, args) for info in sensor_infos
        )

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")


if __name__ == "__main__":
    main()
