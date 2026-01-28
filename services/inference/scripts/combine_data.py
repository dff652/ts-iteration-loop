from iotdb.Session import Session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from tqdm import tqdm
from scipy.ndimage import label

# 兼容不同导入方式
try:
    # 当作为包的一部分导入时
    from .ensemble import replace_outliers_by_3sigma, create_outlier_mask, iqr_find_anomaly_indices, nsigma_find_anomaly_indices, piecewise_linear, standardized_find_anomaly_indices
    from .wavelet import reconstruct_residuals
except ImportError:
    # 当直接运行脚本时
    from ensemble import replace_outliers_by_3sigma, create_outlier_mask, iqr_find_anomaly_indices, nsigma_find_anomaly_indices, piecewise_linear, standardized_find_anomaly_indices
    from wavelet import reconstruct_residuals
from scipy import signal
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
import argparse
import time
import json
from multiprocessing import Pool
import os
print("当前工作目录:", os.getcwd())

from datetime import datetime
from dateutil import parser

from typing import Union, List
from config import load_args, save_args_to_file


def load_config(config_filename):
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的绝对路径
    conlabel_path = os.path.join(script_dir, 'configs', config_filename)

    with open(conlabel_path, 'r') as f:
        config = json.load(f)
    return config


def get_global_data(directory, sensor_id, method='piecewise_linear'):
    
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
            if method is None:
                print(file_path)
                data = pd.read_csv(file_path,index_col=0)
                return data
            else:
                if method in file_path:
                    print(file_path)
                    data = pd.read_csv(file_path,index_col=0)
                    return data
    else:
        print("未找到包含指定点名的CSV文件。")

def time_convert(time_str):
    
    # 原始时间字符串 "2023-06-01 14:35:05+08:00"
    
    # 解析时间字符串
    parsed_time = parser.isoparse(time_str)

    # # 创建目标时间
    # new_time = datetime(2023, 1, 1, 11, 46, 28)

    # 格式化为指定格式
    formatted_time = parsed_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        
    return formatted_time
    

       
def filter_data(data, mask = 'outlier_mask', threshold=2):
    combine_masks = data.copy()
    combine_masks['label'] = combine_masks[mask].apply(lambda x: 1 if x >= threshold else 0)
    combine_masks['Group'] = (combine_masks['label'] != combine_masks['label'].shift()).cumsum()
    label_data = combine_masks[combine_masks['label'] == 1]
    grouped = label_data.groupby('Group')
    
    target_data = pd.DataFrame(columns=['测点', '开始时间', '结束时间'])
    for name, group in grouped:
        start_time = group.index.min()  # 获取组的最小时间作为开始时间
        end_time = group.index.max()  # 获取组的最大时间作为结束时间
        sensor = label_data.columns[0]  # 获取测点的列名，假设第一列是测点
        # tag = '标记'  # 标签信息，固定为 '标记'

        # 将结果添加到 target_data 中
        target_data = pd.concat([target_data, pd.DataFrame({
            '测点': [sensor],
            '开始时间': [start_time],
            '结束时间': [end_time],
            '标签': [mask],
        })], ignore_index=True)
        
    target_data['开始时间'] = target_data['开始时间'].apply(lambda x: time_convert(x))
    target_data['结束时间'] = target_data['结束时间'].apply(lambda x: time_convert(x))
    return target_data

def process_combined_column(column, data_path,  methods, root_path):
    """
    处理每个列的数据，在三种方法的标签中，有至少两个方法检测到异常值时，将其标记为异常值。
    
    参数:
    - column: 需要处理的列名。
    - data_path: 全局数据路径。
    - methods: 一个包含方法名的列表，例如 ['wavelet', 'piecewise_linear', 'standardized']。
    """
    
    combine_masks = None  # 初始设置为 None，表示还未初始化

    for i, method in enumerate(methods):
        # if method == 'combined':
        #     data = get_global_data(data_path, column, None)
        # else:
        #     data = get_global_data(data_path, column, method) 
        
        data = get_global_data(data_path, column, None)
             
        if data is None or data.empty:
            print(f"{data_path}未找到全局数据文件或数据为空: {column} (Method: {method})")
            continue
        
        for mask in data.columns[1:]:
            print(mask)
            # 第一次读取数据时初始化 combine_masks
            # if combine_masks is None:
            
            combine_masks = pd.DataFrame()
            combine_masks[column] = data[column]
            combine_masks[mask] = 0  # 初始化为0


        # # 确保索引对齐，防止数据集之间的时间错位
        # data = data.reindex(combine_masks.index, fill_value=0)

            # 累加异常掩码
            combine_masks[mask] += data[mask].fillna(0)
            
            res = filter_data(combine_masks, mask, threshold=1)
            
            # excluded_dates = [('2024-04-09', '2024-04-09'),
            #                    ('2024-04-27', '2024-06-27')]
            
            # for start_date, end_date in excluded_dates:
            #     print(f"排除日期: {start_date} - {end_date}")
            #     res = res[~((res['start_time'] >= start_date) & (res['end_time'] <= end_date))]
             
        
            label_path = os.path.join(root_path, f"{res.shape[0]}_{mask}_{column}_{len(methods)}.csv")
            
            if os.path.exists(label_path):
                print(f"标签数据已存在，跳过生成: {label_path}")
                continue
            
            res.to_csv(label_path, encoding='utf-8',quotechar='"', quoting=1, index=False)
            # res.to_csv(label_path, encoding='utf-8',quotechar='"', quoting=1, index=False)
            
            print(f"csv文件已保存: {label_path}")
        
    
    






def main( ):
    
    
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Combine outlier labels remain start and end time')
    parser.add_argument('--task_name', type=str, default='local', help='global or local')
    parser.add_argument('--methods', type=str, default='wavelet', help='methods')
    parser.add_argument('--config_file', type=str, default='test_points_config_gold.json', help='config file')
    parser.add_argument('--data_path', type=str, default='/opt/results/', help='data path')
    parser.add_argument('--label_path', type=str, default='/opt/results/figs', help='save fig path')
    parser.add_argument('--n_jobs', type=int, default=12, help='number of processes')
    
    args = parser.parse_args()
    print(args)
    point_config = load_config(args.config_file)
    
    # save_args_to_file(args, script_name = os.path.basename(__file__), save_dir= args.label_path, file_prefix=f"{args.task_name}_params")   
    save_args_to_file(args, script_name = os.path.basename(__file__), save_dir= './params', file_prefix=f"{args.task_name}_params")   
    
    data_path = os.path.join(args.data_path, args.task_name)
    label_path = os.path.join(args.label_path, args.task_name)    
    
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    
    if not os.path.exists(data_path):
        print(f"数据路径不存在: {data_path}, 创建目录...")
        os.makedirs(data_path)
        
    
    columns_to_process = []
    for path, sensor_variables in point_config.items():
        print(f"Processing path: {path}")
        print(f"Sensor variables: {sensor_variables}")
        columns_to_process.extend(sensor_variables['columns'])
    
    n_jobs =  args.n_jobs
    
    print(f"使用 {n_jobs} 个进程并行处理任务...")
    
    methods = args.methods
    
    if isinstance(methods, list):
        
        with Pool(processes=n_jobs) as pool:
            pool.starmap(process_combined_column, [(col, data_path, methods, label_path) for col in columns_to_process])
            
    elif isinstance(methods, str):
        # 单标签绘图
        with Pool(processes=n_jobs) as pool:
            pool.starmap(process_combined_column, [(col, data_path, [methods], label_path) for col in columns_to_process])
    
    print(f"任务完成，总用时: {time.time() - start_time} 秒")
    
    

if __name__ == '__main__':
    # methods = ['piecewise_linear', 'wavelet','standardized','iforest']
    # excute_time = time.strftime("%Y-%m-%d", time.localtime())
    
    # point_conlabel_path = 'test_points_config.json'
    
    # methods = ['combined']  
    # for method in methods: 
    #     label_path = f'/opt/results/test_{method}_{excute_time}'
    #     main( point_conlabel_path, method, label_path = label_path , n_jobs=8)
    main()