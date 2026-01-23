
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from datetime import datetime
import argparse
import time
import json
from multiprocessing import Pool
import os
print("当前工作目录:", os.getcwd())
import seaborn as sns

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap




def load_config(config_filename):
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的绝对路径
    conlabel_path = os.path.join(script_dir, 'configs', config_filename)

    with open(conlabel_path, 'r') as f:
        config = json.load(f)
    return config


def get_global_data(directory, sensor_id, method=None):
    
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
        
        
def calculate_time_scales(indices_df):
    indices_df_int = indices_df.astype(int)
    ratio_2h = indices_df_int.resample('2H', label='right',closed = 'right').mean().fillna(0)
    ratio_1d = indices_df_int.resample('1D', label='right',closed = 'right').mean().fillna(0)
    ratio_1w = indices_df_int.resample('7D', label='right',closed = 'right').mean().fillna(0)
    
    return ratio_2h, ratio_1d, ratio_1w 


def convert2heatmap(data_path, 
                    col, 
                    label, 
                    method, 
                    resample='2H'):
    print('='*50)
    print(f"正在转化热力图数据: {col}")
    
    df_heatmap = get_global_data(data_path, col, method)    
    
    if df_heatmap is None or df_heatmap.empty:
        print(f"{data_path}未找到全局数据文件或数据为空: {col} (Method: {method})")
    
    df_heatmap.index = pd.to_datetime(df_heatmap.index )
    # ratio_2h, ratio_1d, ratio_1w = calculate_time_scales(df_heatmap[[label + '_mask']])
    
    df_heatmap = df_heatmap[[label + '_mask']].astype(int)
    df_resample = df_heatmap.resample(resample, label='right',closed = 'right').mean().fillna(0)
    
    # ratio_2h.columns = [col]
    # ratio_1d.columns = [col]
    # ratio_1w.columns = [col]
    
    df_resample.columns = [col]
    
    return df_resample




def get_sum(df, axis=0):
    if axis == 0:
        df_sum = df.mean(axis=axis)
        return pd.DataFrame(df_sum, columns=['mean'])
    else:
        df_sum = df.sum(axis=axis)
  
        return pd.DataFrame(df_sum, columns=['sum'])

def col2binary(column):
    """
    将列转换为二进制列。
    """
    binary_column = column.apply(lambda x: 1 if x != 0 else 0)
    return binary_column

def get_matrix(data1,data2):
    
    result_matrix = np.zeros_like(data1.values)

    # 只有data1有数据的区域，标记为1
    result_matrix[(data1.values != 0) & (data2.values == 0)] = 1

    # 只有data2有数据的区域，标记为2
    result_matrix[(data1.values == 0) & (data2.values != 0)] = 2

    # data1和data2都有数据的区域，标记为3
    result_matrix[(data1.values != 0) & (data2.values != 0)] = 3
    
    return result_matrix
    
    
        
# 两个热力图的合并
def fig_heatmap(path_1, 
                path_2,
                save_path = '/home/share/results/heatmap/global/', 
                binary=True):
    
    file_name_1 = os.path.basename(path_1)
    file_name_2 = os.path.basename(path_2)
    
    data_1 = pd.read_csv(path_1, index_col=0)
    data_2 = pd.read_csv(path_2, index_col=0)
    
    
    
    
    metric_colsum = get_sum(data_1, axis=0)
    sorted_colsum = metric_colsum.sort_values(by="mean", ascending=True)
    
    # 二值化处理和逻辑或累积
    if binary:
        df_sorted_1 = data_1[sorted_colsum.index].apply(col2binary, axis=0)
        
        df_sorted_2 = data_2[sorted_colsum.index].apply(col2binary, axis=0)
        

    else:
        df_sorted_1 = data_1[sorted_colsum.index]
        df_sorted_2 = data_2[sorted_colsum.index]
           
    result_matrix = get_matrix(df_sorted_1, df_sorted_2)
    
    colors = ['#D3D3D3', '#FF0000', '#00FF00', '#0000FF']  # 对应 0, 1, 2, 3 的颜色
    cmap = ListedColormap(colors)

    plt.figure(figsize=(20, 10))

    # 使用 seaborn 绘制热力图
    # sns.heatmap(df_sorted_1.T, 
    #             cmap="Blues", 
    #             cbar_kws={'label': file_name_1[6:-4]}, 
    #             annot=False,
    #             label = file_name_1[6:-4],
    #             )  # 第一份数据的热力图
    # sns.heatmap(df_sorted_2.T, 
    #             cmap="Reds", 
    #             cbar_kws={'label': file_name_2[6:-4]}, 
    #             annot=False, 
    #             alpha=0.5,
    #             label = file_name_2[6:-4],
    #             )  # 第二份数据的热力图
    
    
    # 绘制热力图，并设置线框
    sns.heatmap(result_matrix.T, 
                cmap=cmap, 
                cbar=True, 
                annot=False, 
                xticklabels=df_sorted_1.index, 
                yticklabels=df_sorted_1.columns, 
                cbar_kws={'ticks': [0, 1, 2, 3]},
                linewidths=0.5, 
                linecolor='black'
                )  # 设置线框宽度为0.5，线条颜色为黑色

    # 设置标题
    plt.title("Heatmap of data1 and data2")

    # 添加自定义图例
    labels = ['0: Both have no data', '1: Only data1', '2: Only data2', '3: Both have data']
    colors = ['#D3D3D3', '#FF0000', '#00FF00', '#0000FF']  # 对应的颜色

    # # 创建图例
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1))

    # 调整 x 轴时间刻度
    plt.xticks(ticks=np.arange(0, len(df_sorted_1.index), step=10), labels=df_sorted_1.index[::10], rotation=45)

    # 添加标题
    plt.title(f"data1: {file_name_1[6:-4]} data2: {file_name_2[6:-4]} ")

    # 保存图像
    save_path = os.path.join(save_path, f"{file_name_1[6:-4]}_{file_name_2[6:-4]}.png")
    plt.savefig(save_path)
    print(f'热力图已保存到: {save_path}')
    

   
def generate_heatmap_data( ):
    
    
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Combine outlier labels remain start and end time')
    parser.add_argument('--task_name', type=str, default='global', help='global or local')
    parser.add_argument('--method', type=str, default='wavelet', help='method')
    parser.add_argument('--config_file', type=str, default='test_points_config_gold.json', help='config file')
    parser.add_argument('--data_path', type=str, default='/opt/results/', help='data path')
    parser.add_argument('--save_path', type=str, default='/opt/results/heatmap', help='heatmap')
    parser.add_argument('--n_jobs', type=int, default=12, help='number of processes')
    parser.add_argument('--resample', type=str, default='1D', help='2H, 1D, 1W')
    
    parser.add_argument('--heatmap', type = bool, default=False, help='fig heatmap')
    parser.add_argument('--path_1', type = str, default='', help='path_1')
    parser.add_argument('--path_2', type = str, default='', help='path_2')
    parser.add_argument('--save_fig', type = str, default='', help='save_fig')
    
    args = parser.parse_args()
    print(args)
    point_config = load_config(args.config_file)
    
    data_path = os.path.join(args.data_path, args.task_name)
    save_path = os.path.join(args.save_path, args.task_name) 
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if not os.path.exists(data_path):
        print(f"数据路径不存在: {data_path}, 创建目录...")
        os.makedirs(data_path)
        
    if args.heatmap:
        print(f"开始生成热力图...")
        # path_1 = "/opt/results/heatmap/global/1D_54_20250223_2143.csv"
        # path_2 = "/opt/results/heatmap/global/1D_54_20250223_2205.csv"
        # save_path = '/home/douff/'
        path_1 = args.path_1
        path_2 = args.path_2
        save_fig = args.save_fig
        fig_heatmap(path_1, path_2, save_fig)
        
    else:
        print(f"开始生成热力图数据...") 
        
        columns_to_process = []
        for path, sensor_variables in point_config.items():
            print(f"Processing path: {path}")
            print(f"Sensor variables: {sensor_variables}")
            columns_to_process.extend(sensor_variables['columns'])
            
        n_jobs =  args.n_jobs
        
        print(f"使用 {n_jobs} 个进程并行处理任务...")
        
        method = args.method
        
        # data_heatmap = {'2h': [], '1d': [], '1w': []}
        
        # data_2h = pd.DataFrame()
        # data_1d = pd.DataFrame()
        # data_1w = pd.DataFrame()
        
        data_resample = pd.DataFrame()
        
            
        with Pool(processes=n_jobs) as pool:
            results = pool.starmap(convert2heatmap, 
                                [(data_path, col, args.task_name, method, args.resample) for col in columns_to_process]
                                )
        
        for ratio_resample in results:
            # data_2h = pd.concat([data_2h, ratio_2h], axis=1)
            # data_1d = pd.concat([data_1d, ratio_1d], axis=1)    
            # data_1w = pd.concat([data_1w, ratio_1w], axis=1)  
            
            data_resample = pd.concat([data_resample, ratio_resample], axis=1)
        
        # data_heatmap['2h'] = pd.concat(data_heatmap['2h'], axis=1)
        # data_heatmap['1d'] = pd.concat(data_heatmap['1d'], axis=1)
        # data_heatmap['1w'] = pd.concat(data_heatmap['1w'], axis=1)

        
        
        
        # data_heatmap = pd.DataFrame(data_heatmap)
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # 获取当前时间并格式化
        save_path = os.path.join(save_path, f"{args.resample}_{len(columns_to_process)}_{timestamp}.csv")
        
        data_resample.to_csv(save_path)
            
        
        print(f"任务完成，总用时: {time.time() - start_time} 秒")



if __name__ == '__main__':
    generate_heatmap_data()
    
    # path_2 = "/opt/results/heatmap/global/1D_54_20250224_2238.csv"
    # path_1 = "/opt/results/heatmap/global/1D_54_20250224_1630.csv"
    # save_path = './'
    # fig_heatmap(path_1, path_2, save_path)
    
    
    