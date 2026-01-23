from iotdb.Session import Session
import pandas as pd
import numpy as np
import ruptures as rpt
import json
import time
import os
import logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


log_file_path = os.path.join(os.getcwd(), 'processing.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_filename):
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的绝对路径
    config_path = os.path.join(script_dir, 'configs', config_filename)

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config






def check_time_continuity_global(df: pd.DataFrame,
                                 st_global: str,
                                 et_global: str,
                                 discontinuity_threshold=None
                                 ) -> pd.DataFrame:
    """
    获取全数据
    Parameters
    ----------
    df : DataFrame
        有效数据集
    st : str
        页面选取的开始时间
    et : str
        页面选取的结束时间
    discontinuity_threshold  : int
        采样间隔，默认为1秒
    Returns
    -------
    data : DataFrame
        全数据
    """
    data = df.copy()

    st_global = pd.to_datetime(st_global)
    et_global = pd.to_datetime(et_global)

    ts = 'ts'
    time_index = pd.DataFrame()
    time_index[ts] = data.index
    interval = (time_index[ts] - time_index[ts].shift(1))
    interval_seconds = interval.dt.total_seconds()

    if discontinuity_threshold is None or discontinuity_threshold == '':
        # 根据数据中的时间间隔推断采样频率
        print(interval_seconds.mode())
        discontinuity_threshold = interval_seconds.mode()[0]
    else:
        discontinuity_threshold = int(discontinuity_threshold)

    freq = f'{int(discontinuity_threshold)}s'

    # 构造完整的时间戳序列
    global_timestamps = pd.date_range(start=st_global, end=et_global, freq=freq)

    global_indices = pd.DataFrame(data=True, index=global_timestamps, columns=data.columns)
    global_indices.index = global_indices.index.tz_localize('Asia/Shanghai')

    local_timestamps = data.index
    global_indices.loc[global_indices.index.isin(local_timestamps), :] = False

    return global_indices


def get_indices(indices_df):
    """
    获取时间片段记录
    Parameters
    ----------
    indices_df : DataFrame
        布尔索引，标记每个数据点是否为异常值
    Returns
    -------
    indices_slice_df : DataFrame
        片段记录，记录时间片段的开始和结束时间，以及问题类型
    """
    if indices_df.empty:
        return None

    all_intervals = []

    for label in indices_df.columns:
        # 查找布尔值变化的位置
        change_points = indices_df[label].astype(int).diff().fillna(0)

        # 找到起点和终点
        start_indices = change_points[change_points == 1].index
        end_indices = change_points[change_points == -1].index

        # 如果开头是 True，则第一个起点从开头开始
        if indices_df[label].iloc[0]:
            start_indices = start_indices.insert(0, indices_df.index[0])

        # 如果结尾是 True，则最后一个终点是最后一个索引
        if indices_df[label].iloc[-1]:
            end_indices = end_indices.append(pd.Index([indices_df.index[-1]]))

        # 将起点和终点组合为时间片段
        for start, end in zip(start_indices, end_indices):
            all_intervals.append({'Start': start, 'End': end, 'Label': label})

    # 转换为 DataFrame
    indices_slice_df = pd.DataFrame(all_intervals)

    return indices_slice_df

def plot_global(data, col, save_path, method='1'):

    
    plt.figure(figsize=(20, 12))
    
    # 确保数据索引是时间类型
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
        
    # 绘制原始数据曲线，使用索引序列号作为横坐标
    data_len = data.shape[0]
    plt.plot(range(data_len) ,data[col], label='Original Data')
    
    # 根据 mask 列的值填充异常点的区域
    plt.fill_between(range(data_len), data[col].min()*1.02, data[col].max()*1.02, 
                     where=data['outlier_mask'] == 1, color='red', alpha=0.5, label='Anomalies')
    
    # 设置日期格式
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    
    # 手动设置横坐标
    xticks = np.linspace(0, data_len - 1, num=10, dtype=int)  # 取 10 个间隔均匀的点
    xtick_labels = data.index[xticks].to_pydatetime()  # 对应的时间标签

    plt.xticks(xticks, [label.strftime('%Y-%m-%d %H:%M:%S') for label in xtick_labels] ,rotation=45, ha='right')

   
    # plt.xticks(rotation=45)
    plt.legend(loc="upper right")
    plt.title(f'Sensor ID: {col} Method {method}')  # 可以添加标题显示sensor_id
    
    # 保存图像

    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    
    plt.savefig(save_path, dpi=50)
    plt.close()   
    
    

def plot_local(data, label_data, fig_save_path='test_anomaly', mask = 'global_mask'):
    
    data.index = pd.to_datetime(data.index)
    column = label_data.iloc[0, 0]
    
    for i in range(label_data.shape[0]):
        start_time = pd.to_datetime(label_data.iloc[i, 0])
        end_time = pd.to_datetime(label_data.iloc[i, 1])
        
        save_path = os.path.join(fig_save_path, f'{label_data.shape[0]}_{mask}_{column}_{start_time}_{end_time}_local_plot.png')
        if os.path.exists(save_path):
            print(f"图像已存在，跳过生成: {save_path}")
            continue

        # 前后5小时和20小时的时间范围
        five_hours_before = start_time - pd.Timedelta(hours=5)
        five_hours_after = end_time + pd.Timedelta(hours=5)
        twenty_hours_before = start_time - pd.Timedelta(hours=20)
        twenty_hours_after = end_time + pd.Timedelta(hours=20)

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
        start_idx = data.index.get_loc(start_time)
        end_idx = data.index.get_loc(end_time)

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
       
        plt.savefig(save_path, dpi=100)
        print(f'图像已保存到: {save_path}')
        plt.close()


def read_iotdb(
        host="192.168.199.185",
        port="6667",
        user='root',
        password='root',
        path='root.supcon.nb.whlj.LJSJ',
        target_column='*',
        st='2023-06-01 00:00:00',
        et='2024-08-01 00:00:00',
        limit=1000000000,

):
    session = Session(host, port, user, password, fetch_size=2000000)
    session.open(False)

    ststring = ">=" + st.replace(' ', 'T')
    etstring = "<=" + et.replace(' ', 'T')

    query = f"select `{target_column}` from {path}"

    if st:
        ststring = ">=" + st.replace(' ', 'T')

    if et:
        etstring = "<=" + et.replace(' ', 'T')

    if st and not et:
        query = query + f" where time {ststring}"
    if et and not st:
        query = query + f" where time {etstring}"
    if st and et:
        query = query + f" where time {ststring} and time {etstring}"

    if limit:
        query = query + f" limit {limit}"

    result = session.execute_query_statement(query)
    result.set_fetch_size(3000000)

    df = result.todf()

    df.set_index('Time', inplace=True)

    df.index = pd.to_datetime(df.index.astype('int64')).tz_localize('UTC').tz_convert('Asia/Shanghai')

    column_rename = {}
    for column in df.columns:
        if column.endswith('`'):
            column_new = column[(column.rindex('`', 0, len(column) - 2) + 1): len(column) - 1]
        else:
            column_new = column.split('.')[-1].replace('`', '')
        column_rename[column] = column_new

    df = df.rename(columns=column_rename)
    return df




def process_sensor(info, root_path):
    path, column, st, et = info
    data = read_iotdb(path = path, target_column=column, st=st, et=et)
    print(f"Processing {path} {column} from {st} to {et}")
    
    discontinuity_indices = check_time_continuity_global(data, st, et)
    
    indices_slice = get_indices(discontinuity_indices)
    
    save_path = os.path.join(root_path, f"{indices_slice.shape[0]}_{column}_discontinuity.csv")
    
    indices_slice.to_csv(save_path)
    print(f"csv文件已保存: {save_path}")
    
    plot_local(data, indices_slice, root_path, 'discontinuity')
    
    print(f"png文件已保存: {save_path.replace('.csv', '.png')}")
    


def main(root_path ='/opt/results/test' ,n_jobs = 2):
    start_time = time.time()
    
    point_config = load_config('more_few_points_config.json')
    
    sensor_infos = []
    for path, sensor_variables in point_config.items():
        columns = sensor_variables['columns']
        st = sensor_variables['st']
        et = sensor_variables['et']

        for column in columns:
            sensor_infos.append((path, column, st, et))

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sensor)(info, root_path) for info in sensor_infos
    )

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")



if __name__ == "__main__":
    
    main(n_jobs=2)