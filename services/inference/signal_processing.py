from iotdb.Session import Session
from tsdownsample import MinMaxLTTBDownsampler, M4Downsampler
from statsmodels.tsa.seasonal import STL
# import hastl
import pandas as pd
import numpy as np
import functools
import logging
import time
# import torch
import os 
import subprocess
from scipy.signal import welch
import pywt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from itertools import combinations
# 兼容不同导入方式
try:
    # 当作为包的一部分导入时（如main.py导入）
    from .wavelet import reconstruct_residuals
except ImportError:
    # 当直接运行脚本时
    from wavelet import reconstruct_residuals
    
    
# 配置日志文件 - 统一写入 logs 目录
import os as _os
_logs_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'logs')
if not _os.path.exists(_logs_dir):
    _os.makedirs(_logs_dir)
logging.basicConfig(
    filename=_os.path.join(_logs_dir, "function_timer.log"),  # 日志文件名
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(message)s"  # 日志格式
)

# 自动检测设备（GPU 或 CPU）
# def detect_device():
#     if torch.cuda.is_available():
#         print("检测到可用的 GPU，使用 GPU 版本的 STL")
#         os.environ["HASTL_BACKENDS"] = "cuda"  # 设置环境变量
#         return 'cuda'
#     else:
#         print("没有检测到 GPU，使用 CPU 版本的 STL")
#         return 'c'
    
def detect_device():
    try:
        # 使用 subprocess 调用 nvidia-smi 查看 GPU 状态
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("检测到可用的 GPU，使用 GPU 版本的 STL")
            return 'cuda'
        else:
            print("没有检测到 GPU，使用 CPU 版本的 STL")
            return 'c'
    except FileNotFoundError:
        print("未找到 nvidia-smi 命令，使用 CPU 版本的 STL")
        return 'c'


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


_stl_instance = None
# 全局变量存储单例实例
@timer_decorator
def create_stl(b="cuda"):
    """
    创建并返回一个 HaSTL 对象。
    :param b: 后端选择，可以是 'cuda', 'opencl', 'multicore', 'c'
    :return: STL 对象
    """
    global _stl_instance  # 使用全局变量
    if _stl_instance is None:  # 如果尚未初始化
        try:
            import hastl
            _stl_instance = hastl.STL(backend=b)
            print(f"成功创建 {b} 后端的 STL 对象")
        except ValueError as e:
            print(f"创建 {b} 后端的 STL 对象失败: {e}")
            return None
    return _stl_instance


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


def check_time_continuity(data, discontinuity_threshold=None):
    """
    检查时间序列的连续性
    Parameters
    ----------
    data : DataFrame
        输入的数据集，索引为时间戳
    sLength : int
        采样间隔，以秒为单位，默认为1秒
    Returns
    -------
    continuity_ratio : float
        时间戳中断的比例（相邻两个时间戳之差大于标准采样频率）
    continuity : Series
        布尔类型序列，标记每个间隔是否超过采样频率
    missing_timestamps : DatetimeIndex
        缺失的时间戳
    missing_ratio : float
        缺失时间戳占完整模板时间戳的比例
    """
    # score_df = pd.DataFrame(index=['time_continuity_ratio'],columns=data.columns)
    ts = 'ts'
    time_index = pd.DataFrame(columns=[ts], index=data.index)
    time_index[ts] = data.index
    interval = (time_index[ts] - time_index[ts].shift(1))
    interval_seconds = interval.dt.total_seconds()  # .values.ravel()
    # print('interval_seconds.mode :', interval_seconds.mode())
    if discontinuity_threshold is None or discontinuity_threshold == '':
        # 根据数据中的时间间隔推断采样频率
        # print(interval_seconds.mode())
        if len(interval_seconds) > 1:
            discontinuity_threshold = interval_seconds.mode()[0]
        else:
            return pd.DataFrame({
                'missing_ratio': [0],
                'missing_timestamps_count': [0]
            }).transpose(), pd.Series(False, index=data.index)
    else:
        discontinuity_threshold = int(discontinuity_threshold)

    continuity = interval_seconds > int(discontinuity_threshold)
    continuity_ratio = continuity.sum() / len(interval_seconds)

    # score = np.round((continuity_ratio) * 100, 2)
    # score = np.vectorize(lambda x: "{:.2f}%".format(x))(score)
    # score_df.loc[:,:] = score
    # discontinuity_index=np.where(interval_seconds > 1)

    start_time = time_index[ts].min()
    end_time = time_index[ts].max()
    # freq = pd.infer_freq(time_index[ts])
    freq = f'{int(discontinuity_threshold)}s'

    if freq is None:
        raise ValueError("无法推断时间序列的频率，请确保输入是有序时间戳序列")

    # 构造完整的时间戳序列
    full_timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)

    # 找出缺少的时间戳
    missing_timestamps = full_timestamps.difference(time_index[ts])

    # 创建布尔序列以标记缺失和不连续的时间戳
    continuity = pd.Series(False, index=full_timestamps)
    continuity[missing_timestamps] = True  # 标记缺失的时间戳为 True

    # existing_continuity = pd.Series(interval_seconds > discontinuity_threshold, index=data.index)
    # continuity.update(existing_continuity)

    missing_count = continuity.sum()

    # 计算缺少时间戳的比例
    missing_ratio = missing_count / len(data)

    result_df = pd.DataFrame({
        f'missing_ratio': [missing_ratio],
        f'missing_timestamps_count': [missing_count]
    }).transpose()

    return result_df, continuity


def get_fulldata(data,
                 col_name, ):
    # print('-' * 40 + '开始' + '-' * 40)
    # print(f'\n开始读取数据{col_name},时间段{st}到{et}')

    # data = read_iotdb(
    #     target_column=col_name,
    #     path=path,
    #     st=st,
    #     et=et)
    # print(f'原始数据：{data.head()}')
    # print(f'原始数据大小：{data.shape}')

    result_df, continuity = check_time_continuity(data)
    # print('*'*40)
    # print(f'填充前缺失时间戳：{result_df}')

    df = data.copy()

    missing_timestamps = continuity[continuity > 0].index
    # print(f'缺失时间戳:\n {missing_timestamps}')

    # for ts in missing_timestamps:
    #     if ts not in df.index:
    # df.loc[missing_timestamps] = np.nan

    full_time_index = df.index.append(pd.DatetimeIndex(missing_timestamps)).unique()
    df = df.reindex(full_time_index)

    df.sort_index(inplace=True)

    # 首先进行后向填充
    df[col_name] = df[col_name].bfill()

    # 然后进行前向填充
    df[col_name] = df[col_name].ffill()
    # 使用前向填充和后向填充填充缺失值

    # df[col_name] = df[col_name].fillna(method='ffill')
    # df[col_name] = df[col_name].fillna(method='bfill')

    # result_df, continuity = check_time_continuity(df)
    # print('*'*40)
    # print(f'填充前缺失时间戳：{result_df}')

    return df


@timer_decorator
def ts_downsample(data, downsampler='m4', n_out=100000):
    """
    Downsample time series data
    :param data: pd.Series - 输入时间序列数据
    :param downsampler: str - 降采样方法 ('m4' 或 'minmax')
    :param n_out: int - 输出数据点数
    :return: tuple of (downsampled_data, downsampled_time, position_index)
        - downsampled_data: pd.Series - 降采样后的数据
        - downsampled_time: pd.Index - 降采样后的时间戳/索引
        - position_index: np.ndarray - 降采样点在原始数据中的位置索引（整数数组）
    """

    if downsampler == 'm4':
        s_ds = M4Downsampler().downsample(data, n_out=n_out)
    elif downsampler == 'minmax':
        s_ds = MinMaxLTTBDownsampler().downsample(data, n_out=n_out)

    downsampled_data = data.iloc[s_ds]
    downsampled_time = data.index[s_ds]
    
    position_index = np.asarray(s_ds, dtype=np.int64)

    return downsampled_data, downsampled_time, position_index

@timer_decorator
def adaptive_downsample(data, downsampler='m4', sample_param=0.1, min_threshold=1000):
    """
    自适应降采样函数，支持按比例或固定数量降采样
    
    参数:
        data: pd.Series 或带有单列的 pd.DataFrame - 输入数据
        downsampler: str - 降采样方法，支持 'm4', 'minmax', 'none'或None，为None或'none'时不降采样
        sample_param: float 或 None 
                     - 0到1之间表示按比例降采样
                     - 大于1时自动转为None，表示使用min_threshold作为固定降采样数量
        min_threshold: int - 两个用途:
                     - 当数据量小于此值时不进行降采样
                     - 当sample_param > 1或为None时，作为固定降采样数量
                     
    返回:
        tuple of (downsampled_data, downsampled_ts, position_index)
        - downsampled_data: pd.Series - 降采样后的数据
        - downsampled_ts: pd.Index - 降采样后的时间戳/索引
        - position_index: np.ndarray - 降采样点在原始数据中的位置索引（整数数组）
    """
    # 如果输入是DataFrame，提取第一列作为数据
    if isinstance(data, pd.DataFrame):
        col_name = data.columns[0]
        series_data = data[col_name].copy()
    else:
        series_data = data.copy()
    
    # 获取原始数据长度
    data_length = len(series_data)
    
    # 如果数据长度小于阈值或降采样方法为None或'none'，则不进行降采样
    if data_length < min_threshold or downsampler is None or downsampler.lower() == 'none':
        position_index = np.arange(data_length, dtype=np.int64)
        if isinstance(data, pd.DataFrame):
            return data.copy(), data.index.copy(), position_index
        return series_data.copy(), series_data.index.copy(), position_index
    
    # 内部处理sample_param参数
    if sample_param is None or sample_param > 1:
        # 如果sample_param为None或大于1，使用min_threshold作为固定降采样数量
        n_out = min_threshold
    elif 0 < sample_param <= 1:  
        # 按比例降采样
        n_out = int(data_length * sample_param)
    else:
        # 不支持的参数值，使用min_threshold作为默认
        logging.warning(f"不支持的sample_param值: {sample_param}，应该在0-1之间或大于1。使用min_threshold作为默认值。")
        n_out = min_threshold
    
    # 确保n_out不超过原始数据长度
    n_out = min(n_out, data_length)
    
    # 如果是M4降采样，确保n_out是4的倍数
    if downsampler.lower() == 'm4':
        n_out = n_out + (4 - n_out % 4) if n_out % 4 != 0 else n_out
    
    # 执行降采样
    return ts_downsample(series_data, downsampler, n_out)


@timer_decorator
def stl_decompose_cpu(data, period=60):
    """
    Decompose time series using STL
    :param ts: pandas.Series
    :param period: int
    :return: pandas.DataFrame
    """
    res = STL(data, period=period).fit()
    return res.trend, res.seasonal, res.resid

@timer_decorator
def stl_decompose_gpu(data,  period=60):
    """
    Decompose time series using STL
    :param data: pandas.Series
    :param stl_type: object
    :param period: int
    :return: pandas.DataFrame

    """
    # 获取设备类型
    global _stl_instance
    _stl_instance = create_stl()  # 自动检测设备并选择后端
    
    data_array = data.values.ravel().reshape(1, -1)
    start_time = time.time()
    seasonal, trend, remainder = _stl_instance.fit(data_array, 
                                                   n_p=period, 
                                                   q_s=7,
                                                   q_t=None, 
                                                   q_l=None, 
                                                   d_s=1, 
                                                   d_t=1, 
                                                   d_l=1, 
                                                   jump_s=1, 
                                                   jump_t=1, 
                                                   jump_l=1
                                                   )
    end_time = time.time()
    print(f"------STL decomposition took {end_time - start_time:.4f} seconds")
    ts = data.index
    res = pd.DataFrame({
        'trend': trend[0],
        'seasonal': seasonal[0],
        'residual': remainder[0]
    }, index=ts)
    return res.trend, res.seasonal, res.residual

# def stl_decompose(data, period=60):
#     if torch.cuda.is_available():
#         return stl_decompose_gpu(data, stl_type=stl_backend, period=period)
#     else:
#         return stl_decompose_cpu(data, period=period)

def stl_decompose(data, period=60, device='auto'):
    """
    根据设备类型选择 STL 分解方法（GPU 或 CPU）
    :param data: pandas.Series
    :param period: int
    :param device: str, 选择设备类型，'auto' (自动检测), 'gpu' (强制使用 GPU), 'cpu' (强制使用 CPU)
    :return: pandas.DataFrame
    """
    if device == 'auto':
        # 自动检测设备
        if detect_device() == 'cuda':
            return stl_decompose_gpu(data, period=period)
        else:
            return stl_decompose_cpu(data, period=period)
    elif device == 'gpu':
        # 强制使用 GPU 版本
        if detect_device() == 'cuda':
            return stl_decompose_gpu(data, period=period)
        else:
            raise RuntimeError("GPU 不可用，无法使用 GPU 版本进行分解。")
    elif device == 'cpu':
        # 强制使用 CPU 版本
        return stl_decompose_cpu(data, period=period)
    else:
        raise ValueError("无效的设备类型。请使用 'auto', 'gpu' 或 'cpu'。")
    
def min_max_scaling(data):
    # 计算最小值和最大值
    min_val = np.min(data)
    max_val = np.max(data)
    # 防止除以零的情况
    if max_val == min_val:
        return data

        # 执行0-1化
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


def calculate_group_median_variance(df, column):
    group_median_variances = []

    # 按组遍历 df
    for group, data in df.groupby('group'):
        # 对每个组的数据按时间排序（保证时间序列的顺序）
        data = data.sort_index()

        # 动态计算窗口大小为组内数量的5%，且至少为1
        window_size = max(int(len(data) * 0.05), 1)

        # 计算滑动方差
        rolling_variance = data[column].rolling(window=window_size).var()

        # 取滑动方差的中位数
        median_variance = rolling_variance.median()

        # 存储组编号和对应的中位数方差
        group_median_variances.append({'group': group, 'median_variance': median_variance})

    # 将结果转换为 DataFrame
    group_median_variances_df = pd.DataFrame(group_median_variances)
    return group_median_variances_df


def variance_filter(data,
                    mask='global_mask',
                    method='mean',
                    threshold=0.05):
    df = data.copy()
    column = df.columns[0]
    df['group'] = (df[mask] != df[mask].shift()).cumsum()
    # 筛选出global_mask为0的数据
    df_nonoutlier = df[df[mask] == 0]

    # group_variances = df_nonoutlier.groupby('group')[column].var().reset_index()
    group_variances = calculate_group_median_variance(df_nonoutlier, column)
    # 检查 group_variances 是否为空
    if group_variances.empty:
        print(f'警告: group_variances 为空，跳过 variance_filter 处理')
        return df
    group_variances.columns = ['group', 'variance']
    # group_num = group_variances[group_variances['variance'] < threshold]['group'].values

    #
    # group_variances['variance_log'] = np.log(group_variances['variance'] + 1e-6)

    # # 计算方差的相对变化率
    # group_variances['variance_change'] = group_variances['variance'].pct_change().fillna(0)

    # # 计算 MAD Score
    # mad = np.median(np.abs(group_variances['variance_log'] - np.median(group_variances['variance_log'])))
    # group_variances['mad_score'] = np.abs(group_variances['variance_log'] - np.median(group_variances['variance_log'])) / mad

    # # 标记异常组
    # group_variances['is_anomaly'] = group_variances['mad_score'] > threshold

    # 整体方差和组内方差的差值判断异常
    # overall_var = df_nonoutlier[column].var()
    # group_variances['var_deviation'] = abs(group_variances['variance']) - abs(overall_var)
    # deviation_threshold = 0.5 * abs(overall_var)
    # group_variances['is_anomaly_var'] = abs(group_variances['var_deviation']) < deviation_threshold

    if method == 'mean':
        # 计算每个组的均值，去除5%的最大值和5%的最小值
        group_metric = df_nonoutlier.groupby('group')[column].apply(
            lambda x: x[(x.quantile(0.05) < x) & (x < x.quantile(0.95))].mean()
        ).reset_index(name='mean')
        overall_metric = df_nonoutlier[(df_nonoutlier[column] > df_nonoutlier[column].quantile(0.05)) &
                                       (df_nonoutlier[column] < df_nonoutlier[column].quantile(0.95))][column].mean()
    elif method == 'median':
        # 计算每个组的中位数，去除10%的最大值和10%的最小值
        group_metric = df_nonoutlier.groupby('group')[column].apply(
            lambda x: x[(x.quantile(0.05) < x) & (x < x.quantile(0.95))].median()
        ).reset_index(name='median')
        overall_metric = df_nonoutlier[(df_nonoutlier[column] > df_nonoutlier[column].quantile(0.05)) &
                                       (df_nonoutlier[column] < df_nonoutlier[column].quantile(0.95))][column].median()

    if group_metric.isnull().values.any():
        print('数据片段太小，没办法去除极值')
        overall_metric = df_nonoutlier[column].mean()
        group_metric = df_nonoutlier.groupby('group')[column].mean().reset_index()

    group_metric.columns = ['group', 'group_metric']

    # 将均值合并到方差表中
    group_variances = group_variances.merge(group_metric, on='group', how='left')

    # 计算组均值与整体均值的偏差
    group_variances['metric_deviation'] = abs(group_variances['group_metric'] - overall_metric)

    # 定义均值偏差阈值（例如 ±10%）
    print('---均值偏差阈值---', threshold)
    deviation_threshold = threshold * abs(overall_metric)

    # 综合判断异常：动态阈值 & 均值偏差
    group_variances['is_anomaly'] = (group_variances['metric_deviation'] > deviation_threshold)

    # n = 1.5
    # group_variances['is_anomaly'] = (group_variances['is_anomaly_var']) & (
    #     (group_variances['group_mean'] > n * overall_mean) | (group_variances['group_mean'] < -n * overall_mean)
    # )

    group_num = group_variances[group_variances['is_anomaly'] == True]['group'].values

    print(f'group_num:{group_num}')
    for group in group_num:
        df.loc[df['group'] == group, mask] = 1
        
     # 如果处理后整体异常占比过高或者异常增量过高，则不进行异常标记
    anomaly_ratio = df[mask].sum() / len(df)
    change_ratio = np.array(df[mask].sum()-data[mask].sum()) / len(data)

    if change_ratio>0.3:
        return data.iloc[:, :3]
    else:
        return df.iloc[:, :-1]    
    


def get_true_indices(data):

    # 重置索引，确保索引是连续的自然索引
    df_reset = data.reset_index(drop=True)
    
    # 找到值为 True 的行
    true_rows = df_reset[df_reset[df_reset.columns[0]] == True]
    
    # 获取自然索引
    natural_indices = true_rows.index.tolist()
    
    return natural_indices

def check_state(data, col):
    values_unique = data[col].unique()
    return len(values_unique)


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


def psd_noise_analysis(signal, fs=1.0, noise_freq_threshold=0.1,
                       nperseg=256, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = welch(signal, fs, nperseg=nperseg,
                       noverlap=noverlap, window='hann')
    df         = freqs[1] - freqs[0]
    total_e    = np.sum(psd) * df
    noise_e    = np.sum(psd[freqs >= noise_freq_threshold]) * df
    ratio      = noise_e / total_e if total_e else 0.0
    return ratio, freqs, psd


def wavelet_noise_analysis(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energy = [np.sum(c**2) for c in coeffs]
    total  = sum(energy)
    detail = sum(energy[1:])
    ratio  = detail / total if total else 0.0
    # 仅用近似系数重构
    recon  = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]],
                          wavelet)
    return ratio, recon, energy


def noise_analysis(signal, fs=1.0,nperseg=256,wavelet='db4', level=5, threshold=0.5):
    
    noise_freq_threshold= fs/10
    signal = signal - np.mean(signal)
    psd_ratio,  psd_f,  psd_dens  = psd_noise_analysis(signal, fs, noise_freq_threshold)

    wavelet_ratio, wavelet_recon, wavelet_energy = wavelet_noise_analysis(signal, wavelet, level)

    combined_ratio = (psd_ratio + wavelet_ratio)/2
    is_noisy = combined_ratio > threshold
    if is_noisy:
        return "noise"
    
    
def detect_noise_data(ts_data):
    """检测高燥数据
    
    参数:
        ts_data: 时间序列数据
        
    返回:
        "noise" 如果检测为高燥数据，否则返回空字符串
    """
    try:
        data_filtered = gaussian_filter1d(ts_data, sigma=3)
        analytic_signal = hilbert(data_filtered)
        data_phase = np.angle(analytic_signal)
        res = STL(data_phase, period=60).fit()
        db1 = reconstruct_residuals(res.resid + res.seasonal)

        def normalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        def nsigma_indices(df, window_size=600):
            df = pd.DataFrame(df, columns=['data_db1'])

            global_mean = df['data_db1'].mean()
            global_std = df['data_db1'].std()

            if global_mean != 0:
                df['rolling_mean'] = df['data_db1'].rolling(window=window_size, center=True).mean()
                df['rolling_std'] = df['data_db1'].rolling(window=window_size, center=True).std()

                df['rolling_mean'] = df['rolling_mean'].ffill().bfill()
                df['rolling_std'] = df['rolling_std'].ffill().bfill()

                if ((df['rolling_mean'] >= global_mean * 0.9) &
                    (df['rolling_mean'] <= global_mean * 1.1) &
                    (df['rolling_std'] >= global_std * 0.9) &
                    (df['rolling_std'] <= global_std * 1.1)).all():
                    return 1

        try:
            if np.max(db1) != np.min(db1):
                n = nsigma_indices(normalize(db1))
                if n == 1:
                    return "noise"
        except Exception as e:
            print(f'【高燥数据检测】出现错误：{e}')

    except Exception as e:
        print(f'【高燥数据检测】数据计算出现错误：{e}')
    return ""


@timer_decorator
def is_noisy_data(ts_data, fs=1.0, nperseg=256, wavelet='db4', level=5, threshold=0.45):
    """综合噪声检测函数
    
    参数:
        ts_data: 时间序列数据
        fs: 采样频率
        nperseg: PSD分析窗口大小
        wavelet: 小波类型
        level: 小波分解层级
        threshold: 噪声阈值
        
    返回:
        True 如果检测为噪声数据，否则返回False
    """
    # 调用noise_analysis检测
    noise_result = noise_analysis(ts_data, fs, nperseg, wavelet, level, threshold)
    
    # 调用detect_noise_data检测
    high_noise_result = detect_noise_data(ts_data.values.ravel())
    
    # 任一方法检测到噪声则返回True
    return (noise_result == "noise" or high_noise_result == "noise")

def find_constant_segments(data):
    if not data:
        return []

    left = 0
    right = 1
    segments,continous_list = [],[]
    points = 0
    n = len(data)

    while right < n:

        # 当right到达末尾或值不等于left时记录区间
        if data[right] != data[left]:
            left += 1
            right += 1
        else:
            while right < n and data[right] == data[left]:
                right += 1
            #相同值重复次数超过5次，才认为是一次阶跃
            if right - left > 5:
                segments.append((left, right - 1))
                points += (right - left)
            left = right
            right = left + 1

    return segments,round(points/n,3)

def is_step_by_distribution(data,th=0.01, bins=100, tn=5):
    hist, _ = np.histogram(data, bins=bins)
    tp = np.argsort(hist)[-tn:][::-1]
    trh = np.max(hist)*th
    pairs = [(min(u, v), max(u, v)) for u, v in combinations(tp, 2)]
    for left,right in pairs:
        if (not np.any(hist[left+1:right] < trh)) and (min(hist[left],hist[right]) > trh):
            return False
        elif (sum(x > trh for x in hist) / len(hist)> 0.25):
            return False
    return True


@timer_decorator
def is_step_data(ts_data, zero_threshold=0.8):
    # 阶跃数据检测:首先根据差分后0值比例，若判断连续，则最终类别为连续；若判断为阶跃，则根据分布进一步判断，若为连续，则最终类别为连续；若也判断为阶跃，则最终类别为阶跃。
    values = ts_data.values.tolist()
    _, constant_ratio = find_constant_segments(values)
    if constant_ratio > zero_threshold:
        # 若判为阶跃信号，则根据分布进一步判断
        distribution_label = is_step_by_distribution(values)
        if distribution_label:
            #两种方案均判断为阶跃
            return True
        else:
            #0值比例方案判断为阶跃，分布方案修正为连续
            return False
    else:
        #0值比例方案判断为连续
        return False


@timer_decorator
def calculate_sampling_rate(data):
    """计算时间序列数据的采样率
    
    参数:
        data: pd.DataFrame - 包含时间戳索引的数据
        
    返回:
        median_interval: float - 中位时间间隔(秒)
        sampling_rate: float - 采样率(Hz)
    """
    # 应用前1000行数据进行采样率计算
    if len(data) > 1000:
        data = data.iloc[:1000]
        
    intervals = (data.index[1:] - data.index[:-1]).total_seconds()
    median_interval = np.median(intervals)
    sampling_rate = 1.0 / median_interval if median_interval > 0 else 0
    
    
    return sampling_rate


if __name__ == '__main__':
    
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("检测到 CUDA 环境，使用 GPU 版本的 STL")
        
    else:
        print("没有检测到 GPU，使用 CPU 版本的 STL")
