


from iotdb.Session import Session
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download


from gluonts.dataset.repository import dataset_recipes

from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.plot import plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from sklearn.ensemble import IsolationForest
import os
import functools
import time 
import json 

def load_config(config_filename):
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的绝对路径
    config_path = os.path.join(script_dir, 'configs', config_filename)

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def timer_decorator(func):
    """装饰器用于记录函数运行时间"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Function {func.__name__!r} took {run_time:.4f} seconds")
        return result

    return wrapper_timer

@timer_decorator
def read_iotdb(
        host="192.168.199.185",
        port="6667",
        user='root',
        password='root',
        path='root.supcon3.one.dev1',
        target_column='*',
        st='2023-01-01 00:00:00',
        et='2023-02-01 00:00:00',
        limit=100000000,

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

    if discontinuity_threshold is None or discontinuity_threshold == '':
        # 根据数据中的时间间隔推断采样频率
        print(interval_seconds.mode())
        discontinuity_threshold = interval_seconds.mode()[0]
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


@timer_decorator
def get_dataset(col_name,
                st,
                et, 
                path= 'root.hbsn.ops.second.hbsn_second',):
    # muti_var_list =['AI_2010.PV']
    # muti_var_list =['TI_7007B.PV']
    # path ='root.nmxf.all.nmxf'

    # for sensor_id in muti_var_list:
    #     st: str = "2024-01-15 00:00:00"
    #     et: str = "2024-01-20 00:00:00"
    
    print('-' * 40 + '开始' + '-' * 40)
    print(f'\n开始读取数据{col_name},时间段{st}到{et}')

    data = read_iotdb(
        target_column=col_name,
        path=path,
        st=st,
        et=et)
    print(f'原始数据：{data.head()}')
    print(f'原始数据大小：{data.shape}')
    
    result_df, continuity = check_time_continuity(data)
    print('*'*40)
    print(f'填充前缺失时间戳：{result_df}')

    df = data.copy()

    missing_timestamps  = continuity[continuity > 0].index
    print(f'缺失时间戳:\n {missing_timestamps}')
    
    # for ts in missing_timestamps:
    #     if ts not in df.index:
    # df.loc[missing_timestamps] = np.nan

    full_time_index = df.index.append(pd.DatetimeIndex(missing_timestamps)).unique()
    df = df.reindex(full_time_index)
    
    df.sort_index(inplace=True)
    
    df[col_name] = df[col_name].fillna(method='bfill')

    result_df, continuity = check_time_continuity(df)
    print('*'*40)
    print(f'填充前缺失时间戳：{result_df}')

    return df

@timer_decorator
def get_forecasts(data,
                  csv_path, 
                  column,
                  SIZE = "large",
                  PDT = 500,
                  CTX = 5000,
                  PSZ = "auto",
                  BSZ = 64,
                  TEST = 50000):
    
    
    
    
    # SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
    # PDT = 500 # prediction length: any positive integer
    # CTX = 5000    # context length: any positive integer
    # PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    # BSZ = 64 # batch size: any positive integer
    
    # num = data.shape[0] / PDT
    
    # TEST = int(num) * PDT     # test set length: any positive integer
    
    # diff = data.shape[0] - TEST
    
    # raw_data = data.iloc[diff:,:]
    
    ds = PandasDataset(dict(data))
    
    train, test_template = split(ds, offset=-TEST)

    # 生成滚动窗口的测试集实例
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # 每次预测的步长
        windows = (TEST // PDT) ,  # 滚动窗口的数量
        distance=PDT,  # 窗口之间的距离，设为预测长度PDT，以便没有重叠
    )

    # train_data = train.generate_instances(
    #     prediction_length=PDT,  # 每次预测的步长
    #     windows=TEST // PDT,  # 滚动窗口的数量
    #     distance=PDT,  # 窗口之间的距离，设为预测长度PDT，以便没有重叠
    # )
    model_path = f'/opt/ts_models/better464/moirai-1___0-R-{SIZE}'
    # 准备预训练模型
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(model_path),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=50,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

    # 创建预测器
    predictor = model.create_predictor(batch_size=BSZ)
    
    
    forecasts = predictor.predict(test_data.input)

    all_forecasts = []

    # 迭代获取所有的预测结果
    for forecast in forecasts:
        # print(forecast.samples.shape)
        # `forecast.samples` 通常是一个包含多个采样的数组，取均值或中位数作为最终预测值
        forecast_mean = forecast.samples.mean(axis=0)  # 对样本取均值，得到点预测
        all_forecasts.append(forecast_mean)

    all_forecasts_array = np.array(all_forecasts).reshape(-1, 1)

    print(all_forecasts_array.shape)
    
    ypred_res = pd.DataFrame(all_forecasts_array)
    
    ypred_res.to_csv(csv_path)
    
    return ypred_res 

@timer_decorator
def detect_isolation_forest(data, outlier_ratio=0.01):
    """
    使用孤立森林检测异常值。
    参数:
        data : 输入的数据
        outlier_ratio : 异常值的比例
    返回:
        异常值的布尔掩码
        
    """
    isolation_forest = IsolationForest(contamination=outlier_ratio)
    outliers = isolation_forest.fit_predict(data.values.reshape(-1, 1))
    df_outliers = pd.DataFrame(outliers, index=data.index, columns=['outliers'])
    outlier_mask = df_outliers['outliers'] == -1

    return outlier_mask

@timer_decorator
def plot_global(data, col, save_path, method='1'):

    
    plt.figure(figsize=(30, 12))
    
    # 确保数据索引是时间类型
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
        
    # 绘制原始数据曲线，使用索引序列号作为横坐标
    data_len = data.shape[0]
    plt.plot(range(data_len) ,data[col], label='Original Data')
    
    # 根据 mask 列的值填充异常点的区域
    plt.fill_between(range(data_len), data[col].min()*1.02, data[col].max()*1.02, 
                     where=data['outlier_mask'] == 1, color='red', alpha=0.3, label='Anomalies')

    xticks = np.linspace(0, data_len - 1, num=10, dtype=int)  # 取 10 个间隔均匀的点
    xtick_labels = data.index[xticks].to_pydatetime()  # 对应的时间标签

    plt.xticks(xticks, [label.strftime('%Y-%m-%d %H:%M:%S') for label in xtick_labels], rotation=45, ha='right')
    
    # plt.xticks(rotation=45)
    plt.legend(loc="upper right")
    plt.title(f'Sensor ID: {col} Method {method}')  # 可以添加标题显示sensor_id
    
    # 保存图像

    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    
    plt.savefig(save_path, dpi=200)
    plt.close()   
    

def main(path, column, st, et):
    # path =  "root.hbsn.ops.second.hbsn_second"
    # column = "TT_032111A.PV"
    # st: str = "2024-01-01 00:00:00"
    # et: str = "2024-09-01 00:00:00"
    
    
   
    df = get_dataset(column, st, et,path =path)
    
    SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
    PDT = 500 # prediction length: any positive integer
    CTX = 5000    # context length: any positive integer
    PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    BSZ = 64 # batch size: any positive integer
    
    num = df.shape[0] / PDT
    
    TEST = int(num) * PDT     # test set length: any positive integer
    
    diff = df.shape[0] - TEST
    
    raw_data = df.iloc[diff:,:]
    
    outlier_ratio = 0.005
    
    save_path = f'./img/{raw_data.shape[0]}_{column}_{outlier_ratio}_{1}_isof_plot.png'
    
    if not os.path.exists('./img'):
        os.makedirs('./img')
    
    csv_path = f'moirai_forecasts_{column}_{SIZE}_{PDT}_{CTX}.csv'
    
    if os.path.exists(csv_path):
        print(f"预测结果已存在，跳过生成: {csv_path}")
        ypred_res = pd.read_csv(csv_path, index_col=0)
    else: 
        print(f"预测结果不存在，模型预测中: {csv_path}")
        ypred_res  = get_forecasts(raw_data, csv_path, column, SIZE, PDT, CTX, PSZ, BSZ, TEST)
    
    residual =  raw_data.values - ypred_res.values
    
    df_residual = pd.DataFrame(residual,index=raw_data.index)
    
    
    outlier_mask = detect_isolation_forest(df_residual, outlier_ratio = outlier_ratio)  
    
    raw_data['outlier_mask'] = outlier_mask
    
    

    
    if os.path.exists(save_path):
        print(f"图像已存在，跳过生成: {save_path}")
        return
    

    plot_global(raw_data, column, save_path, 4 )


if __name__ == '__main__':
    point_config = load_config('more_few_points_config.json')
    for path, sensor_variables in point_config.items():
            print(f"Processing path: {path}")
            print(f"Sensor variables: {sensor_variables}")
            columns = sensor_variables['columns']
            st = sensor_variables['st']
            et = sensor_variables['et']
                
            for column in columns:  # 每列一个任务
                main(path, column, st, et)