from iotdb.Session import Session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import os
from tqdm import tqdm
from scipy.ndimage import label
import warnings
from numpy.polynomial.polyutils import RankWarning
# 设置警告过滤器，便捕获RankWarning
warnings.filterwarnings('error', category=RankWarning)


def replace_outliers_by_3sigma(data):
    """
    使用3sigma规则检测异常点。

    参数:
    data : 包含数据的数组。

    返回:
    np.array: 填充异常点后的数据数组副本。
    """
    # 创建数据的副本
    pre_data = np.copy(data)
    pre_data = pre_data.ravel()
    mean_error = np.mean(pre_data)
    std_error = np.std(pre_data)

    # 设置3sigma阈值
    lower_bound = mean_error - 3 * std_error
    upper_bound = mean_error + 3 * std_error

    # 找到超过阈值的异常点的索引
    anomalies = np.where((pre_data < lower_bound) | (pre_data > upper_bound))[0]

    # 用均值填充异常点
    pre_data[anomalies] = mean_error

    return pre_data

def piecewise_linear(data):
    """
    使用分段多项式回归检测跳变。
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # 用0填充NaN值
    data = data.fillna(0)

    # 检查是否包含无穷大值
    if np.any(np.isinf(data)):
        print("数据包含无穷大值，进行处理。")
        data = data.replace([np.inf, -np.inf], 0)

    # data = min_max_scale(df.values).ravel()
    n = len(data)
    # 首尾各去除10%数据点，分别进行2阶多项式拟合

    trim_size = max(1, int(0.1 * n))

    try:
        coefficients_start = np.polyfit(np.arange(trim_size), data[:trim_size], 2, rcond=1e-10)
        coefficients_end = np.polyfit(np.arange(n - trim_size, n), data[-trim_size:], 2, rcond=1e-10)
    except np.linalg.LinAlgError as e:
        print(f"线性拟合出错: {e}")
        return None, None

    # 整体12阶多项式拟合
    for degree in range(12, 3, -1):
        try:
            coefficients = np.polyfit(np.arange(n), data, degree)
            fitted_curve = np.polyval(coefficients, np.arange(n))
            print(f"成功拟合阶数为 {degree} 的多项式")
            break  # 成功拟合后退出循环
        except RankWarning as e:
            print(f"捕获到警告：{e}，尝试降低阶数到 {degree - 1}")

    # 拼接拟合结果
    fitted_curve[:trim_size] = np.polyval(coefficients_start, np.arange(trim_size))
    fitted_curve[n - trim_size:] = np.polyval(coefficients_end, np.arange(n - trim_size, n))

    residual = data - fitted_curve

    return fitted_curve, residual

def iqr_find_anomaly_indices(data, th=1.5, N=600):
    # IQR
    # 计算第一四分位数(Q1)和第三四分位数(Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    # 定义异常值的阈值，通常是1.5倍的IQR
    lower_bound = Q1 - th * IQR
    upper_bound = Q3 + th * IQR
    # 找出超过阈值的异常点的索引
    anomalies = np.where((data < lower_bound) | (data > upper_bound))[0]

    # 初始化一个列表，用于收集异常段的下标
    anomaly_indices = []

    # 遍历异常点
    for i in range(len(anomalies)):
        start = anomalies[i]
        # 检查下一个异常点是否在N个点之内
        if i < len(anomalies) - 1:
            next_anomaly = anomalies[i + 1]
            if next_anomaly - start <= N:
                # 将这一段的下标添加到列表中
                anomaly_indices.extend(list(range(start, next_anomaly + 1)))
            else:
                # 只添加当前异常点的下标
                anomaly_indices.append(start)
        else:
            # 添加最后一个异常点的下标
            anomaly_indices.append(start)

    # 去重并排序
    anomaly_indices = sorted(set(anomaly_indices))

    return np.array(anomaly_indices)


def standardized_find_anomaly_indices(data, th=10, N=600):
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.std(data)

    # 标准化数据
    standardized_data = (data - mean) / std
    lower_bound = -th
    upper_bound = th
    # 找到超过边界的异常点的索引
    anomalies = np.where((standardized_data < lower_bound) | (standardized_data > upper_bound))[0]

    # 初始化一个列表，用于收集异常段的下标
    anomaly_indices = []

    # 遍历异常点
    for i in range(len(anomalies)):
        start = anomalies[i]
        # 检查下一个异常点是否在N个点之内
        if i < len(anomalies) - 1:
            next_anomaly = anomalies[i + 1]
            if next_anomaly - start <= N:
                # 将这一段的下标添加到列表中
                anomaly_indices.extend(list(range(start, next_anomaly + 1)))
            else:
                # 只添加当前异常点的下标
                anomaly_indices.append(start)
        else:
            # 添加最后一个异常点的下标
            anomaly_indices.append(start)

    # 去重并排序
    anomaly_indices = sorted(set(anomaly_indices))

    return np.array(anomaly_indices)


def nsigma_find_anomaly_indices(data, th=3, N=600):
    mean = np.mean(data)
    std = np.std(data)
    # 定义异常值的阈值，通常是3倍的标准差
    lower_bound = mean - th * std
    upper_bound = mean + th * std
    # 找到超过阈值的异常点的索引
    anomalies = np.where((data < lower_bound) | (data > upper_bound))[0]

    # 初始化一个列表，用于收集异常段的下标
    anomaly_indices = []

    # 遍历异常点
    for i in range(len(anomalies)):
        start = anomalies[i]
        # 检查下一个异常点是否在N个点之内
        if i < len(anomalies) - 1:
            next_anomaly = anomalies[i + 1]
            if next_anomaly - start <= N:
                # 将这一段的下标添加到列表中
                anomaly_indices.extend(list(range(start, next_anomaly + 1)))
            else:
                # 只添加当前异常点的下标
                anomaly_indices.append(start)
        else:
            # 添加最后一个异常点的下标
            anomaly_indices.append(start)

    # 去重并排序
    anomaly_indices = sorted(set(anomaly_indices))

    return np.array(anomaly_indices)


def create_outlier_mask(df, anomaly):
    """
    创建一个与DataFrame同型的布尔掩码，标记跳变点。

    参数:
    df : DataFrame，输入的DataFrame。
    anomaly : list，包含变化点的索引列表。

    返回:
    np.ndarray : 与输入DataFrame同型的布尔掩码，变化点位置为True，其余为False。
    """
    outlier_mask = np.full_like(df, False)

    for idx in anomaly[:]:
        outlier_mask[idx] = True

    df_copy = df.copy()
    df_copy.loc[:] = outlier_mask

    return df_copy


# def create_outlier_mask(df, anomaly):
#     """
#     创建一个与DataFrame同型的布尔掩码，标记跳变点。

#     参数:
#     df : DataFrame，输入的DataFrame。
#     anomaly : list，包含变化点的索引列表。

#     返回:
#     np.ndarray : 与输入DataFrame同型的布尔掩码，变化点位置为True，其余为False。
#     """
#     outlier_mask = np.full_like(df, False)


#     for idx in anomaly:
#         outlier_mask[idx] = True
#     print(6666)
#     print(outlier_mask.sum())
#     df_copy = df.copy()
#     df_copy.loc[:] = outlier_mask
#     print(6666)
#     print(df_copy.sum())
#     return df_copy

# def sigma_filtered_and_piecewise_linear(data, th=3, N=600):
#     pre_data = replace_outliers_by_3sigma(data)
#     fitted_curve = piecewise_linear(pre_data,th)
#     anomaly_indices = find_anomaly_indices(data.values.ravel(),fitted_curve, th, N)
#     #print(anomaly_indices)
#     outlier_mask = create_outlier_mask(data,anomaly_indices)
#     print(1111)
#     print(outlier_mask.sum())
#     return outlier_mask,anomaly_indices,fitted_curve,pre_data


def aggregate_anomalies(anomaly_mask, threshold):
    """
    在时间域内聚合接近的异常集群。

    参数:
    - anomaly_mask: 异常的二进制掩码（异常为1，正常为0）。
    - threshold: 认为集群之间接近的最大距离阈值。
    
    返回:
    - aggregated_mask: 更新后的异常掩码，其中聚合了接近的异常。
    """
    # 识别连接组件（异常区域）
    labeled_array, num_features = label(anomaly_mask)

    # 初始化聚合的异常掩码
    aggregated_mask = np.zeros_like(anomaly_mask)

    # 遍历每个检测到的区域
    for i in range(1, num_features + 1):
        region_indices = np.where(labeled_array == i)[0]

        if i < num_features:
            next_region_indices = np.where(labeled_array == i + 1)[0]
            if next_region_indices[0] - region_indices[-1] < threshold:
                # 合并区域
                aggregated_mask[region_indices[0]:next_region_indices[-1] + 1] = 1
            else:
                # 保持分开
                aggregated_mask[region_indices] = 1
        else:
            aggregated_mask[region_indices] = 1

    return aggregated_mask


def get_mask(data, col):
    print('shape of data:', data.shape)
    df_label = data.copy()
    res = pd.DataFrame()
    df_normal = df_label[df_label['mask'] != 1]
    print('shape of df_normal:', df_label.shape)
    for num in df_normal.Group.unique():
        df_group = df_normal[df_normal['Group'] == num]
        print('shape of df_group:', df_group.shape)
        outlier_mask, anomaly_indices, fitted_curve, pre_data = sigma_filtered_and_piecewise_linear(df_group[[col]], 3,
                                                                                                    600)
        print('sum of outlier_mask', outlier_mask.sum())
        adj_thres = 2 / 100 * df_label.shape[0]
        # adj_thres = 20*60/dt   # 20min
        merged_anomalies = aggregate_anomalies(outlier_mask, adj_thres)
        # df_group['mask'] = merged_anomalies
        print('sum of outlier_mask', merged_anomalies.sum())
        # 更新全量数据mask
        df_label.loc[df_group.index, 'mask'] = merged_anomalies

    return df_label


if __name__ == '__main__':
    # 读取数据
    st: str = "2023-06-01 00:59:59"
    # st: str = "2024-04-01 00:59:59"
    et: str = "2024-08-01 23:59:59"

    var_list = ["NB.LJSJ.TT_2A234F.PV", ]

    for sensor_id in tqdm(var_list, desc="Processing sensors"):
        data = read_iotdb(target_column=sensor_id, st=st, et=et)
        # raw = data.values.ravel()

        if data.empty:
            print(f"当前DataFrame为空，跳过{sensor_id}")
            continue  # 跳过此次循环，继续下一个循环迭代

        outlier_mask, anomaly_indices, fitted_curve, pre_data = sigma_filtered_and_piecewise_linear(data, 3, 600)

        adj_thres = 2 / 100 * data.shape[0]
        # adj_thres = 20*60/dt   # 20min
        merged_anomalies = aggregate_anomalies(outlier_mask, adj_thres)

        data.index = pd.to_datetime(data.index, errors='coerce')
        data = data[data.index.notnull()]

        data['mask'] = merged_anomalies
        data['Group'] = (data['mask'] != data['mask'].shift()).cumsum()

        max_value = data.mean() + data.std()
        min_value = data.mean() - data.std()

        print(data['mask'].sum())
        result_data = get_mask(data, sensor_id)
        print(result_data['mask'].sum())

        # res = get_mask(result_data,sensor_id)
        # print(res)
