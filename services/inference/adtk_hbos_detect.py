import copy
import numpy as np
import pandas as pd
from kneed import KneeLocator
from adtk.data import validate_series
from adtk.transformer import DoubleRollingAggregate
from signal_processing import read_iotdb,ts_downsample,get_fulldata,calculate_sampling_rate
from wavelet import split_continuous_outliers
from ensemble import create_outlier_mask

class AnomalyBordersDetector:
    """
    跳变边界检测，包括整体跳变和局部跳变。
    参数:
        ratio (float): 过滤跳变小的异常的阈值，默认算法自适应，若传入该参数，则以传入的数值为准，传入范围[0,1]。
    方法:
        detect(data:pd.Series): data为待检测的单维时序数据
    Returns:
        jump_indices(list): 异常点的自然索引数组，如[3855,3856,8121,8122,8123,11000,11001,11002]
    """
    def __init__(self, ratio=None):
        self.ratio = ratio

    def nsigma_find_anomaly_indices(self,data, th=3):
        clean_data = data[data != 0]
        mean = np.mean(clean_data)
        std = np.std(clean_data)
        lower_bound = mean - th * std
        upper_bound = mean + th * std
        anomalies = np.where((data < lower_bound) | (data > upper_bound))[0]
        return list(anomalies)

    def adtk_drift_detect(self,data):
        ts = validate_series(data)
        s_transformed_fast = DoubleRollingAggregate(
            agg="mean",
            window=(100, 3),  # The tuple specifies the left window to be 3, and right window to be 1
            diff="l1").transform(ts).rename("before100_after3")
        s_transformed_middle = DoubleRollingAggregate(
            agg="mean",
            window=(3000, 100),  # The tuple specifies the left window to be 3, and right window to be 1
            diff="l1").transform(ts).rename("before3000_after100")
        s_transformed_slow = DoubleRollingAggregate(
            agg="mean",
            window=(100000, 2000),  # The tuple specifies the left window to be 3, and right window to be 1
            diff="l1").transform(ts).rename("before100000_after2000")
        s_transformed_fast = s_transformed_fast.fillna(0)
        fast_anomalies = self.nsigma_find_anomaly_indices(s_transformed_fast, th=5)
        s_transformed_middle = s_transformed_middle.fillna(0)
        middle_anomalies = self.nsigma_find_anomaly_indices(s_transformed_middle, th=5)
        s_transformed_slow = s_transformed_slow.fillna(0)
        slow_anomalies = self.nsigma_find_anomaly_indices(s_transformed_slow, th=5)
        anomalies = list(set(fast_anomalies + middle_anomalies + slow_anomalies))
        return anomalies

    def hbos(self,data, bins=50):
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_width = bin_edges[1] - bin_edges[0]

        densities = hist / (len(data) * bin_width)
        densities = np.clip(densities, 1e-6, None)

        scores = np.zeros_like(data)

        bin_indices = np.digitize(data, bin_edges) - 1
        # 处理超出范围的值
        bin_indices[bin_indices == -1] = 0  # 小于最小边界的值归入第一个bin
        bin_indices[bin_indices == len(bin_edges) - 1] = len(bin_edges) - 2  # 大于最大边界的值归入最后一个bin

        for bin_idx in range(bins):
            indices = np.where(bin_indices == bin_idx)[0]
            scores[indices] = -np.log(densities[bin_idx])

        drops = np.diff(scores)
        idx = np.argmax(drops)
        threshold = scores[idx + 1]
        anomaly_indices = np.where(scores > threshold)[0]
        anomaly_indices = np.sort(anomaly_indices)

        return anomaly_indices

    def get_adapted_ratio(self,values):
        chunks = 200
        global_change = max(values) - min(values)
        split_points = np.linspace(0, len(values), chunks + 1, dtype=int)
        change_ratio_list = []
        for i in range(len(split_points) - 1):
            split_values = np.array(values)[split_points[i]:split_points[i + 1]]
            per_ratio = (max(split_values) - min(split_values)) / global_change
            change_ratio_list.append(per_ratio)
        change_ratio_list.sort()
        indices = np.arange(len(change_ratio_list))
        kn = KneeLocator(indices, change_ratio_list, curve='convex', direction='increasing')
        kne = kn.knee
        if kne:
            trh = change_ratio_list[kne]
        else:
            trh = change_ratio_list[np.argmax(change_ratio_list[1:] / (change_ratio_list[:-1] + 0.001))]
        return trh

    def detect(self,data):
        #基于hbos检测整体跳变，基于adtk检测局部跳变
        border_indices_with_density = self.hbos(data.values.tolist())
        border_indices_with_adtk = self.adtk_drift_detect(data)

        #整体跳变和局部跳变合并，并通过阈值自适应或设定阈值过滤较小异常的跳变
        jump_indices = []
        merged_anomaly_indices = list(set(list(border_indices_with_density) + border_indices_with_adtk))
        merged_anomaly_indices.sort()
        anomaly_groups = split_continuous_outliers(merged_anomaly_indices)

        #阈值自适应
        if not self.ratio and len(np.unique(data)) > 1:
            self.ratio = self.get_adapted_ratio(data)

        #过滤极差比低于阈值的异常
        global_change = max(data) - min(data)
        for group in anomaly_groups:
            if len(group) == 1:
                range_data = data.iloc[group[0] - 10:group[0] + 10]
                local_change = max(range_data) - min(range_data)
            else:
                local_change = max(data.iloc[group]) - min(data.iloc[group])
            if local_change / global_change < self.ratio:
                continue
            jump_indices.extend(group)
        return jump_indices

def read_data(sensor_info):
    path, column, st, et = sensor_info
    raw_data = read_iotdb(path=path, target_column=column, st=st, et=et)
    full_data = get_fulldata(raw_data, column)
    sampling_rate = calculate_sampling_rate(full_data[column])
    ds_data = downsample_data(full_data,column)
    ds_data = ds_data[~ds_data.index.duplicated(keep='first')]
    return ds_data,sampling_rate

def downsample_data(df,col):
    n_out = int(df.shape[0] * 0.1)
    # 确保 n_out 是 4 的倍数，如果不是，向上取整为最近的 4 的倍数
    n_out = n_out + (4 - n_out % 4) if n_out % 4 != 0 else n_out
    downsampled_data, ts, _ = ts_downsample(df[col], "m4", n_out)
    return pd.DataFrame(downsampled_data,index=ts)

if __name__ == '__main__':
    abd = AnomalyBordersDetector()
    sensor_infos = [
        ('root.gdsh_second.gdsh_second', '2501TI10036.PV', '2023-01-01 00:00:00', '2024-02-01 00:00:00'),
        # ('root.supcon.nb.whlj.LJSJ', 'NB.LJSJ.FIC_2A221A.PV', '2023-06-01 00:00:00', '2024-08-01 00:00:00'),
        # ('root.zhlh_202307_202412.ZHLH_4C_1216', 'FV_11301B.OUT', '2023-07-18 12:00:00', '2024-11-05 23:59:59')
    ]
    for info in sensor_infos:
        current_data, sr = read_data(info)
        jump_anomalies = abd.detect(current_data[info[1]])
        jump_mask = create_outlier_mask(current_data, jump_anomalies)
        result_data = copy.deepcopy(current_data)
        result_data["jump_mask"] = jump_mask
        print(result_data.head(5))
