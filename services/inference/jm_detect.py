import copy
import numpy as np
import pandas as pd
from kneed import KneeLocator
from adtk.data import validate_series
from adtk.transformer import DoubleRollingAggregate
# 兼容不同导入方式（包内相对导入 / 顶层脚本运行）
try:
    from .signal_processing import (
        is_step_data, adaptive_downsample, get_fulldata,
        calculate_sampling_rate, dead_value_detection, get_true_indices
    )
    from .wavelet import split_continuous_outliers
    from .ensemble import create_outlier_mask
except ImportError:
    from signal_processing import (
        is_step_data, adaptive_downsample, get_fulldata,
        calculate_sampling_rate, dead_value_detection, get_true_indices
    )
    from wavelet import split_continuous_outliers
    from ensemble import create_outlier_mask

class MeanShiftDetect:
    """
    基于直方图分布，进行均值漂移检测。
    参数:
        bin_nums (int): 直方图划分的区间个数，默认20，若出现漂移区域未补全情况，可根据数据分布情况调小。
    方法:
        detect(data:pd.Series): data为待检测的单维时序数据
    Returns:
        mean_shift_indices(list): 异常点的自然索引数组，如[3855,3856,8121,8122,8123,11000,11001,11002]
    """
    def __init__(self,bin_nums=20):
        self.bin_nums = bin_nums

    def detect(self,data):
        # 计算直方图
        hist, edges = np.histogram(data, bins=self.bin_nums)
        bin_indices = np.digitize(data, edges) - 1
        # 处理超出范围的值
        bin_indices[bin_indices == -1] = 0  # 小于最小边界的值归入第一个bin
        bin_indices[bin_indices == len(edges) - 1] = len(edges) - 2  # 大于最大边界的值归入最后一个bin

        hist = list(hist)
        if len(hist) == 2:
            return list(np.where(bin_indices == np.argmin(hist))[0])

        # 找到最高频的柱子（主分布）
        main_peak_idx = np.argmax(hist)
        max_bin_center = (edges[main_peak_idx] + edges[main_peak_idx + 1]) / 2
        upper, lower = 1.2 * max_bin_center, 0.8 * max_bin_center

        left_border, right_border = main_peak_idx, main_peak_idx
        # 检测左侧断开点
        i = main_peak_idx - 1
        while i > 0:
            if hist[i] > 0.009 * max(hist):
                i -= 1
            else:
                left_border = i
                break

        # 检测右侧断开点
        j = main_peak_idx + 1
        while j < len(hist):
            if hist[j] > 0.009 * max(hist):
                j += 1
            else:
                right_border = j
                break

        # 合并所有断开点索引
        gap_indices = []
        if left_border in [0, main_peak_idx] and right_border not in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(right_border, len(hist)))
        if left_border not in [0, main_peak_idx] and right_border in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(left_border + 1))
        if left_border not in [0, main_peak_idx] and right_border not in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(left_border + 1)) + list(range(right_border, len(hist)))

        # 修正漂移区域
        left_sum = np.sum([hist[i] for i in gap_indices if i < main_peak_idx])
        right_sum = np.sum([hist[i] for i in gap_indices if i > main_peak_idx])
        if left_sum > hist[main_peak_idx] or right_sum > hist[main_peak_idx]:
            revised_gap_indices = [main_peak_idx]
        elif left_sum / hist[main_peak_idx] > 0.85:
            revised_gap_indices = [i for i in gap_indices if i > main_peak_idx]
        elif right_sum / hist[main_peak_idx] > 0.85:
            revised_gap_indices = [i for i in gap_indices if i < main_peak_idx]
        else:
            revised_gap_indices = gap_indices

        # 标记断开点
        outlier_indices = []
        for bin_idx in revised_gap_indices:
            indices = np.where(bin_indices == bin_idx)[0]
            outlier_indices.extend(indices)

        # 删去持续时间较短的离群点
        mean_shift_indices = []
        outlier_indices.sort()
        groups = split_continuous_outliers(outlier_indices)
        for group in groups:
            group_data = data.iloc[group]
            if len(group) > 500 and ((np.mean(group_data) > upper) or (np.mean(group_data) < lower)):
                mean_shift_indices.extend(group)
        return mean_shift_indices
    
    def detect_adapt(self,data):
        # 计算直方图
        hist, edges = np.histogram(data, bins=self.bin_nums)
        bin_indices = np.digitize(data, edges) - 1
        # 处理超出范围的值
        bin_indices[bin_indices == -1] = 0  # 小于最小边界的值归入第一个bin
        bin_indices[bin_indices == len(edges) - 1] = len(edges) - 2  # 大于最大边界的值归入最后一个bin

        hist = list(hist)
        if len(hist) == 2:
            return list(np.where(bin_indices == np.argmin(hist))[0])

        # 找到最高频的柱子（主分布）
        main_peak_idx = np.argmax(hist)
        max_bin_center = (edges[main_peak_idx] + edges[main_peak_idx + 1]) / 2
        upper, lower = 1.2 * max_bin_center, 0.8 * max_bin_center

        left_border, right_border = main_peak_idx, main_peak_idx
        lgh = np.log1p(np.array(hist))
        th = np.expm1(self.knpt(lgh,0))


        i = main_peak_idx-1
        while i > 0:
            if hist[i] > th:
                i -= 1
            else:
                left_border = i
                break

        # 检测右侧断开点
        # right_gaps = []
        j = main_peak_idx + 1
        while j < len(hist):
            if hist[j] > th:
                j += 1
            else:
                right_border = j
                break

        # 合并所有断开点索引
        gap_indices = []
        if left_border in [0, main_peak_idx] and right_border not in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(right_border, len(hist)))
        if left_border not in [0, main_peak_idx] and right_border in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(left_border + 1))
        if left_border not in [0, main_peak_idx] and right_border not in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(left_border + 1)) + list(range(right_border, len(hist)))

        # 修正漂移区域
        left_sum = np.sum([hist[i] for i in gap_indices if i < main_peak_idx])
        right_sum = np.sum([hist[i] for i in gap_indices if i > main_peak_idx])
        if left_sum > hist[main_peak_idx] or right_sum > hist[main_peak_idx]:
            revised_gap_indices = [main_peak_idx]
        elif left_sum / hist[main_peak_idx] > 0.85:
            revised_gap_indices = [i for i in gap_indices if i > main_peak_idx]
        elif right_sum / hist[main_peak_idx] > 0.85:
            revised_gap_indices = [i for i in gap_indices if i < main_peak_idx]
        else:
            revised_gap_indices = gap_indices

        # 标记断开点
        outlier_indices = []
        for bin_idx in revised_gap_indices:
            indices = np.where(bin_indices == bin_idx)[0]
            outlier_indices.extend(indices)

        # 删去持续时间较短的离群点
        mean_shift_indices = []
        outlier_indices.sort()
        groups = split_continuous_outliers(outlier_indices)
        for group in groups:
            group_data = data.iloc[group]
            if len(group) > 500 and ((np.mean(group_data) > upper) or (np.mean(group_data) < lower)):
                mean_shift_indices.extend(group)
        return mean_shift_indices
    

    def detect_patch(self,data):
        tn=5
        hist, bin_edges = np.histogram(data, bins=self.bin_nums)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_indices = np.digitize(data, bin_edges) - 1
        bin_indices[bin_indices == -1] = 0
        bin_indices[bin_indices == len(bin_edges) - 1] = len(bin_edges) - 2
        
        main_peak_idx = np.argmax(hist)
        max_bin_center = (bin_edges[main_peak_idx] + bin_edges[main_peak_idx + 1]) / 2
        upper, lower = 1.2 * max_bin_center, 0.8 * max_bin_center
        
        tp = np.argsort(hist)[-tn:][::-1]

        lgh = np.log1p(np.array(hist))
        trh = np.expm1(self.knpt(lgh,0))
        th_hist = [x for x in hist if x>trh]
        th_idx = [np.where(hist==x)[0][0] for x in hist if x>trh]

        ht = self.idx_groups(th_idx)

        avgh = np.array([sum([hist[xx] for xx in cl]) for cl in ht])
        mh = np.argmax(avgh)
        sf = [y for i, li in enumerate(ht) if i != mh for y in li]


        filtered_indices = []
        anomaly_indices = np.where(np.isin(bin_indices, sf))[0]
        anomaly_indices.sort()
        groups = split_continuous_outliers(anomaly_indices)
        for group in groups:
            group_data = data.iloc[group]
            if len(group) > 500 and ((np.mean(group_data) > upper) or (np.mean(group_data) < lower)):
                filtered_indices.extend(group)
        return filtered_indices

    def knpt(self,scores,mth='auto'):
        from kneed import KneeLocator
        sorted_scores = np.sort(scores)
        indices = np.arange(len(sorted_scores))
        kn = KneeLocator(indices, sorted_scores, curve='convex', direction='increasing')
        kne = kn.knee

        if mth == 'auto':
            if kne:
                print(f"变化量个数：{len(scores)},拐点位置：{kne+1}")
                trh = sorted_scores[kne]
            else:
                # trh = sorted_scores[np.argmax(sorted_scores[1:] / (sorted_scores[:-1] + 0.001))]
                kne2 = self.get_knee2(sorted_scores,indices)
                trh = sorted_scores[kne2]
        else:
            kne2 = self.get_knee2(sorted_scores,indices)
            trh = sorted_scores[kne2]
        return trh

    def get_knee2(self,y1,x1):
        AA = y1[0]-y1[-1]
        BB = x1[-1]
        CC = -x1[-1]*y1[0]


        dist = (AA*x1 + BB*y1 + CC) / np.sqrt(AA**2 + BB**2)
        idx = np.argmax(np.abs(dist))
        knx = x1[idx]
        return knx


    def idx_groups(self,li):
        if not li:
            return []
        groups, group = [], [li[0]]
        for i, j in zip(li, li[1:]):
            (group.append(j) if j == i + 1 else (groups.append(group), group := [j]))
        return groups + [group]


    def ind_split(self,arr):
        if len(arr) == 0:
            return []
        split_indices = np.where(np.diff(arr) != 1)[0] + 1
        sub_arrays = np.split(arr, split_indices)
        return sub_arrays

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


def process_data(raw_data,col,sampler,sr,min_nums):
    full_data = get_fulldata(raw_data, col)
    sampling_rate = calculate_sampling_rate(full_data[col])
    downsampled_data, ts, position_index = adaptive_downsample(
        raw_data[col],
        downsampler=sampler,
        sample_param=sr,
        min_threshold=min_nums
    )
    ds_data = pd.DataFrame(downsampled_data, index=ts)
    ds_data = ds_data[~ds_data.index.duplicated(keep='first')]
    return ds_data, position_index


def adtk_hbos_detect(data: pd.DataFrame,
                    downsampler: str = 'm4',
                    sample_param: float = 0.1,
                    bin_nums: int = 20,
                    min_threshold: int = 200000,
                    ratio=None):
    """
    对时序数据进行：跳变、均值漂移、死值三种类型的异常检测。

    参数:
        data: pd.DataFrame - 输入数据
        downsampler: str - 降采样方法，支持'm4', 'minmax', 'none'或None，为None或'none'时不降采样
        sample_param: float - 采样参数：
                    - 0到1之间按比例降采样(例如0.1表示保留10%)
                    - 大于1时自动使用min_threshold作为固定降采样数量
        bin_nums: int - 直方图划分的区间个数，默认20，若出现漂移区域未补全情况，可根据数据分布情况调小。
        min_threshold: int - 两个用途:
                       1. 当数据量小于此值时不进行降采样
                       2. 当sample_param > 1时，作为固定降采样数量
        ratio: float - 过滤跳变小的异常的阈值，默认算法自适应，若传入该参数，则以传入的数值为准，传入范围[0,1]。

    返回:
        raw_data: 与原始分辨率等长的 DataFrame（布尔/掩码列为点位名），用于直接叠加到原始曲线
        position_index: 本次检测时所用的降采样位置索引（用于保存为 downsample 模式时写入 orig_pos）
    """
    # 统一索引为 DatetimeIndex，避免与 Timestamp 比较时报错
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(
                f"Index must be convertible to DatetimeIndex for adtk_hbos. "
                f"Current index type: {type(data.index)}. Error: {e}"
            )

    # raw_data = copy.deepcopy(data)
    raw_data = data.copy()
    step_type = is_step_data(data)
    point_name = data.columns[0]
    processed_data, position_index = process_data(data, point_name, downsampler, sample_param, min_threshold)
    # 确保降采样数据包含原列名
    if point_name not in processed_data.columns:
        # 将唯一一列重命名为点位名
        if processed_data.shape[1] == 1:
            processed_data.columns = [point_name]
        elif processed_data.shape[1] == 0:
            processed_data[point_name] = np.nan
        else:
            processed_data[point_name] = processed_data.iloc[:, 0]
    global_indices = np.array([], dtype=int)
    if not step_type:
        abd = AnomalyBordersDetector(ratio=ratio)
        msd = MeanShiftDetect(bin_nums=bin_nums)
        jump_anomalies = abd.detect(processed_data[point_name])
        mean_shift_anomalies = msd.detect(processed_data[point_name])
        if not mean_shift_anomalies:
            mean_shift_anomalies = msd.detect_adapt(processed_data[point_name])
            if not mean_shift_anomalies:
                mean_shift_anomalies = msd.detect_patch(processed_data[point_name])
        merged_indices = list(set(list(jump_anomalies) + mean_shift_anomalies))

        # 加入长时间死值检测
        _, dead_df, _ = dead_value_detection(processed_data, 72000)
        dead_per = dead_df.sum() / len(dead_df)
        dead_per = dead_per.values[0]
        if dead_per < 0.7:  # 死值比例小于0.7进行标记，大于0.7不进行标记
            natural_indices = get_true_indices(dead_df)
            global_list = merged_indices + natural_indices
            global_indices = np.unique(global_list)
        else:
            global_indices = merged_indices

        global_mask = create_outlier_mask(processed_data, global_indices)
        processed_data['global_mask'] = global_mask

    else:
        # 对于阶梯型或噪声型数据，创建空掩码
        global_mask = create_outlier_mask(processed_data, global_indices)
        processed_data['global_mask'] = global_mask

    processed_data['Group'] = (processed_data['global_mask'] != processed_data['global_mask'].shift()).cumsum()

    # 处理每一组 Group
    grouped = processed_data[processed_data['global_mask'] == 1].groupby('Group')

    # 回映射到原始分辨率：新增一列布尔标记
    raw_data[point_name] = False
    for name, group in grouped:
        start_time = group.index.min()
        end_time = group.index.max()
        raw_data.loc[(raw_data.index >= start_time) & (raw_data.index <= end_time), point_name] = True

    return raw_data, position_index

