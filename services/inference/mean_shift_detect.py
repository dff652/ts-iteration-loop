import copy
import numpy as np
import pandas as pd

from signal_processing import read_iotdb,ts_downsample,get_fulldata,calculate_sampling_rate
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
        print(f'{len(filtered_indices)}')
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

def mcol(df,target):
    mask = df.astype(str).apply(lambda col: col.str.contains(target, na=False))
    mc = df.columns[mask.any()].tolist()
    return mc[0]

if __name__ == '__main__':
    import os
    print(os.getcwd())
    point_summary = {}
    sensor_infos = []

    df_points=pd.read_json(f'{os.getcwd()}/ilabel/check_outlier/configs/point_config_0722.json')
    
    xx = df_points.loc['columns'].tolist()

    var_list=['TT_032107.PV']
    sensor_infos = [(mcol(df_points,xxx),xxx,
                    df_points.loc['st',mcol(df_points,xxx)],
                    df_points.loc['et',mcol(df_points,xxx)]) for xxx in var_list]
    msd = MeanShiftDetect(bin_nums=20)
    # sensor_infos = [
    #     ('root.gdsh_second.gdsh_second', '2501TI10036.PV', '2023-01-01 00:00:00', '2024-02-01 00:00:00'),
    #     # ('root.supcon.nb.whlj.LJSJ', 'NB.LJSJ.FIC_2A221A.PV', '2023-06-01 00:00:00', '2024-08-01 00:00:00'),
    #     # ('root.zhlh_202307_202412.ZHLH_4C_1216', 'FV_11301B.OUT', '2023-07-18 12:00:00', '2024-11-05 23:59:59')
    # ]

    for info in sensor_infos:
        current_data, sr = read_data(info)
        mean_shift_anomalies = msd.detect(current_data[info[1]])
        print(f"detect：{len(mean_shift_anomalies)}")
        if not mean_shift_anomalies:
            
            mean_shift_anomalies = msd.detect_adapt(current_data[info[1]])
            print(f"adapt：{len(mean_shift_anomalies)}")
            if not mean_shift_anomalies:
                
                mean_shift_anomalies = msd.detect_patch(current_data[info[1]])
                print(f"patch：{len(mean_shift_anomalies)}")
        mean_shift_mask = create_outlier_mask(current_data, mean_shift_anomalies)
        result_data = copy.deepcopy(current_data)
        result_data["mean_shift_mask"] = mean_shift_mask
        print(result_data.head(5))