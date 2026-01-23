import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pywt


def reconstruct_residuals(data, wavelet='db1', level=3):
    """
    使用小波分解的细节系数重构信号残差。

    参数:
    time_series_data -- 时间序列数据
    wavelet -- 小波名称，默认为 'db4'
    level -- 分解层数，默认为 3

    返回:
    reconstructed_residuals -- 仅使用细节系数重构的残差信号
    """
    # 执行小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # 提取细节系数
    cA, *cD = coeffs
    # 仅使用细节系数重构残差信号，需要构建一个包含所有细节系数的列表
    reconstruct_data = pywt.waverec([None] + cD, wavelet)

    return reconstruct_data


def split_continuous_outliers(outlier_indices, min_size=1,gp=1):
    split_indices = []
    current_split = []

    for i in range(len(outlier_indices)):
        if not current_split or outlier_indices[i] == current_split[-1] + gp:
            current_split.append(outlier_indices[i])
        else:
            if len(current_split) >= min_size:
                split_indices.append(current_split)
            current_split = [outlier_indices[i]]

    if len(current_split) >= min_size:
        split_indices.append(current_split)

    return split_indices


def range_split_outliers(data, outliers, range_th=0.1):
    # 复制数据以避免修改原始数据
    global_data = np.copy(data)

    # 计算全局范围
    global_range = np.max(np.abs(global_data)) - np.min(np.abs(global_data))
    local_outliers = []
    global_outliers = []

    for outlier in outliers:
        # 提取局部数据
        local_data = global_data[outlier]
        # 计算局部范围
        local_range = np.max(np.abs(local_data)) - np.min(np.abs(local_data))

        # 计算极差比
        range_ratio = local_range / global_range

        # 根据极差比判断是全局异常还是局部异常
        if range_ratio >= range_th:
            global_outliers.extend(outlier)
        # elif (range_th - 0.05) < range_ratio < range_th:
        #     local_outliers.extend(outlier)
        else:
            local_outliers.extend(outlier)

    # 将列表转换为numpy数组并返回
    return np.array(global_outliers), np.array(local_outliers)


def extract_outlier_features(data, outliers):
    """
    提取异常值段的统计特征。
    
    Parameters:
        data (np.ndarray or pd.DataFrame): 原始数据
        outliers (list): 异常值索引列表，每个元素是一个连续的索引段
        
    Returns:
        np.ndarray: 特征矩阵，每行对应一个异常值段的特征
    """
    features = []
    
    # 确保数据是numpy数组格式
    if hasattr(data, 'values'):
        # 如果是DataFrame，提取第一列的值
        data_array = data.iloc[:, 0].values if data.shape[1] > 0 else data.values.ravel()
    else:
        # 如果已经是numpy数组，直接使用
        data_array = data.ravel() if hasattr(data, 'ravel') else np.array(data)
    
    for outlier_segment in outliers:
        if len(outlier_segment) == 0:
            continue
            
        # 使用numpy数组索引访问
        local_data = data_array[outlier_segment]
        
        # 基础统计特征
        mean_val = np.mean(local_data)
        std_val = np.std(local_data)
        min_val = np.min(local_data)
        max_val = np.max(local_data)
        range_val = max_val - min_val
        
        # 相对特征（相对于全局数据）
        global_mean = np.mean(data_array)
        global_std = np.std(data_array)
        global_range = np.max(data_array) - np.min(data_array)
        
        # 标准化特征
        mean_ratio = mean_val / global_mean if global_mean != 0 else 0
        std_ratio = std_val / global_std if global_std != 0 else 0
        range_ratio = range_val / global_range if global_range != 0 else 0
        
        # 变异系数
        cv = std_val / abs(mean_val) if mean_val != 0 else 0
        
        # 偏度和峰度（需要足够的数据点）
        if len(local_data) > 3:
            skewness = np.mean(((local_data - mean_val) / std_val) ** 3) if std_val != 0 else 0
            kurtosis = np.mean(((local_data - mean_val) / std_val) ** 4) if std_val != 0 else 0
        else:
            skewness = 0
            kurtosis = 0
        
        # 异常值段的长度特征
        segment_length = len(outlier_segment)
        length_ratio = segment_length / len(data_array)
        
        # 时间连续性特征（相邻点之间的变化）
        if len(local_data) > 1:
            diff_mean = np.mean(np.abs(np.diff(local_data)))
            diff_std = np.std(np.diff(local_data))
        else:
            diff_mean = 0
            diff_std = 0
        
        # 组合所有特征
        # feature_vector = [
        #     mean_val, std_val, min_val, max_val, range_val,
        #     mean_ratio, std_ratio, range_ratio, cv,
        #     skewness, kurtosis, segment_length, length_ratio,
        #     diff_mean, diff_std
        # ]
        
        feature_vector = [
            
            mean_ratio, range_ratio,

        ]
        
        features.append(feature_vector)
    
    return np.array(features)


def cluster_based_outlier_split(data, outliers, n_clusters=2, method='kmeans', random_state=42):
    """
    基于聚类算法将异常值分为全局异常和局部异常。
    
    Parameters:
        data (np.ndarray or pd.DataFrame): 原始数据
        outliers (list): 异常值索引列表，每个元素是一个连续的索引段
        n_clusters (int): 聚类数量，默认为2（全局异常和局部异常）
        method (str): 聚类方法，支持 'kmeans', 'dbscan', 'hierarchical'
        random_state (int): 随机种子
        
    Returns:
        tuple: (global_outliers, local_outliers) 全局异常和局部异常的索引
    """
    if len(outliers) < 2:
        # 如果异常值段太少，直接返回
        if len(outliers) == 1:
            return np.array(outliers[0]), np.array([])
        else:
            return np.array([]), np.array([])
    
    try:
        # 提取特征
        features = extract_outlier_features(data, outliers)
        
        if features.shape[0] < n_clusters:
            # 特征数量少于聚类数，使用简单方法
            return range_split_outliers(data, outliers, range_th=0.1)
        
        # 特征标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            # 自动估计eps参数
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(3, len(features_scaled))).fit(features_scaled)
            distances, _ = nbrs.kneighbors(features_scaled)
            eps = np.percentile(distances[:, -1], 75)
            clusterer = DBSCAN(eps=eps, min_samples=1)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # 执行聚类
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        # 根据聚类结果分类异常值
        # 假设标签0为局部异常，标签1为全局异常
        # 如果使用DBSCAN，-1标签表示噪声点，我们将其归类为局部异常
        local_outliers = []
        global_outliers = []
        
        for i, (outlier_segment, label) in enumerate(zip(outliers, cluster_labels)):
            if label == 0 or label == -1:  # 离群点
                global_outliers.extend(outlier_segment)
            else:  
                local_outliers.extend(outlier_segment)
        
        return np.array(global_outliers), np.array(local_outliers)
        
    except Exception as e:
        print(f"聚类算法执行失败: {e}")
        print("回退到基于阈值的方法")
        return range_split_outliers(data, outliers, range_th=0.1)


def adaptive_outlier_split(data, outliers, method='auto', **kwargs):
    """
    自适应选择异常值分割方法。
    
    Parameters:
        data (np.ndarray or pd.DataFrame): 原始数据
        outliers (list): 异常值索引列表
        method (str): 分割方法，'auto'表示自动选择，'threshold'表示阈值方法，'cluster'表示聚类方法
        **kwargs: 传递给具体方法的参数
        
    Returns:
        tuple: (global_outliers, local_outliers)
    """
    if method == 'auto':
        # 自动选择方法：根据异常值段的数量和特征复杂度
        if len(outliers) < 3:
            method = 'threshold'
        else:
            # 计算特征复杂度
            features = extract_outlier_features(data, outliers)
            if features.shape[0] >= 3:
                method = 'cluster'
            else:
                method = 'threshold'
    
    if method == 'threshold':
        range_th = kwargs.get('range_th', 0.1)
        return range_split_outliers(data, outliers, range_th)
    elif method == 'cluster':
        n_clusters = kwargs.get('n_clusters', 2)
        cluster_method = kwargs.get('cluster_method', 'kmeans')
        random_state = kwargs.get('random_state', 42)
        return cluster_based_outlier_split(data, outliers, n_clusters, cluster_method, random_state)
    else:
        raise ValueError(f"Unsupported method: {method}")


def cv_sort_local_outlier(signal, outliers, th=0.5):
    cv_values = []  # 存储变异系数的列表
    cv_indices = []  # 存储对应下标的列表

    for outlier in outliers:
        local_part = signal[outlier]

        if len(local_part) > 1:
            mean_part = np.mean(local_part)
            if mean_part == 0:
                cv_part = np.std(local_part)
            else:
                cv_part = np.std(local_part) / mean_part
        else:
            cv_part = 0

        cv_values.append(cv_part)
        cv_indices.append(outlier)

    # 根据变异系数排序
    sorted_indices_and_values = sorted(zip(cv_values, cv_indices), reverse=True)
    sorted_cv_values, sorted_cv_indices = zip(*sorted_indices_and_values)

    # 选择异常值的数量
    num_to_select = max(1, int(len(sorted_cv_values) * (1 - th)))
    selected_indices_and_values = sorted_indices_and_values[:num_to_select]
    selected_cv_values, selected_cv_indices = zip(*selected_indices_and_values)

    # 合并并排序选定的异常值索引
    cv_extend_sorted = sorted(list(set([item for idx in selected_cv_indices for item in idx])))
    # print("排序后的变异系数：",selected_cv_values)
    return list(selected_cv_values), cv_extend_sorted


def refine_local_outliers(signal, outliers):
    refined_outliers = []

    for outlier in outliers:
        start_idx = max(outlier[0] - len(outlier), 0)
        end_idx = min(outlier[-1] + len(outlier), len(signal))

        right_parts = signal[start_idx:outlier[0]]
        left_parts = signal[outlier[-1]:end_idx]
        local_parts = signal[outlier]

        # 检查数组长度是否大于1
        if len(right_parts) > 1:
            right_std_dev = np.std(right_parts)
        else:
            right_std_dev = 0  # 或者一个默认值

        if len(left_parts) > 1:
            left_std_dev = np.std(left_parts)
        else:
            left_std_dev = 0  # 或者一个默认值

        if len(local_parts) > 1:
            local_std_dev = np.std(local_parts)
        else:
            local_std_dev = 0  # 或者一个默认值
        # 确定合并方向
        if len(local_parts) == 1:
            combined_parts = np.concatenate([left_parts, local_parts, right_parts])
            combined_start_idx = start_idx
            combined_end_idx = end_idx
        elif not np.isnan(left_std_dev) and (np.isnan(right_std_dev) or left_std_dev > right_std_dev):
            combined_parts = np.concatenate([left_parts, local_parts])
            combined_start_idx = start_idx
            combined_end_idx = end_idx
        elif not np.isnan(right_std_dev):
            combined_parts = np.concatenate([local_parts, right_parts])
            combined_start_idx = start_idx
            combined_end_idx = end_idx
        else:
            combined_parts = local_parts
            combined_start_idx = outlier[0]
            combined_end_idx = outlier[-1]
        # 打印combined_parts的数据长度
        # print(f"Length of combined_parts: {len(combined_parts)}")

        # 以数据长度为窗口，滑窗计算方差
        window_size = len(local_parts)
        max_std = 0
        max_outlier_segment = []

        # 确保滑动窗口的索引是基于原始信号的索引
        for i in range(len(combined_parts) - window_size + 1):
            window = combined_parts[i:i + window_size]
            if len(window) > 1:
                window_std = np.std(window)
                if window_std > max_std:
                    max_std = window_std
                    max_outlier_segment = list(range(combined_start_idx + i, combined_start_idx + i + window_size))

        if max_outlier_segment:
            refined_outliers.extend(max_outlier_segment)
    return np.array(refined_outliers)


def combine_local_outliers(signal, outliers):
    refined_outliers = []

    for outlier in outliers:
        # 确定局部区域的起始和结束索引
        start_idx = max(outlier[0] - len(outlier), 0)
        end_idx = min(outlier[-1] + len(outlier), len(signal))
        print('起始', start_idx, end_idx)

        # 提取左右和局部部分
        right_parts = signal[start_idx:outlier[0]]
        left_parts = signal[outlier[-1]:end_idx]
        local_parts = signal[outlier]

        # 计算标准差
        right_std_dev = np.std(right_parts) if len(right_parts) > 1 else 0
        left_std_dev = np.std(left_parts) if len(left_parts) > 1 else 0
        local_std_dev = np.std(local_parts) if len(local_parts) > 1 else 0

        # 向左扩展
        extend_count_left = 0
        prev_left_std_dev = left_std_dev  # 初始化前一次的标准差
        while start_idx > 0 and extend_count_left < 10:
            start_idx = max(start_idx - len(local_parts), 0)
            left_parts = signal[start_idx:outlier[0]]
            left_std_dev = np.std(left_parts) if len(left_parts) > 1 else 0
            if left_std_dev > prev_left_std_dev:
                prev_left_std_dev = left_std_dev
                extend_count_left += 1
                print('combined-left', start_idx)
            else:
                break

        # 向右扩展
        extend_count_right = 0
        prev_right_std_dev = right_std_dev  # 初始化前一次的标准差
        while end_idx < len(signal) and extend_count_right < 10:
            end_idx = min(end_idx + len(local_parts), len(signal))
            right_parts = signal[outlier[-1]:end_idx]
            right_std_dev = np.std(right_parts) if len(right_parts) > 1 else 0
            if right_std_dev > prev_right_std_dev:
                prev_right_std_dev = right_std_dev
                extend_count_right += 1
                print('combined-right', end_idx)
            else:
                break

        # 合并区域
        combined_parts = np.concatenate([left_parts, local_parts, right_parts])
        combined_start_idx = start_idx
        combined_end_idx = end_idx

        # 将合并的区域作为新的refine
        refined_outliers.extend(list(range(combined_start_idx, combined_end_idx)))
        print('!' * 20)

    return np.array(refined_outliers)


def exclude_indices(data, indices):
    if len(indices) == 0:  # 检查indices是否为空
        print('indices为空:',indices)
        return np.arange(data.shape[0])
    
    idx_arr = np.array(indices,dtype=int)
    mask = np.ones(data.shape[0], dtype=bool)
    mask[idx_arr] = False
    return np.where(mask)[0]


def fit_and_replace_outliers(data, outliers, degree=2):
    continuous_outliers_groups = split_continuous_outliers(outliers)
    data_fitted = np.copy(data)

    for group in continuous_outliers_groups:
        x_values = np.array(group)
        y_values = data[x_values]
        if len(x_values) > 1:
            p_values = Polynomial.fit(x_values, y_values, degree)
            data_fitted[x_values] = p_values(x_values)

    return data_fitted
