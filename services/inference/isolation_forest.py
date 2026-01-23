from sklearn.ensemble import IsolationForest
import numpy as np

def detect_isolation_forest(data, outlier_ratio=0.01 , N=1200):
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
    # 使用 np.where 查找 outliers 数组中值为 -1 的索引
    anomalies = np.where(outliers == -1)[0]
    
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