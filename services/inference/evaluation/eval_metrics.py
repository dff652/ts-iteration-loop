"""
eval_metrics.py - 异常检测评估工具

此模块提供了一系列函数，用于评估异常检测算法的性能，
包括加载数据、计算各种评估指标、批量评估等功能。

使用示例:
    from eval_metrics import batch_evaluate

    result = batch_evaluate(
        truth_dir='./ground_truth',
        data_path='./data',
        point_names=['TI_11101.PV'],
        detect_results=detect_results,
        output_file='./evaluation_results.csv'
    )
"""

import pandas as pd
import numpy as np
import json
import os


def load_ground_truth(truth_file):
    """
    加载真实异常标签
    """
    if not os.path.exists(truth_file):
        print(f"警告: 未找到标注文件 {truth_file}")
        return None
    try:
        with open(truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        anomaly_items = json.loads(ground_truth[0]["conversations"][1]["value"])["detected_anomalies"]
        true_intervals = [info["interval"] for info in anomaly_items]
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return None
    except KeyError as e:
        print(f"错误: 标注文件格式错误 - 缺少键 {e}")
        return None
    except Exception as e:
        print(f"错误: 加载标注文件失败 - {e}")
        return None

    return true_intervals


def load_detection_results(predict_dir):
    """
    加载检测结果
    """
    if not os.path.exists(predict_dir):
        print(f"警告: 未找到检测结果文件 {predict_dir}")
        return None

    with open(predict_dir, 'r', encoding='utf-8') as f:
        detect_results = json.load(f)

    return detect_results


def load_timeseries_data(data_path,point_name):
    """
    加载时间序列数据
    """
    csv_file = os.path.join(data_path,f'{point_name}_ds.csv')

    if not os.path.exists(csv_file):
        print(f"警告: 未找到数据文件 {csv_file}")
        return None

    csv_data = pd.read_csv(csv_file)

    return csv_data


def intervals_to_binary_labels(intervals, length):
    """
    将异常区间转换为二进制标签
    """
    binary_labels = np.zeros(length, dtype=int)

    if intervals is None:
        return binary_labels

    for interval in intervals:
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            start, end = interval
            binary_labels[start:end + 1] = 1

    return binary_labels


def get_harmonic_weight(anomaly_ratio):
    """
    基于异常点数量确定误报和漏报的惩罚权重

    Args:
        anomaly_ratio: 异常点占比 (异常点数 / 总点数)

    Returns:
        mar_weight: 漏报惩罚权重
        far_weight: 误报惩罚权重
    """
    if anomaly_ratio < 0.01:
        mar_weight, far_weight = 0.8, 0.2
    elif anomaly_ratio < 0.05:
        mar_weight, far_weight = 0.7, 0.3
    else:
        mar_weight, far_weight = 0.6, 0.4

    return mar_weight, far_weight

def calculate_combined_metrics(true_intervals, detected_intervals, timeseries_length, consider_anomaly_nums=True):
    """
    计算融合的准确性指标

    参数:
        true_intervals: 真实异常区间列表
        detected_intervals: 检测异常区间列表
        timeseries_length: 时间序列总长度
        consider_anomaly_nums: 计算调和准确度指标时是否考虑异常点数量

    返回:
        dict: 包含各种融合指标的字典
    """
    # 将区间转换为二进制标签
    true_binary = intervals_to_binary_labels(true_intervals, timeseries_length)
    detected_binary = intervals_to_binary_labels(detected_intervals, timeseries_length)

    # 计算混淆矩阵元素
    TP = np.sum((true_binary == 1) & (detected_binary == 1))
    TN = np.sum((true_binary == 0) & (detected_binary == 0))
    FP = np.sum((true_binary == 0) & (detected_binary == 1))
    FN = np.sum((true_binary == 1) & (detected_binary == 0))

    # 基础指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    # 1. F1 Score - Precision和Recall的调和平均
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 2. Fβ Score - 可调整β值来平衡Precision和Recall
    # β > 1: 更重视Recall（适合异常检测，因为漏报代价高）
    # β < 1: 更重视Precision（适合误报代价高的场景）
    # β = 1: 等同于F1 Score
    def f_beta_score(beta):
        if (precision + recall) > 0:
            return (1 + beta ** 2) * precision * recall / ((beta ** 2 * precision) + recall)
        return 0

    f2_score = f_beta_score(2)  # 更重视Recall
    f05_score = f_beta_score(0.5)  # 更重视Precision

    # 3. Balanced Accuracy - 平衡准确率
    # 考虑了Recall和Specificity的平衡
    balanced_accuracy = (recall + specificity) / 2

    # 4. MCC (Matthews Correlation Coefficient) - 马修斯相关系数
    # 范围: [-1, 1]，1表示完美预测，0表示随机预测，-1表示完全相反
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = numerator / denominator if denominator > 0 else 0

    # 5. G-Mean (Geometric Mean) - 几何平均
    # sqrt(Precision * Recall)
    g_mean = np.sqrt(precision * recall) if (precision * recall) >= 0 else 0

    # 6. Kappa系数 - 衡量一致性
    # 范围: [-1, 1]，1表示完全一致，0表示随机一致
    po = (TP + TN) / timeseries_length
    pe_true = ((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / (timeseries_length ** 2)
    kappa = (po - pe_true) / (1 - pe_true) if (1 - pe_true) > 0 else 0

    # 7. Jaccard Index (IoU) - 交并比
    # 范围: [0, 1]
    jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    # 8. Dice Coefficient (F1的另一种形式)
    # 范围: [0, 1]
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    # 9. Overlap Coefficient
    # 范围: [0, 1]
    overlap = TP / min(TP + FN, TP + FP) if min(TP + FN, TP + FP) > 0 else 0

    # 10. Accuracy
    accuracy = (TP + TN) / timeseries_length

    # 11. 基于误报率和漏报率计算的调和准确度
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    mar = FN / (FN + TP) if (FN + TP) > 0 else 0
    if far==1 or mar==1:
        harmonic_accuracy = 0
    else:
        if consider_anomaly_nums:
            anomaly_count = TP + FN
            anomaly_ratio = anomaly_count / timeseries_length if timeseries_length > 0 else 0
            mar_weight, far_weight = get_harmonic_weight(anomaly_ratio)
        else:
            mar_weight, far_weight = 0.7, 0.3
        harmonic_accuracy = 1 - (mar_weight * mar + far_weight * far)

    return {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'Specificity': round(specificity, 3),
        'NPV': round(npv, 3),
        'F1 Score': round(f1_score, 3),
        'F2 Score': round(f2_score, 3),
        'F0.5 Score': round(f05_score, 3),
        'Balanced Accuracy': round(balanced_accuracy, 3),
        'MCC': round(mcc, 3),
        'G-Mean': round(g_mean, 3),
        'Kappa': round(kappa, 3),
        'Jaccard Index': round(jaccard, 3),
        'Dice Coefficient': round(dice, 3),
        'Overlap Coefficient': round(overlap, 3),
        'Accuracy': round(accuracy, 3),
        'FAR': round(far, 3),
        'MAR': round(mar, 3),
        'Harmonic Accuracy': round(harmonic_accuracy, 3)
    }




def batch_evaluate(truth_dir, data_path, point_names, detect_results, output_file=None):
    """
    批量评估多个点位
    """
    all_results = []
    for i, point_name in enumerate(point_names, 1):
        print(f"\n[{i}/{len(point_names)}] 正在评估: {point_name}")
        true_intervals = load_ground_truth(os.path.join(truth_dir,f"{point_name}_ds.csv_annotations.json"))
        detected_intervals = detect_results.get(point_name)
        csv_data = load_timeseries_data(data_path,point_name)
        if csv_data.empty or not true_intervals or detected_intervals is None:
            print(f"未成功获取点位{point_name}相关的真实标签或预测标签或时序数据，请检查。")
            continue
        result = calculate_combined_metrics(true_intervals, detected_intervals,len(csv_data))

        if result is not None:
            result['point_name'] = point_name
            all_results.append(result)

    if output_file and len(all_results) > 0:
        df = pd.DataFrame(all_results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
        return df[["point_name","Harmonic Accuracy"]].to_dict('records')
    else:
        return "无评估结果。"


if __name__ == "__main__":
    print("=" * 70)
    print("融合准确性指标计算工具")
    print("=" * 70)

    ground_truth_dir = './results/ground_truth/'
    prediction_dir = './results/zhlh_100_output_indices_1024_512.json'
    csv_data_dir = './data'
    output_dir = "./eval_results.csv"
    detected_results = load_detection_results(prediction_dir)
    if os.path.exists(ground_truth_dir):
        files = os.listdir(ground_truth_dir)
        point_names = [name.split('_ds.csv_annotations.json')[0] for name in files
                       if name.endswith('_ds.csv_annotations.json')]

        print(f"\n找到 {len(point_names)} 个待评估点位")

        print("\n" + "=" * 70)
        print("开始批量评估...")
        print("=" * 70)

        eval_result = batch_evaluate(
                    ground_truth_dir,
                    csv_data_dir,
                    point_names,
                    detected_results,
                    output_file=output_dir,
        )
        print(f"评估结果：{eval_result}")
    else:
        print("\n未找到 ground_truth 目录")
        print("请确保目录结构正确:")
        print(f"  - {ground_truth_dir} (真实异常标注)")
        print(f"  - {csv_data_dir} (时间序列数据)")
        print(f"  - {prediction_dir} (检测结果)")
