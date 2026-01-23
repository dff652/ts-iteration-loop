import numpy as np


def dilation(data, kernel_size=6):
    """膨胀操作"""
    pad_width = kernel_size // 2
    padded_data = np.pad(data, pad_width, mode='edge')
    dilated_data = np.copy(data)

    for i in range(len(data)):
        # 计算邻域最大值
        dilated_data[i] = np.max(padded_data[i:i + kernel_size])
    return dilated_data


def erosion(data, kernel_size=6):
    """腐蚀操作"""
    pad_width = kernel_size // 2
    padded_data = np.pad(data, pad_width, mode='edge')
    eroded_data = np.copy(data)

    for i in range(len(data)):
        # 计算邻域最小值
        eroded_data[i] = np.min(padded_data[i:i + kernel_size])
    return eroded_data


def morphological_gradient(data, kernel_size=6):
    """形态学梯度"""
    # 执行膨胀操作
    dilated_data = dilation(data, kernel_size)
    # 执行腐蚀操作
    eroded_data = erosion(data, kernel_size)
    # 计算形态学梯度
    gradient_data = dilated_data - eroded_data
    return gradient_data


def morphological_opening(data, kernel_size=6):
    """形态学开运算"""
    # 执行腐蚀操作
    eroded_data = erosion(data, kernel_size)
    # 执行膨胀操作
    opened_data = dilation(eroded_data, kernel_size)
    return opened_data


def morphological_closing(data, kernel_size=6):
    """形态学闭运算"""
    # 执行膨胀操作
    dilated_data = dilation(data, kernel_size)
    # 执行腐蚀操作
    closed_data = erosion(dilated_data, kernel_size)
    return closed_data
