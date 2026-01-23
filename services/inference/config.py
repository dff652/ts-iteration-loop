import json
import time
import argparse
from datetime import datetime
from joblib import Parallel, delayed
import os

# def save_args_to_file(args, save_dir="/opt/results",file_prefix="args_log"):
#     """将运行参数保存到文件中，并在文件名中加入时间戳"""
    
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间并格式化
    
#     file_path = os.path.join(save_dir, f"{file_prefix}_{timestamp}.json")
#     # 将时间戳加入文件名
#     with open(file_path, "w") as f:
#         json.dump(vars(args), f, indent=4)
#     print(f"运行参数已保存到 {file_path}")

# 核心配置字段，这些字段始终从 default_params.json 获取，不保存到历史记录中
CORE_CONFIG_FIELDS = {'config_file', 'data_path', 'label_path', 'fig_path'}

def save_args_to_file(args, point_summary = None, script_name = None, save_dir= None, file_prefix=None, exclude_fields=None):
    """
    将运行参数保存到统一的 JSON 文件中，自动获取当前脚本名称。
    :param args: 参数对象
    :param save_path: 统一的 JSON 文件路径
    :param exclude_fields: 要排除的字段集合，默认排除核心配置字段
    """
    # 获取当前脚本的文件名
    # script_name = os.path.basename(__file__)
    print(f"当前脚本文件名: {script_name}")
    
    if save_dir is None:
        save_dir = os.path.join(args.save_path, args.task_name)
        # 确保保存目录存在
    else:
        save_dir = os.path.join(save_dir, args.task_name)
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    timestamp = datetime.now().strftime("%Y%m%d_%H")  # 获取当前时间并格式化
    save_path = os.path.join(save_dir, f"{file_prefix}_{timestamp}.json")
    
    # 检查 JSON 文件是否存在
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    print(666666)
    # 将当前脚本的参数写入对应键
    if args.task_name not in data:
        data[args.task_name] = []
    
    # 排除核心配置字段，这些字段始终从 default_params.json 获取
    if exclude_fields is None:
        exclude_fields = CORE_CONFIG_FIELDS
    
    args_dict = {k: v for k, v in vars(args).items() if k not in exclude_fields}
    data[args.task_name].append(args_dict)
    
    if point_summary is not None:
        data["point_summary"] = point_summary

    # 保存回 JSON 文件
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"运行参数已保存到 {save_path} 的 {script_name} 部分")

def load_args(file_path):
    """模拟加载配置文件的函数"""
    with open(file_path, "r") as f:
        return json.load(f)
    
    