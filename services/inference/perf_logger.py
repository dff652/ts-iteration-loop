"""
@File    :   perf_logger.py
@Time    :   2024/12/09
@Author  :   DouFengfeng
@Version :   1.0.0
@Desc    :   性能跟踪日志模块，用于记录异常检测过程中的各项性能指标

功能说明:
- 记录每个点位的处理信息（点位名称、数据时间范围等）
- 记录各阶段耗时（数据读取、降采样、模型推理、后处理等）
- 记录数据量统计（原始数据量、降采样后数据量）
- 记录 GPU 使用情况（GPU 占用率、显存使用、显存总量）
- 记录异常检测结果统计（异常点数、异常比例）
- 记录系统资源使用（CPU、内存）
"""

import logging
import os
import time
import functools
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

# 尝试导入 GPU 监控库
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# 尝试导入系统监控库
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerfMetrics:
    """性能指标数据类"""
    # === 点位信息 ===
    sensor_path: str = ""                    # IoTDB 路径
    sensor_column: str = ""                  # 测点名称
    start_time: str = ""                     # 数据开始时间
    end_time: str = ""                       # 数据结束时间
    process_id: int = 0                      # 进程 ID
    
    # === 方法信息 ===
    method: str = ""                         # 异常检测方法
    downsampler: str = ""                    # 降采样方法
    device: str = ""                         # 使用的设备 (cpu/cuda:x)
    
    # === 数据量统计 ===
    raw_data_length: int = 0                 # 原始数据长度
    downsampled_data_length: int = 0         # 降采样后数据长度
    downsample_ratio: float = 0.0            # 实际降采样比例
    sampling_rate_hz: float = 0.0            # 数据采样率 (Hz)
    
    # === 耗时统计 (秒) ===
    total_time: float = 0.0                  # 总耗时
    data_read_time: float = 0.0              # 数据读取耗时
    downsample_time: float = 0.0             # 降采样耗时
    preprocess_time: float = 0.0             # 预处理耗时 (插值、类型判断等)
    model_inference_time: float = 0.0        # 模型推理耗时 (LLM 方法)
    stl_decompose_time: float = 0.0          # STL 分解耗时
    anomaly_detect_time: float = 0.0         # 异常检测算法耗时
    postprocess_time: float = 0.0            # 后处理耗时
    save_time: float = 0.0                   # 结果保存耗时
    
    # === GPU 资源 (如果使用 GPU) ===
    gpu_id: int = -1                          # GPU ID
    gpu_name: str = ""                        # GPU 名称
    gpu_utilization_percent: float = 0.0      # GPU 利用率 (%)
    gpu_memory_used_mb: float = 0.0           # 显存使用量 (MB)
    gpu_memory_total_mb: float = 0.0          # 显存总量 (MB)
    gpu_memory_percent: float = 0.0           # 显存使用比例 (%)
    gpu_temperature_c: float = 0.0            # GPU 温度 (°C)
    gpu_power_w: float = 0.0                  # GPU 功耗 (W)
    
    # === 系统资源 ===
    cpu_percent: float = 0.0                  # CPU 使用率 (%)
    memory_used_gb: float = 0.0               # 内存使用量 (GB)
    memory_percent: float = 0.0               # 内存使用比例 (%)
    
    # === 检测结果统计 ===
    anomaly_count: int = 0                    # 检测到的异常点数
    anomaly_ratio: float = 0.0                # 异常比例 (%)
    global_anomaly_count: int = 0             # 全局异常点数
    local_anomaly_count: int = 0              # 局部异常点数
    
    # === 数据质量信息 ===
    is_step_data: bool = False                # 是否为阶跃数据
    is_noisy_data: bool = False               # 是否为噪声数据
    data_type: List[str] = field(default_factory=list)  # 数据类型标签
    
    # === 其他信息 ===
    status: str = "success"                   # 处理状态 (success/failed/skipped)
    error_message: str = ""                   # 错误信息 (如果失败)
    timestamp: str = ""                       # 记录时间戳


class PerfLogger:
    """性能跟踪日志器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, log_file: str = None, 
                 json_log_file: str = None,
                 level: int = logging.INFO):
        if PerfLogger._initialized:
            return
        
        # 默认日志路径：脚本所在目录下的 logs 子目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, "logs")
        
        # 确保 logs 目录存在
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # 使用默认路径或用户指定路径
        if log_file is None:
            log_file = os.path.join(logs_dir, "perf_tracking.log")
        if json_log_file is None:
            json_log_file = os.path.join(logs_dir, "perf_tracking.json")
        
        self.log_file = log_file
        self.json_log_file = json_log_file
        
        # 配置文本日志
        self.logger = logging.getLogger("perf_logger")
        self.logger.setLevel(level)
        self.logger.handlers = []  # 清除已有 handler
        
        # 文件 handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - [PERF] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # 控制台 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # 初始化 pynvml
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                self.logger.warning(f"pynvml 初始化失败: {e}")
                self._nvml_initialized = False
        else:
            self._nvml_initialized = False
        
        # 初始化 JSON 日志文件
        if not os.path.exists(json_log_file):
            with open(json_log_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        PerfLogger._initialized = True
        self.logger.info("=" * 60)
        self.logger.info("性能跟踪日志器已初始化")
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info(f"JSON 日志: {json_log_file}")
        self.logger.info(f"GPU 监控: {'可用' if self._nvml_initialized else '不可用'}")
        self.logger.info(f"系统监控: {'可用' if PSUTIL_AVAILABLE else '不可用'}")
        self.logger.info("=" * 60)
    
    def get_gpu_metrics(self, device: str = "cuda:0") -> Dict[str, Any]:
        """获取 GPU 使用指标"""
        metrics = {
            "gpu_id": -1,
            "gpu_name": "",
            "gpu_utilization_percent": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
            "gpu_memory_percent": 0.0,
            "gpu_temperature_c": 0.0,
            "gpu_power_w": 0.0
        }
        
        if not self._nvml_initialized:
            return metrics
        
        try:
            # 解析 GPU ID
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
            elif device == "cuda":
                gpu_id = 0
            else:
                return metrics
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # GPU 名称
            metrics["gpu_id"] = gpu_id
            metrics["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
            if isinstance(metrics["gpu_name"], bytes):
                metrics["gpu_name"] = metrics["gpu_name"].decode("utf-8")
            
            # GPU 利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_utilization_percent"] = util.gpu
            
            # 显存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["gpu_memory_used_mb"] = mem_info.used / 1024 / 1024
            metrics["gpu_memory_total_mb"] = mem_info.total / 1024 / 1024
            metrics["gpu_memory_percent"] = (mem_info.used / mem_info.total) * 100
            
            # 温度
            try:
                metrics["gpu_temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
            
            # 功耗
            try:
                metrics["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            except:
                pass
                
        except Exception as e:
            self.logger.debug(f"获取 GPU 指标失败: {e}")
        
        return metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统资源指标"""
        metrics = {
            "cpu_percent": 0.0,
            "memory_used_gb": 0.0,
            "memory_percent": 0.0
        }
        
        if not PSUTIL_AVAILABLE:
            return metrics
        
        try:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            metrics["memory_used_gb"] = mem.used / 1024 / 1024 / 1024
            metrics["memory_percent"] = mem.percent
        except Exception as e:
            self.logger.debug(f"获取系统指标失败: {e}")
        
        return metrics
    
    def log_metrics(self, metrics: PerfMetrics):
        """记录完整的性能指标"""
        metrics.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 文本日志
        self.logger.info("=" * 80)
        self.logger.info(f"【点位处理完成】{metrics.sensor_column}")
        self.logger.info("-" * 40)
        
        # 点位信息
        self.logger.info(f"[点位信息]")
        self.logger.info(f"  路径: {metrics.sensor_path}")
        self.logger.info(f"  测点: {metrics.sensor_column}")
        self.logger.info(f"  时间范围: {metrics.start_time} ~ {metrics.end_time}")
        self.logger.info(f"  进程 ID: {metrics.process_id}")
        
        # 方法信息
        self.logger.info(f"[方法配置]")
        self.logger.info(f"  检测方法: {metrics.method}")
        self.logger.info(f"  降采样方法: {metrics.downsampler}")
        self.logger.info(f"  设备: {metrics.device}")
        
        # 数据量
        self.logger.info(f"[数据量统计]")
        self.logger.info(f"  原始数据量: {metrics.raw_data_length:,}")
        self.logger.info(f"  降采样后: {metrics.downsampled_data_length:,}")
        self.logger.info(f"  降采样比例: {metrics.downsample_ratio:.4f}")
        self.logger.info(f"  采样率: {metrics.sampling_rate_hz:.4f} Hz")
        
        # 耗时统计
        self.logger.info(f"[耗时统计] (单位: 秒)")
        self.logger.info(f"  总耗时: {metrics.total_time:.4f}")
        self.logger.info(f"  ├─ 数据读取: {metrics.data_read_time:.4f}")
        self.logger.info(f"  ├─ 预处理: {metrics.preprocess_time:.4f}")
        self.logger.info(f"  ├─ 降采样: {metrics.downsample_time:.4f}")
        if metrics.stl_decompose_time > 0:
            self.logger.info(f"  ├─ STL分解: {metrics.stl_decompose_time:.4f}")
        if metrics.model_inference_time > 0:
            self.logger.info(f"  ├─ 模型推理: {metrics.model_inference_time:.4f}")
        self.logger.info(f"  ├─ 异常检测: {metrics.anomaly_detect_time:.4f}")
        self.logger.info(f"  ├─ 后处理: {metrics.postprocess_time:.4f}")
        self.logger.info(f"  └─ 结果保存: {metrics.save_time:.4f}")
        
        # GPU 资源 (如果使用)
        if metrics.gpu_id >= 0:
            self.logger.info(f"[GPU 资源] GPU:{metrics.gpu_id} ({metrics.gpu_name})")
            self.logger.info(f"  利用率: {metrics.gpu_utilization_percent:.1f}%")
            self.logger.info(f"  显存: {metrics.gpu_memory_used_mb:.0f}/{metrics.gpu_memory_total_mb:.0f} MB ({metrics.gpu_memory_percent:.1f}%)")
            self.logger.info(f"  温度: {metrics.gpu_temperature_c:.0f}°C")
            self.logger.info(f"  功耗: {metrics.gpu_power_w:.1f}W")
        
        # 系统资源
        self.logger.info(f"[系统资源]")
        self.logger.info(f"  CPU 使用率: {metrics.cpu_percent:.1f}%")
        self.logger.info(f"  内存: {metrics.memory_used_gb:.2f} GB ({metrics.memory_percent:.1f}%)")
        
        # 检测结果
        self.logger.info(f"[检测结果]")
        self.logger.info(f"  异常点数: {metrics.anomaly_count:,}")
        self.logger.info(f"  异常比例: {metrics.anomaly_ratio:.4f}%")
        self.logger.info(f"  全局异常: {metrics.global_anomaly_count:,}")
        self.logger.info(f"  局部异常: {metrics.local_anomaly_count:,}")
        
        # 数据质量
        if metrics.data_type:
            self.logger.info(f"[数据质量]")
            self.logger.info(f"  数据类型: {', '.join(metrics.data_type) if metrics.data_type else '正常'}")
        
        # 状态
        self.logger.info(f"[状态] {metrics.status}")
        if metrics.error_message:
            self.logger.info(f"  错误信息: {metrics.error_message}")
        
        self.logger.info("=" * 80)
        
        # JSON 日志 (追加)
        self._append_json_log(metrics)
    
    def _append_json_log(self, metrics: PerfMetrics):
        """追加 JSON 日志"""
        try:
            with open(self.json_log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            data = []
        
        # 处理 data_type 字段，确保它是列表
        metrics_dict = asdict(metrics)
        if isinstance(metrics_dict.get('data_type'), list):
            pass  # 已经是列表
        elif metrics_dict.get('data_type') is None:
            metrics_dict['data_type'] = []
        
        data.append(metrics_dict)
        
        with open(self.json_log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def log_summary(self, method: str, total_sensors: int, 
                    total_time: float, success_count: int, 
                    failed_count: int, skipped_count: int):
        """记录总体运行摘要"""
        self.logger.info("=" * 80)
        self.logger.info("【运行摘要】")
        self.logger.info(f"  检测方法: {method}")
        self.logger.info(f"  总点位数: {total_sensors}")
        self.logger.info(f"  成功: {success_count}")
        self.logger.info(f"  失败: {failed_count}")
        self.logger.info(f"  跳过: {skipped_count}")
        self.logger.info(f"  总耗时: {total_time:.2f} 秒")
        if total_sensors > 0:
            self.logger.info(f"  平均每点位: {total_time / total_sensors:.2f} 秒")
        self.logger.info("=" * 80)


class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def get_perf_logger(log_file: str = None,
                    json_log_file: str = None) -> PerfLogger:
    """获取性能日志器单例
    
    Args:
        log_file: 日志文件路径，默认为脚本目录下的 logs/perf_tracking.log
        json_log_file: JSON 日志文件路径，默认为脚本目录下的 logs/perf_tracking.json
    """
    return PerfLogger(log_file=log_file, json_log_file=json_log_file)


# 便捷装饰器，用于跟踪函数执行时间
def track_time(metric_name: str):
    """装饰器，用于跟踪函数执行时间并返回"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            # 返回结果和耗时
            return result, elapsed
        return wrapper
    return decorator
