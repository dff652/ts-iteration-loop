"""
数据模型定义 (Pydantic schemas)
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from enum import Enum


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==================== 数据服务 ====================

class DatasetInfo(BaseModel):
    """数据集信息"""
    id: str
    name: str
    path: str
    point_count: int
    created_at: datetime
    version: Optional[str] = None


class AcquireTaskRequest(BaseModel):
    """数据采集任务请求"""
    source: str  # IoTDB 路径
    host: str = "192.168.199.185"
    port: str = "6667"
    user: str = "root"
    password: str = "root"
    point_name: str = "*"
    target_points: int = 5000  # 降采样目标点数
    start_time: Optional[str] = None
    end_time: Optional[str] = None


# ==================== 标注服务 ====================

class AnnotationSegment(BaseModel):
    """标注数据段"""
    start: int
    end: int


class Annotation(BaseModel):
    """单条标注"""
    label: str
    color: str
    segments: List[AnnotationSegment]
    question: Optional[str] = None
    analysis: Optional[str] = None


class AnnotationFile(BaseModel):
    """文件标注"""
    filename: str
    annotations: List[Annotation]
    overall_attribute: Optional[dict] = None


# ==================== 微调服务 ====================

class TrainingConfig(BaseModel):
    """训练配置"""
    name: str
    base_model: str
    dataset: str
    method: str = "lora"  # lora / full
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-5
    lora_rank: int = 8


class TrainingTaskRequest(BaseModel):
    """训练任务请求"""
    config_name: str
    version_tag: Optional[str] = None
    model_family: str = "chatts"


# ==================== 推理服务 ====================

class InferenceTaskRequest(BaseModel):
    """推理任务请求"""
    model: str  # 模型名称或路径
    algorithm: str = "chatts"  # chatts / adtk_hbos
    input_files: List[str]


class InferenceResult(BaseModel):
    """推理结果"""
    filename: str
    anomalies: List[dict]
    confidence: Optional[float] = None


# ==================== 版本管理 ====================

class IterationVersion(BaseModel):
    """迭代版本"""
    id: str
    version: str
    created_at: datetime
    dataset_version: str
    annotation_count: int
    model_path: Optional[str] = None
    metrics: Optional[dict] = None


# ==================== 通用 ====================

class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: TaskStatus
    message: str


class ApiResponse(BaseModel):
    """API 通用响应"""
    success: bool
    data: Optional[dict] = None
    message: str = ""
