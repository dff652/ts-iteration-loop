"""
数据模型定义 (Pydantic schemas)
"""
from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class TaskStatus(str, Enum):
    """任务状态"""
    BLOCKED = "blocked"
    RUNNABLE = "runnable"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TriggerMode(str, Enum):
    """任务触发方式"""
    MANUAL = "manual"
    AUTO = "auto"
    SCHEDULE = "schedule"
    EVENT = "event"


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
    user: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    point_name: str = "*"
    target_points: int = 5000  # 降采样目标点数
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    @field_validator("user", "password")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("字段不能为空")
        return normalized


class IotdbSourceCreate(BaseModel):
    """IoTDB 数据源：创建请求"""
    name: str = Field(..., min_length=1, max_length=200)
    host: str = "192.168.199.185"
    port: str = "6667"
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)  # root.xxx.yyy
    point_name: str = "*"
    target_points: int = 5000
    description: Optional[str] = None


class IotdbSourceUpdate(BaseModel):
    """IoTDB 数据源：更新请求"""
    name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    source_path: Optional[str] = None
    point_name: Optional[str] = None
    target_points: Optional[int] = None
    description: Optional[str] = None


class IotdbSourceAcquireRequest(BaseModel):
    """IoTDB 数据源：采集任务请求"""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    target_points: Optional[int] = None  # 可覆盖数据源默认值


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
    auto_eval: bool = False
    eval_truth_dir: Optional[str] = None
    eval_data_dir: Optional[str] = None
    eval_dataset_name: Optional[str] = None
    eval_output_dir: Optional[str] = None
    eval_device: Optional[str] = None
    eval_method: Optional[str] = None
    params: Optional[dict] = None


class TrainingEvalRequest(BaseModel):
    """训练模型评估请求"""
    model_path: str
    model_family: str = "chatts"
    truth_dir: str
    data_dir: str
    dataset_name: str = "golden"
    output_dir: Optional[str] = None
    device: Optional[str] = None
    method: Optional[str] = None


# ==================== 推理服务 ====================

class InferenceTaskRequest(BaseModel):
    """推理任务请求"""
    model: str  # 模型名称或路径
    algorithm: str = "chatts"  # chatts / adtk_hbos
    input_files: List[str]
    params: Optional[dict] = None


class InferenceResult(BaseModel):
    """推理结果"""
    filename: str
    anomalies: List[dict]
    confidence: Optional[float] = None


# ==================== 任务中心 ====================

class TaskCenterDefinitionRequest(BaseModel):
    """任务中心：任务定义创建请求"""
    name: str = Field(..., min_length=1)
    task_type: str = "acquire_inference"
    trigger_mode: TriggerMode = TriggerMode.MANUAL
    schedule_cron: Optional[str] = None
    enabled: bool = True
    config: Optional[dict[str, Any]] = None
    created_by: Optional[str] = None


class TaskCenterRunCreateRequest(BaseModel):
    """任务中心：运行实例创建请求"""
    definition_id: Optional[str] = None
    task_type: str = "acquire_inference"
    trigger_mode: TriggerMode = TriggerMode.MANUAL
    input_payload: Optional[dict[str, Any]] = None
    steps: List[str] = Field(default_factory=lambda: ["acquire", "inference"])
    step_specs: Optional[List[dict[str, Any]]] = None
    max_retries: Optional[int] = Field(default=None, ge=0, le=10)
    retry_policy: Optional[str] = Field(default=None)
    retry_delay_sec: Optional[int] = Field(default=None, ge=0, le=3600)
    retry_backoff_factor: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    retry_max_delay_sec: Optional[int] = Field(default=None, ge=1, le=86400)
    retry_on_errors: Optional[List[str]] = None
    timeout_sec: Optional[int] = Field(default=None, ge=1, le=86400)
    auto_execute: bool = False

    @field_validator("retry_policy")
    @classmethod
    def _validate_retry_policy(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"fixed", "exponential"}:
            raise ValueError("retry_policy 仅支持 fixed/exponential")
        return normalized

    @field_validator("retry_on_errors")
    @classmethod
    def _validate_retry_on_errors(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        items: List[str] = []
        for raw in value:
            text = str(raw or "").strip()
            if text and text not in items:
                items.append(text)
        return items or None


class TaskCenterRunExecuteRequest(BaseModel):
    """任务中心：执行请求"""
    simulate: bool = True


class TaskCenterEventTriggerRequest(BaseModel):
    """任务中心：事件触发请求"""
    event_key: str = Field(..., min_length=1)
    payload: Optional[dict[str, Any]] = None
    definition_id: Optional[str] = None
    dedupe_key: Optional[str] = None
    execute_mode: str = Field(default="dispatch")

    @field_validator("event_key")
    @classmethod
    def _validate_event_key(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not normalized:
            raise ValueError("event_key 不能为空")
        return normalized

    @field_validator("execute_mode")
    @classmethod
    def _validate_execute_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"dispatch", "simulate", "none"}:
            raise ValueError("execute_mode 仅支持 dispatch/simulate/none")
        return normalized


class TaskCenterDeadLetterReplayRequest(BaseModel):
    """任务中心：死信重放请求"""
    event_id: Optional[str] = None
    execute_mode: str = Field(default="dispatch")

    @field_validator("execute_mode")
    @classmethod
    def _validate_execute_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"dispatch", "simulate", "none"}:
            raise ValueError("execute_mode 仅支持 dispatch/simulate/none")
        return normalized


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


class ModelEvalInfo(BaseModel):
    """模型评估结果"""
    id: str
    task_id: Optional[str] = None
    model_family: str
    model_path: str
    dataset_id: Optional[str] = None
    dataset_name: str
    metrics: dict
    created_at: datetime


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
