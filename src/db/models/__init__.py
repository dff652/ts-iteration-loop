"""
暴露所有的模型以便 database.py 继续向后兼容导入
"""
from src.db.models.base import Base
from src.db.models.tasks import Task, TaskCenterDefinition, TaskCenterRun, TaskCenterStepRun, TaskCenterResultIndex
from src.db.models.datasets import Dataset, DatasetAsset, DatasetItem, IterationVersion
from src.db.models.annotations import AnnotationRecord, AnnotationSegment
from src.db.models.inference import InferenceResult, SegmentScore, MetricRecord, ReviewQueue
from src.db.models.registry import ModelEval, ModelRegistry
from src.db.models.sources import IotdbSource
from src.db.models.users import User

__all__ = [
    "Base",
    "Task",
    "TaskCenterDefinition",
    "TaskCenterRun",
    "TaskCenterStepRun",
    "TaskCenterResultIndex",
    "Dataset",
    "DatasetAsset",
    "DatasetItem",
    "IterationVersion",
    "AnnotationRecord",
    "AnnotationSegment",
    "InferenceResult",
    "SegmentScore",
    "MetricRecord",
    "ReviewQueue",
    "ModelEval",
    "ModelRegistry",
    "IotdbSource",
    "User",
]
