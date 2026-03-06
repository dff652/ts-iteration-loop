"""
模型注册与评估相关模型
"""
from sqlalchemy import Column, String, Text, DateTime, Float
from sqlalchemy.sql import func

from configs.settings import settings
from src.utils.time_utils import utc_now_naive
from src.db.models.base import Base


class ModelEval(Base):
    """训练后黄金集评估结果"""
    __tablename__ = "model_evals"

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36))
    model_family = Column(String(50), default="chatts")
    model_path = Column(String(500))
    dataset_id = Column(String(36))
    dataset_name = Column(String(200))
    metrics = Column(Text)  # JSON
    created_at = Column(DateTime, default=utc_now_naive)


class ModelRegistry(Base):
    """模型注册表（模型库）"""
    __tablename__ = "model_registry"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    model_family = Column(String(50), default="chatts")    # chatts / qwen
    model_type = Column(String(20), default="lora")        # lora / full / base
    version = Column(String(50))                           # v1.0 / v20260304
    model_path = Column(String(500), nullable=False)       # 模型文件路径
    base_model = Column(String(500))                       # 基础模型路径
    config = Column(Text)          # JSON: 训练参数快照
    metrics = Column(Text)         # JSON: 评估指标
    train_loss = Column(Float)     # 最终训练 loss
    status = Column(String(20), default="active")          # active / archived / deprecated
    tags = Column(Text)            # JSON list: 自定义标签
    description = Column(Text)     # 备注
    source_task_id = Column(String(36))  # 关联训练任务 run_id
    created_by = Column(String(100), default=settings.DEFAULT_USER)
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=func.now())
