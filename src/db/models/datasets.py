"""
数据资产与迭代版本相关模型
"""
from sqlalchemy import Column, String, Text, DateTime, Integer
from sqlalchemy.sql import func

from configs.settings import settings
from src.utils.time_utils import utc_now_naive
from src.db.models.base import Base


class IterationVersion(Base):
    """迭代版本表"""
    __tablename__ = "iteration_versions"
    
    id = Column(String(36), primary_key=True)
    version = Column(String(50), nullable=False)
    description = Column(Text)
    dataset_path = Column(String(500))
    annotation_count = Column(Integer, default=0)
    model_path = Column(String(500))
    metrics = Column(Text)  # JSON 指标
    created_at = Column(DateTime, default=utc_now_naive)


class Dataset(Base):
    """数据集表"""
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    path = Column(String(500))
    point_count = Column(Integer)
    source = Column(String(500))  # 来源 (IoTDB路径等)
    version_id = Column(String(36))  # 关联迭代版本
    created_at = Column(DateTime, default=utc_now_naive)


class DatasetAsset(Base):
    """数据资产集（点位集合）"""
    __tablename__ = "dataset_assets"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    owner_id = Column(String(100), default=settings.DEFAULT_USER)
    org_id = Column(String(100), default=settings.DEFAULT_ORG)
    created_by = Column(String(100), default=settings.DEFAULT_USER)
    updated_by = Column(String(100), default=settings.DEFAULT_USER)
    dataset_type = Column(String(20), nullable=False)  # train / golden
    status = Column(String(20), default="draft")  # draft / frozen
    point_count = Column(Integer, default=0)
    meta = Column(Text)  # JSON
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=func.now())


class DatasetItem(Base):
    """数据资产集条目（点位）"""
    __tablename__ = "dataset_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(36), nullable=False)
    point_id = Column(String(200))
    point_name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=utc_now_naive)
