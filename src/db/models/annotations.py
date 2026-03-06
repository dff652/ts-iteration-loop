"""
标注相关模型
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, Float
from sqlalchemy.sql import func

from src.utils.time_utils import utc_now_naive
from src.db.models.base import Base


class AnnotationRecord(Base):
    """标注统一实体（DB-First 在线真相源）"""
    __tablename__ = "annotation_records"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(100), nullable=False)
    point_id = Column(String(200))
    source_id = Column(String(200), nullable=False)  # 规范化点位名
    filename = Column(String(500), nullable=False)
    source_kind = Column(String(20), default="human")  # auto / human
    source_inference_id = Column(String(36))
    method = Column(String(50))
    status = Column(String(20), default="draft")
    is_human_edited = Column(Boolean, default=True)
    annotation_count = Column(Integer, default=0)
    segment_count = Column(Integer, default=0)
    overall_attribute_json = Column(Text)  # JSON
    annotations_json = Column(Text)  # JSON
    meta = Column(Text)  # JSON
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=func.now())


class AnnotationSegment(Base):
    """段级统一实体（标注段）"""
    __tablename__ = "annotation_segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    annotation_id = Column(String(36), nullable=False)
    user_id = Column(String(100), nullable=False)
    point_id = Column(String(200))
    source_id = Column(String(200), nullable=False)
    ann_index = Column(Integer, default=0)
    seg_index = Column(Integer, default=0)
    start = Column(Integer)
    end = Column(Integer)
    count = Column(Integer)
    label_id = Column(String(200))
    label_text = Column(String(200))
    score = Column(Float)
    review_status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=func.now())
