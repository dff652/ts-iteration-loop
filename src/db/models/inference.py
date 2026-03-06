"""
推理与审核队列相关模型
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, Float
from sqlalchemy.sql import func

from src.utils.time_utils import utc_now_naive
from src.db.models.base import Base


class InferenceResult(Base):
    """推理结果索引（含置信度摘要）"""
    __tablename__ = "inference_results"

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36))
    point_id = Column(String(200))
    method = Column(String(50))
    model = Column(String(200))
    point_name = Column(String(200))
    result_path = Column(String(500))
    metrics_path = Column(String(500))
    segments_path = Column(String(500))
    score_avg = Column(Float)
    score_max = Column(Float)
    segment_count = Column(Integer)
    meta = Column(Text)  # JSON
    created_at = Column(DateTime, default=utc_now_naive)


class SegmentScore(Base):
    """异常段级评分"""
    __tablename__ = "segment_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inference_id = Column(String(36))
    start = Column(Integer)
    end = Column(Integer)
    score = Column(Float)
    raw_p = Column(Float)
    left = Column(Float)
    right = Column(Float)
    created_at = Column(DateTime, default=utc_now_naive)


class MetricRecord(Base):
    """通用指标记录（多指标扩展）"""
    __tablename__ = "metrics"

    id = Column(String(36), primary_key=True)
    owner_type = Column(String(50))  # inference / model / dataset
    owner_id = Column(String(36))
    name = Column(String(100))
    data = Column(Text)  # JSON
    created_at = Column(DateTime, default=utc_now_naive)


class ReviewQueue(Base):
    """人工审核队列"""
    __tablename__ = "review_queue"

    id = Column(String(36), primary_key=True)
    source_type = Column(String(50))  # inference / annotation
    source_id = Column(String(36))
    point_id = Column(String(200))
    method = Column(String(50))
    model = Column(String(200))
    point_name = Column(String(200))
    score = Column(Float)
    strategy = Column(String(50))  # topk / low_score / random
    status = Column(String(20), default="pending")
    reviewer = Column(String(100))
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=func.now())
