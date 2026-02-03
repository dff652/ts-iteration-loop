"""
数据库模型和连接
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from configs.settings import settings

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False}  # SQLite 需要
)

# 创建会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类
Base = declarative_base()


# ==================== 数据库模型 ====================

class Task(Base):
    """异步任务表"""
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True)
    type = Column(String(50), nullable=False)  # acquire / training / inference
    status = Column(String(20), default="pending")
    config = Column(Text)  # JSON 配置
    result = Column(Text)  # JSON 结果
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


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
    created_at = Column(DateTime, default=datetime.utcnow)


class Dataset(Base):
    """数据集表"""
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    path = Column(String(500))
    point_count = Column(Integer)
    source = Column(String(500))  # 来源 (IoTDB路径等)
    version_id = Column(String(36))  # 关联迭代版本
    created_at = Column(DateTime, default=datetime.utcnow)


class InferenceResult(Base):
    """推理结果索引（含置信度摘要）"""
    __tablename__ = "inference_results"

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36))
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
    created_at = Column(DateTime, default=datetime.utcnow)


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
    created_at = Column(DateTime, default=datetime.utcnow)


class MetricRecord(Base):
    """通用指标记录（多指标扩展）"""
    __tablename__ = "metrics"

    id = Column(String(36), primary_key=True)
    owner_type = Column(String(50))  # inference / model / dataset
    owner_id = Column(String(36))
    name = Column(String(100))
    data = Column(Text)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)


class ReviewQueue(Base):
    """人工审核队列"""
    __tablename__ = "review_queue"

    id = Column(String(36), primary_key=True)
    source_type = Column(String(50))  # inference / annotation
    source_id = Column(String(36))
    method = Column(String(50))
    model = Column(String(200))
    point_name = Column(String(200))
    score = Column(Float)
    strategy = Column(String(50))  # topk / low_score / random
    status = Column(String(20), default="pending")
    reviewer = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=func.now())


class ModelEval(Base):
    """训练后黄金集评估结果"""
    __tablename__ = "model_evals"

    id = Column(String(36), primary_key=True)
    model_path = Column(String(500))
    dataset_name = Column(String(200))
    metrics = Column(Text)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)


# 创建所有表
def init_db():
    """初始化数据库"""
    Base.metadata.create_all(bind=engine)


# 获取数据库会话
def get_db():
    """获取数据库会话 (FastAPI 依赖注入)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
