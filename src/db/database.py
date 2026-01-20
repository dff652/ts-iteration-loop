"""
数据库模型和连接
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Enum
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
