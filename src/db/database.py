"""
数据库模型和连接（核心入口与向后兼容导出层）

所有 ORM 模型已拆分至 src/db/models/ 子模块中。
此类仅保留核心引擎及为了向旧业务代码兼容的无缝导入暴露。
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from configs.settings import settings

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False}  # SQLite 需要
)

# 创建会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==================== 数据库模型向后兼容层 ====================
# 这里将全部模型通过 import 重新导出，
# 这保证了 src.db.database.Task, AnnotationRecord 等能够继续在所有原代码中被访问
from src.db.models import (
    Base,
    Task,
    TaskCenterDefinition,
    TaskCenterRun,
    TaskCenterStepRun,
    TaskCenterResultIndex,
    IterationVersion,
    Dataset,
    DatasetAsset,
    DatasetItem,
    AnnotationRecord,
    AnnotationSegment,
    InferenceResult,
    SegmentScore,
    MetricRecord,
    ReviewQueue,
    ModelEval,
    ModelRegistry,
    IotdbSource,
    User,
)

# 为了让 wildcard import `from src.db.database import *` 不出问题（如果有的话）
__all__ = [
    "engine",
    "SessionLocal",
    "Base",
    "init_db",
    "get_db",
    "Task",
    "TaskCenterDefinition",
    "TaskCenterRun",
    "TaskCenterStepRun",
    "TaskCenterResultIndex",
    "IterationVersion",
    "Dataset",
    "DatasetAsset",
    "DatasetItem",
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


# ==================== 核心功能函数 ====================

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
