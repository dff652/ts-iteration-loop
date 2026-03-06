"""
任务与调度相关模型
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean
from sqlalchemy.sql import func

from configs.settings import settings
from src.utils.time_utils import utc_now_naive
from src.db.models.base import Base


class Task(Base):
    """异步任务表"""
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True)
    type = Column(String(50), nullable=False)  # acquire / training / inference
    status = Column(String(20), default="pending")
    config = Column(Text)  # JSON 配置
    result = Column(Text)  # JSON 结果
    error = Column(Text)
    created_at = Column(DateTime, default=utc_now_naive)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


class TaskCenterDefinition(Base):
    """任务中心：任务定义"""
    __tablename__ = "task_definitions"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    task_type = Column(String(50), nullable=False, default="acquire_inference")
    trigger_mode = Column(String(20), nullable=False, default="manual")  # manual / auto / schedule
    schedule_cron = Column(String(100))
    enabled = Column(Boolean, default=True)
    config = Column(Text)  # JSON
    created_by = Column(String(100), default=settings.DEFAULT_USER)
    updated_by = Column(String(100), default=settings.DEFAULT_USER)
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=func.now())


class TaskCenterRun(Base):
    """任务中心：任务执行实例"""
    __tablename__ = "task_runs"

    id = Column(String(36), primary_key=True)
    definition_id = Column(String(36))
    task_type = Column(String(50), nullable=False, default="acquire_inference")
    trigger_mode = Column(String(20), nullable=False, default="manual")
    status = Column(String(20), default="pending")
    input_payload = Column(Text)  # JSON
    result = Column(Text)  # JSON
    error = Column(Text)
    created_at = Column(DateTime, default=utc_now_naive)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


class TaskCenterStepRun(Base):
    """任务中心：步骤执行实例"""
    __tablename__ = "task_step_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), nullable=False)
    step_name = Column(String(50), nullable=False)  # acquire / inference
    status = Column(String(20), default="pending")
    depends_on = Column(Text)  # JSON list[str]，依赖的 step_name 集合
    message = Column(Text)
    logs = Column(Text)
    result = Column(Text)  # JSON
    created_at = Column(DateTime, default=utc_now_naive)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=func.now())


class TaskCenterResultIndex(Base):
    """任务中心：结果索引"""
    __tablename__ = "task_run_result_index"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), nullable=False)
    point_id = Column(String(200))
    model_version = Column(String(200))
    result_path = Column(String(500))
    status = Column(String(20), default="completed")
    meta = Column(Text)  # JSON
    created_at = Column(DateTime, default=utc_now_naive)
