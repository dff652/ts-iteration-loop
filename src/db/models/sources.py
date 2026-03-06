"""
IoTDB 数据源配置模型
"""
from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlalchemy.sql import func

from src.utils.time_utils import utc_now_naive
from src.db.models.base import Base


class IotdbSource(Base):
    """IoTDB 数据源配置"""
    __tablename__ = "iotdb_sources"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    host = Column(String(200), default="192.168.199.185")
    port = Column(String(10), default="6667")
    username = Column(String(100), nullable=False)
    password = Column(String(200), nullable=False)
    source_path = Column(String(500), nullable=False)       # root.xxx.yyy
    point_name = Column(String(200), default="*")
    target_points = Column(Integer, default=5000)
    description = Column(Text)
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=func.now())
