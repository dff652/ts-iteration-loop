# TS-Iteration-Loop 项目配置

from pathlib import Path
from pydantic_settings import BaseSettings

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """应用配置"""
    
    # 应用信息
    APP_NAME: str = "TS-Iteration-Loop"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # 服务端口
    API_PORT: int = 8000
    
    # 数据库
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT}/data/iteration_loop.db"
    
    # 外部项目路径
    DATA_PROCESSING_PATH: str = "/home/douff/ts/Data-Processing"
    ANNOTATOR_PATH: str = "/home/douff/ts/timeseries-annotator-v2"
    CHATTS_TRAINING_PATH: str = "/home/douff/ts/ChatTS-Training"
    CHECK_OUTLIER_PATH: str = "/home/douff/ilabel/check_outlier"
    
    # 标注工具配置 (复用 JWT)
    ANNOTATOR_API_URL: str = "http://localhost:5000"
    JWT_SECRET_KEY: str = "your-secret-key"  # 需与标注工具一致
    JWT_ALGORITHM: str = "HS256"
    
    # Redis (任务队列)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # 版本管理
    VERSIONS_DIR: str = str(PROJECT_ROOT / "data" / "versions")
    
    class Config:
        env_file = ".env"

settings = Settings()
