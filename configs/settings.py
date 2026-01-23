# TS-Iteration-Loop 项目配置

import os
import sys
from pathlib import Path
from pydantic_settings import BaseSettings

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """应用配置"""
    
    # 应用信息
    APP_NAME: str = "TS-Iteration-Loop"
    APP_VERSION: str = "0.2.0"  # 版本升级：Monorepo 整合
    DEBUG: bool = True
    
    # 服务端口
    API_PORT: int = 8000
    
    # 数据库
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT}/data/iteration_loop.db"
    
    # ========== 模块路径配置 ==========
    # 使用本地整合模块（Monorepo 模式）
    USE_LOCAL_MODULES: bool = True
    
    # 本地模块路径（services/ 目录）
    LOCAL_CHECK_OUTLIER_PATH: str = str(PROJECT_ROOT / "services" / "inference")
    LOCAL_TRAINING_PATH: str = str(PROJECT_ROOT / "services" / "training")
    LOCAL_DATA_PROCESSING_PATH: str = str(PROJECT_ROOT / "services" / "data_processing")
    LOCAL_ANNOTATOR_PATH: str = str(PROJECT_ROOT / "services" / "annotator")
    
    # 外部模块路径（兼容旧配置，USE_LOCAL_MODULES=False 时使用）
    EXTERNAL_DATA_PROCESSING_PATH: str = "/home/douff/ts/Data-Processing"
    EXTERNAL_ANNOTATOR_PATH: str = "/home/douff/ts/timeseries-annotator-v2"
    EXTERNAL_CHATTS_TRAINING_PATH: str = "/home/douff/ts/ChatTS-Training"
    EXTERNAL_CHECK_OUTLIER_PATH: str = "/home/douff/ilabel/check_outlier"
    
    @property
    def DATA_PROCESSING_PATH(self) -> str:
        return self.LOCAL_DATA_PROCESSING_PATH if self.USE_LOCAL_MODULES else self.EXTERNAL_DATA_PROCESSING_PATH
    
    @property
    def ANNOTATOR_PATH(self) -> str:
        return self.LOCAL_ANNOTATOR_PATH if self.USE_LOCAL_MODULES else self.EXTERNAL_ANNOTATOR_PATH
    
    @property
    def CHATTS_TRAINING_PATH(self) -> str:
        return self.LOCAL_TRAINING_PATH if self.USE_LOCAL_MODULES else self.EXTERNAL_CHATTS_TRAINING_PATH
    
    @property
    def CHECK_OUTLIER_PATH(self) -> str:
        return self.LOCAL_CHECK_OUTLIER_PATH if self.USE_LOCAL_MODULES else self.EXTERNAL_CHECK_OUTLIER_PATH
    
    # ========== Python 解释器配置 ==========
    # 统一环境 Python 解释器（优先使用环境变量或当前解释器）
    PYTHON_UNIFIED: str = os.getenv("PYTHON_UNIFIED", sys.executable)
    
    # 各模块独立环境（兼容模式，USE_LOCAL_MODULES=False 时可能需要）
    PYTHON_ILABEL: str = "/opt/miniconda3/envs/chatts_test/bin/python"
    PYTHON_TRAINING: str = "/opt/miniconda3/envs/chatts_tune/bin/python"
    PYTHON_ANNOTATOR: str = "/opt/miniconda3/envs/test-env/bin/python"
    PYTHON_DATA_PROCESSING: str = "/opt/conda_envs/douff/ts_iter_loop/bin/python"
    
    # 标注工具配置 (复用 JWT)
    ANNOTATOR_API_URL: str = "http://localhost:5000"
    JWT_SECRET_KEY: str = "your-secret-key"  # 需与标注工具一致
    JWT_ALGORITHM: str = "HS256"
    
    # Redis (任务队列) - Temporary switch to SQLite for verification
    # REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "sqla+sqlite:///" + str(PROJECT_ROOT / "data" / "celery_broker.db")
    CELERY_RESULT_BACKEND: str = "db+sqlite:///" + str(PROJECT_ROOT / "data" / "celery_results.db")
    
    # 数据处理配置 - 统一降采样参数确保数据链路一致性
    DEFAULT_DOWNSAMPLE_POINTS: int = 5000  # 与 Data-Processing 的 target_points 保持一致
    
    # 版本管理
    VERSIONS_DIR: str = str(PROJECT_ROOT / "data" / "versions")
    
    class Config:
        env_file = ".env"

settings = Settings()

