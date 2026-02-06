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
    APP_VERSION: str = "0.3.5"
    DEBUG: bool = True
    
    # 服务端口
    # 服务端口
    API_PORT: int = 8000
    WEBUI_HOST: str = "192.168.199.126"
    
    # 用户配置
    DEFAULT_USER: str = "douff"
    
    # 数据库
    DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT}/data/iteration_loop.db"
    
    # ========== 标准化数据目录 ==========
    DATA_ROOT: str = "/home/share/data"
    DATA_RAW_DIR: str = "/home/share/data/raw"
    DATA_DOWNSAMPLED_DIR: str = "/home/share/data/downsampled"
    DATA_IMAGES_DIR: str = "/home/share/data/images"
    DATA_INFERENCE_DIR: str = "/home/share/data/inference"
    # 训练数据目录（按模型类型拆分）
    DATA_TRAINING_CHATTS_DIR: str = str(PROJECT_ROOT / "services" / "training" / "data" / "chatts")
    DATA_TRAINING_QWEN_DIR: str = str(PROJECT_ROOT / "services" / "training" / "data" / "qwen")
    DATA_TRAINING_QWEN_IMAGES_DIR: str = DATA_IMAGES_DIR  # 统一使用全局图片池
    # 兼容旧配置（默认指向 ChatTS）
    DATA_TRAINING_DIR: str = DATA_TRAINING_CHATTS_DIR
    ANNOTATIONS_ROOT: str = "/home/share/data/annotations"
    # 评估 (黄金集) 配置
    EVAL_GOLDEN_TRUTH_DIR: str = "/home/share/data/annotations/douff"
    EVAL_GOLDEN_DATA_DIR: str = "/home/share/data/downsampled"
    EVAL_DEFAULT_DATASET_NAME: str = "golden"
    EVAL_DEFAULT_OUTPUT_DIR: str = ""
    
    # ========== 模块路径配置 ==========
    # 使用本地整合模块（Monorepo 模式）
    USE_LOCAL_MODULES: bool = True
    
    # 本地模块路径（services/ 目录）
    LOCAL_CHECK_OUTLIER_PATH: str = str(PROJECT_ROOT / "services" / "inference")
    LOCAL_TRAINING_PATH: str = str(PROJECT_ROOT / "services" / "training")
    LOCAL_DATA_PROCESSING_PATH: str = str(PROJECT_ROOT / "services" / "data_processing")
    LOCAL_ANNOTATOR_PATH: str = str(PROJECT_ROOT / "services" / "annotator")
    
    # 外部模块路径（已禁用：代码已整合，避免修改外部独立项目）
    # EXTERNAL_DATA_PROCESSING_PATH: str = "/home/douff/ts/Data-Processing"
    # EXTERNAL_ANNOTATOR_PATH: str = "/home/douff/ts/timeseries-annotator-v2"
    # EXTERNAL_CHATTS_TRAINING_PATH: str = "/home/douff/ts/ChatTS-Training"
    # EXTERNAL_CHECK_OUTLIER_PATH: str = str(PROJECT_ROOT / "services" / "inference")
    
    @property
    def DATA_PROCESSING_PATH(self) -> str:
        return self.LOCAL_DATA_PROCESSING_PATH
    
    @property
    def ANNOTATOR_PATH(self) -> str:
        return self.LOCAL_ANNOTATOR_PATH
    
    @property
    def CHATTS_TRAINING_PATH(self) -> str:
        return self.LOCAL_TRAINING_PATH
    
    @property
    def CHECK_OUTLIER_PATH(self) -> str:
        return self.LOCAL_CHECK_OUTLIER_PATH
    
    # ========== Python 解释器配置 ==========
    # 环境模式: unified (默认) | legacy (使用独立环境)
    ENV_MODE: str = os.getenv("ENV_MODE", "unified").lower()
    
    # 统一环境 Python 解释器（优先使用环境变量或当前解释器）
    PYTHON_UNIFIED: str = os.getenv("PYTHON_UNIFIED", sys.executable)
    
    # 动态计算默认路径 (内部辅助变量)
    _is_unified = os.getenv("ENV_MODE", "unified").lower() == "unified"
    _default_training = sys.executable if _is_unified else "/opt/miniconda3/envs/chatts_tune/bin/python"
    _default_annotator = sys.executable if _is_unified else "/opt/miniconda3/envs/test-env/bin/python"
    _default_processing = sys.executable if _is_unified else "/opt/conda_envs/douff/ts_iter_loop/bin/python"
    
    # 各模块环境配置
    PYTHON_ILABEL: str = "/opt/miniconda3/envs/chatts_test/bin/python"
    PYTHON_TRAINING: str = os.getenv("PYTHON_TRAINING", _default_training)
    PYTHON_TRAINING_CHATTS: str = os.getenv("PYTHON_TRAINING_CHATTS", PYTHON_TRAINING)
    PYTHON_TRAINING_QWEN: str = os.getenv("PYTHON_TRAINING_QWEN", PYTHON_TRAINING)
    PYTHON_ANNOTATOR: str = os.getenv("PYTHON_ANNOTATOR", _default_annotator)
    PYTHON_DATA_PROCESSING: str = os.getenv("PYTHON_DATA_PROCESSING", _default_processing)
    
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
