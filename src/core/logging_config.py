"""
TS-Iteration-Loop 统一日志配置

按模块分 logger，支持控制台 + 文件双输出。

用法:
    from src.core.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("启动成功")
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from configs.settings import settings, PROJECT_ROOT

# ==================== 配置常量 ====================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if settings.DEBUG else "INFO").upper()
LOG_FILE = LOG_DIR / "app.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# 日志格式
CONSOLE_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)-28s │ %(message)s"
FILE_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ==================== 初始化 ====================
_initialized = False


def setup_logging() -> None:
    """初始化全局日志配置。应在应用启动时调用一次。"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # 文件 Handler（RotatingFileHandler 自动滚动）
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # 文件始终记录 DEBUG
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(file_handler)

    # 降低第三方库的日志级别
    for noisy_logger in (
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "httpx",
        "httpcore",
        "sqlalchemy.engine",
        "gradio",
        "celery",
        "urllib3",
        "watchfiles",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # 首条日志
    logger = logging.getLogger("ts.core")
    logger.info(
        "日志系统初始化完成 | level=%s | file=%s",
        LOG_LEVEL,
        LOG_FILE,
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取按模块命名的 logger。

    建议用法:
        logger = get_logger(__name__)

    自动将模块名映射到 ts.* 命名空间:
        src.api.inference  → ts.api.inference
        src.adapters.xxx   → ts.adapter.xxx
        services.annotator → ts.annotator
    """
    # 自动映射为更短的命名空间
    short = name
    if name.startswith("src.api"):
        short = name.replace("src.api", "ts.api", 1)
    elif name.startswith("src.adapters"):
        short = name.replace("src.adapters", "ts.adapter", 1)
    elif name.startswith("src.core"):
        short = name.replace("src.core", "ts.core", 1)
    elif name.startswith("src.webui"):
        short = name.replace("src.webui", "ts.webui", 1)
    elif name.startswith("src.db"):
        short = name.replace("src.db", "ts.db", 1)
    elif name.startswith("src.utils"):
        short = name.replace("src.utils", "ts.utils", 1)
    elif name.startswith("services.annotator"):
        short = name.replace("services.annotator", "ts.annotator", 1)
    elif name.startswith("services.inference"):
        short = name.replace("services.inference", "ts.inference", 1)
    elif name.startswith("configs"):
        short = name.replace("configs", "ts.config", 1)
    elif not name.startswith("ts."):
        short = f"ts.{name}"

    return logging.getLogger(short)
