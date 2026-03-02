"""
TS-Iteration-Loop 主入口
"""
import os
import getpass
import subprocess
import sys
import atexit
import signal
import time

# 设置 Gradio 临时目录，避免与其他用户冲突
gradio_temp_dir = f"/tmp/{getpass.getuser()}/gradio"
os.makedirs(gradio_temp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = gradio_temp_dir

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import gradio as gr

from configs.settings import settings
from src.core.logging_config import setup_logging, get_logger
from src.db.database import init_db
from src.db.migration import apply_pending_migrations

# 初始化日志（必须在所有业务模块导入前调用）
setup_logging()
logger = get_logger(__name__)

# 导入 API 路由
from src.api import data, annotation, training, inference, assets, points, evaluation

# 导入 Gradio 界面
from src.webui.training_ui import training_ui

# 创建 FastAPI 应用
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    settings.validate_security_settings()
    init_db()
    applied = apply_pending_migrations()
    logger.info("=" * 60)
    logger.info("🚀 %s 启动成功", settings.APP_NAME)
    logger.info("📖 API 文档: http://localhost:%s/docs", settings.API_PORT)
    logger.info("🎯 微调界面: http://localhost:%s/train-ui", settings.API_PORT)
    if applied:
        logger.info("🗄️ 已应用迁移: %s", ", ".join(applied))
    logger.info("=" * 60)
    yield
    # Shutdown logic if needed

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="时序异常检测迭代循环系统 API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(data.router, prefix="/api/v1/data", tags=["数据服务"])
app.include_router(annotation.router, prefix="/api/v1/annotation", tags=["标注服务"])
app.include_router(training.router, prefix="/api/v1/training", tags=["微调服务"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["推理服务"])
app.include_router(assets.router, prefix="/api/v1/assets", tags=["数据资产"])
app.include_router(points.router, prefix="/api/v1/points", tags=["点位中心"])
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["评估服务"])

# 导入并注册迭代版本管理路由
from src.api import iteration
app.include_router(iteration.router, prefix="/api/v1/iteration", tags=["迭代管理"])

# 挂载 Gradio 微调界面到 /train-ui
app = gr.mount_gradio_app(app, training_ui, path="/train-ui")

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "train_ui": "/train-ui",  # 新增：微调界面入口
        "apis": {
            "data": "/api/v1/data",
            "annotation": "/api/v1/annotation",
            "training": "/api/v1/training",
            "inference": "/api/v1/inference",
            "assets": "/api/v1/assets",
            "points": "/api/v1/points",
            "evaluation": "/api/v1/evaluation",
        }
    }

@app.get("/train")
async def redirect_to_train_ui():
    """重定向到微调界面"""
    return RedirectResponse(url="/train-ui")

if __name__ == "__main__":
    # 启动 Annotator 后端子进程
    logger.info("-" * 60)
    logger.info("🚀正在启动 Annotator 后端服务 (Port: 5000)...")
    annotator_process = subprocess.Popen([sys.executable, "-m", "services.annotator.backend.app"])
    
    def cleanup():
        if annotator_process.poll() is None:
            logger.info("🛑 正在停止 Annotator 服务 (PID: %s)...", annotator_process.pid)
            annotator_process.terminate()
            annotator_process.wait()
            logger.info("✅ Annotator 服务已停止")
            
    atexit.register(cleanup)
    
    # 等待几秒让后端启动
    time.sleep(2)
    logger.info("-" * 60)

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
