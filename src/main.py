"""
TS-Iteration-Loop 主入口
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import gradio as gr

from configs.settings import settings
from src.db.database import init_db

# 导入 API 路由
from src.api import data, annotation, training, inference

# 导入 Gradio 界面
from src.webui.training_ui import training_ui

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="时序异常检测迭代循环系统 API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(data.router, prefix="/api/v1/data", tags=["数据服务"])
app.include_router(annotation.router, prefix="/api/v1/annotation", tags=["标注服务"])
app.include_router(training.router, prefix="/api/v1/training", tags=["微调服务"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["推理服务"])

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
            "inference": "/api/v1/inference"
        }
    }

@app.get("/train")
async def redirect_to_train_ui():
    """重定向到微调界面"""
    return RedirectResponse(url="/train-ui")

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    init_db()
    print("=" * 60)
    print(f"🚀 {settings.APP_NAME} 启动成功")
    print(f"📖 API 文档: http://localhost:{settings.API_PORT}/docs")
    print(f"🎯 微调界面: http://localhost:{settings.API_PORT}/train-ui")
    print("=" * 60)

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
