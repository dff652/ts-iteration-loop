"""
微调服务 API
封装 ChatTS-Training 项目功能
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List
import json
import uuid

from src.db.database import get_db, Task
from src.models.schemas import (
    TrainingConfig, TrainingTaskRequest,
    TrainingEvalRequest,
    TaskResponse, TaskStatus, ApiResponse
)
from src.adapters.chatts_training import ChatTSTrainingAdapter

router = APIRouter()

def get_adapter(model_family: str) -> ChatTSTrainingAdapter:
    return ChatTSTrainingAdapter(model_family=model_family or "chatts")


def get_adapter_from_task(task: Task) -> ChatTSTrainingAdapter:
    model_family = "chatts"
    try:
        if task and task.config:
            cfg = json.loads(task.config)
            model_family = cfg.get("model_family", "chatts")
    except Exception:
        pass
    return get_adapter(model_family)


@router.get("/configs", response_model=ApiResponse)
async def list_training_configs(model_family: str = Query("chatts", description="chatts or qwen")):
    """获取训练配置列表"""
    try:
        configs = get_adapter(model_family).list_configs()
        return ApiResponse(
            success=True,
            data={"configs": configs},
            message=f"找到 {len(configs)} 个配置"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ApiResponse)
async def list_trained_models(model_family: str = Query("chatts", description="chatts or qwen")):
    """获取已训练模型列表"""
    try:
        models = get_adapter(model_family).list_models()
        return ApiResponse(
            success=True,
            data={"models": models},
            message=f"找到 {len(models)} 个模型"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=ApiResponse)
async def list_training_datasets(model_family: str = Query("chatts", description="chatts or qwen")):
    """获取训练数据集列表"""
    try:
        datasets = get_adapter(model_family).get_dataset_list()
        return ApiResponse(
            success=True,
            data={"datasets": datasets},
            message=f"找到 {len(datasets)} 个数据集"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=TaskResponse)
async def start_training(
    request: TrainingTaskRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """启动训练任务"""
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    task = Task(
        id=task_id,
        type="training",
        status=TaskStatus.PENDING,
        config=request.model_dump_json()
    )
    db.add(task)
    db.commit()
    
    # 后台执行训练
    background_tasks.add_task(
        get_adapter(request.model_family).run_training,
        task_id=task_id,
        config_name=request.config_name,
        version_tag=request.version_tag,
        auto_eval=request.auto_eval,
        eval_truth_dir=request.eval_truth_dir,
        eval_data_dir=request.eval_data_dir,
        eval_dataset_name=request.eval_dataset_name,
        eval_output_dir=request.eval_output_dir,
        eval_device=request.eval_device,
        eval_method=request.eval_method,
    )
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="训练任务已提交，请通过 /status 查询进度"
    )


@router.get("/status/{task_id}", response_model=ApiResponse)
async def get_training_status(task_id: str, db: Session = Depends(get_db)):
    """获取训练任务状态"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 获取训练进度
    progress = get_adapter_from_task(task).get_training_progress(task_id)
    
    return ApiResponse(
        success=True,
        data={
            "task_id": task.id,
            "status": task.status,
            "progress": progress
        },
        message=""
    )


@router.post("/stop/{task_id}", response_model=TaskResponse)
async def stop_training(task_id: str, db: Session = Depends(get_db)):
    """停止训练任务"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="任务未在运行中")
    
    try:
        get_adapter_from_task(task).stop_training(task_id)
        task.status = TaskStatus.CANCELLED
        db.commit()
        
        return TaskResponse(
            task_id=task.id,
            status=TaskStatus.CANCELLED,
            message="训练已停止"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=ApiResponse)
async def get_model_details(model_name: str, model_family: str = Query("chatts", description="chatts or qwen")):
    """获取模型详细信息（包含训练产物）"""
    try:
        # 查找模型
        models = get_adapter(model_family).list_models()
        model = next((m for m in models if m["name"] == model_name), None)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")
        
        # 获取详细信息
        details = get_adapter(model_family).get_model_details(model["path"])
        
        return ApiResponse(
            success=True,
            data=details,
            message="获取模型详情成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/log", response_model=ApiResponse)
async def get_model_training_log(model_name: str, model_family: str = Query("chatts", description="chatts or qwen")):
    """获取模型训练日志（用于绘制 loss 曲线）"""
    try:
        # 查找模型
        models = get_adapter(model_family).list_models()
        model = next((m for m in models if m["name"] == model_name), None)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")
        
        # 获取训练日志
        logs = get_adapter(model_family).get_training_log(model["path"])
        
        return ApiResponse(
            success=True,
            data={"logs": logs, "count": len(logs)},
            message=f"获取 {len(logs)} 条训练日志"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/loss-image")
async def get_model_loss_image(model_name: str, model_family: str = Query("chatts", description="chatts or qwen")):
    """获取模型 loss 曲线图片"""
    from fastapi.responses import FileResponse
    
    try:
        # 查找模型
        models = get_adapter(model_family).list_models()
        model = next((m for m in models if m["name"] == model_name), None)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")
        
        loss_image = model.get("loss_image")
        if not loss_image:
            raise HTTPException(status_code=404, detail="Loss 曲线图不存在")
        
        return FileResponse(
            loss_image,
            media_type="image/png",
            filename=f"{model_name}_loss.png"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=ApiResponse)
async def evaluate_model(request: TrainingEvalRequest):
    """手动触发模型评估（黄金集）"""
    try:
        from src.utils.model_eval import evaluate_model_on_golden
        result = evaluate_model_on_golden(
            model_path=request.model_path,
            model_family=request.model_family,
            truth_dir=request.truth_dir,
            data_dir=request.data_dir,
            dataset_name=request.dataset_name,
            output_dir=request.output_dir,
            device=request.device,
            method=request.method,
        )
        return ApiResponse(
            success=bool(result.get("success")),
            data=result,
            message="评估完成" if result.get("success") else (result.get("error") or "评估失败")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
