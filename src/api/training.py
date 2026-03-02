"""
微调服务 API
封装 ChatTS-Training 项目功能
"""
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import json
import uuid

from src.core.logging_config import get_logger
from src.core.tasks import celery_app
from src.db.database import get_db, Task
from src.models.schemas import (
    TrainingTaskRequest,
    TrainingEvalRequest,
    TaskResponse, TaskStatus, ApiResponse
)
from src.adapters.chatts_training import ChatTSTrainingAdapter
from src.utils.time_utils import utc_now_naive

router = APIRouter()
logger = get_logger(__name__)


def _dispatch_training_task(task_id: str, request: TrainingTaskRequest) -> str:
    async_result = celery_app.send_task(
        "training.run",
        kwargs={
            "task_id": task_id,
            "config_name": request.config_name,
            "version_tag": request.version_tag,
            "model_family": request.model_family,
            "auto_eval": request.auto_eval,
            "eval_truth_dir": request.eval_truth_dir,
            "eval_data_dir": request.eval_data_dir,
            "eval_dataset_name": request.eval_dataset_name,
            "eval_output_dir": request.eval_output_dir,
            "eval_device": request.eval_device,
            "eval_method": request.eval_method,
            "params": request.params or {},
        },
    )
    return str(async_result.id)

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


def _task_output_dir(task: Task) -> Path | None:
    """根据任务配置/结果推导训练输出目录。"""
    try:
        result_data = json.loads(task.result or "{}")
        if isinstance(result_data, dict):
            output_dir = result_data.get("output_dir")
            if output_dir:
                return Path(str(output_dir))
    except Exception:
        pass

    try:
        cfg = json.loads(task.config or "{}")
    except Exception:
        cfg = {}
    config_name = str(cfg.get("config_name") or "").strip()
    if not config_name:
        return None
    model_family = str(cfg.get("model_family") or "chatts")
    version_tag = str(cfg.get("version_tag") or "").strip() or task.id[:8]
    return get_adapter(model_family).saves_path / f"{config_name}_{version_tag}"


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

    try:
        celery_task_id = _dispatch_training_task(task_id, request)
    except Exception as e:
        logger.error("提交 Celery 训练任务失败 task_id=%s: %s", task_id, e, exc_info=True)
        task.status = TaskStatus.FAILED
        task.error = f"Celery dispatch failed: {e}"
        db.commit()
        raise HTTPException(status_code=503, detail="任务队列不可用，提交失败")

    config_data = request.model_dump()
    config_data.update({
        "executor": "celery",
        "celery_task_id": celery_task_id,
    })
    task.config = json.dumps(config_data, ensure_ascii=False)
    db.commit()

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

    status_text = str(task.status or "").lower()
    if status_text == TaskStatus.FAILED:
        msg = f"训练失败: {task.error or 'unknown error'}"
    elif status_text == TaskStatus.COMPLETED:
        msg = "训练已完成"
    elif status_text == TaskStatus.CANCELLED:
        msg = "训练已取消"
    elif status_text == TaskStatus.RUNNING:
        msg = "训练进行中"
    else:
        msg = "任务排队中"

    progress = {"status": status_text or "unknown", "progress": 0}
    if status_text in {TaskStatus.PENDING, TaskStatus.RUNNING}:
        try:
            progress = get_adapter_from_task(task).get_training_progress(task_id)
            if (
                not isinstance(progress, dict)
                or str(progress.get("status") or "").lower() in {"", "unknown"}
            ):
                progress = {"status": status_text or "unknown", "progress": 0}
        except Exception:
            progress = {"status": status_text or "unknown", "progress": 0}

    output_dir = _task_output_dir(task)
    log_path = str(output_dir / "train.log") if output_dir else None

    return ApiResponse(
        success=True,
        data={
            "task_id": task.id,
            "status": task.status,
            "progress": progress,
            "message": msg,
            "error": task.error or "",
            "output_dir": str(output_dir) if output_dir else None,
            "log_path": log_path,
        },
        message=msg
    )


@router.post("/stop/{task_id}", response_model=TaskResponse)
async def stop_training(task_id: str, db: Session = Depends(get_db)):
    """取消训练任务（best effort，优先 Celery revoke）。"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
        return TaskResponse(task_id=task.id, status=TaskStatus(task.status), message="任务已结束")

    celery_task_id = None
    try:
        cfg = json.loads(task.config or "{}")
        celery_task_id = cfg.get("celery_task_id")
    except Exception:
        celery_task_id = None

    try:
        if celery_task_id:
            celery_app.control.revoke(celery_task_id, terminate=True)
    except Exception as e:
        logger.warning("撤销 Celery 训练任务失败 task_id=%s celery_id=%s: %s", task_id, celery_task_id, e)

    try:
        # 兼容历史本地训练任务（非 Celery）兜底停止
        get_adapter_from_task(task).stop_training(task_id)
    except Exception:
        pass

    try:
        task.status = TaskStatus.CANCELLED
        task.completed_at = utc_now_naive()
        db.commit()

        return TaskResponse(
            task_id=task.id,
            status=TaskStatus.CANCELLED,
            message="训练已停止"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/log/{task_id}", response_model=ApiResponse)
async def get_training_task_log(
    task_id: str,
    offset: int = Query(0, ge=0),
    max_bytes: int = Query(200000, ge=1, le=2000000),
    db: Session = Depends(get_db),
):
    """按 offset 读取训练任务日志增量。"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    output_dir = _task_output_dir(task)
    if output_dir is None:
        return ApiResponse(
            success=True,
            data={
                "task_id": task.id,
                "status": task.status,
                "log": "",
                "offset": offset,
                "exists": False,
                "log_path": None,
            },
            message="任务输出目录尚不可用",
        )

    log_path = output_dir / "train.log"
    if not log_path.exists():
        return ApiResponse(
            success=True,
            data={
                "task_id": task.id,
                "status": task.status,
                "log": "",
                "offset": offset,
                "exists": False,
                "log_path": str(log_path),
            },
            message="日志文件尚未生成",
        )

    file_size = int(log_path.stat().st_size)
    safe_offset = min(int(offset), file_size)
    content = ""
    new_offset = safe_offset
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(safe_offset)
            content = f.read(max_bytes)
            new_offset = int(f.tell())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志失败: {e}")

    return ApiResponse(
        success=True,
        data={
            "task_id": task.id,
            "status": task.status,
            "log": content,
            "offset": new_offset,
            "exists": True,
            "log_path": str(log_path),
            "eof": new_offset >= file_size,
            "error": task.error or "",
        },
        message="",
    )


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
