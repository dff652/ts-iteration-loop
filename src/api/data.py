"""
数据服务 API
封装 Data-Processing 项目功能
"""
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid

from src.core.logging_config import get_logger
from src.core.tasks import celery_app
from src.db.database import get_db, Task
from src.models.schemas import (
    AcquireTaskRequest,
    TaskResponse, TaskStatus, ApiResponse
)
from src.adapters.data_processing import DataProcessingAdapter

router = APIRouter()
adapter = DataProcessingAdapter()
logger = get_logger(__name__)


def _sanitize_task_config(config: dict) -> dict:
    masked = dict(config or {})
    if masked.get("password"):
        masked["password"] = "***"
    return masked


def _dispatch_acquire_task(task_id: str, request: AcquireTaskRequest) -> str:
    async_result = celery_app.send_task(
        "data.acquire",
        kwargs={
            "task_id": task_id,
            "source": request.source,
            "target_points": request.target_points,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "host": request.host,
            "port": request.port,
            "user": request.user,
            "password": request.password,
            "point_name": request.point_name,
        },
    )
    return str(async_result.id)


def _dt_iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _build_task_status_payload(task: Task) -> dict:
    status_text = str(task.status or "").lower()
    if status_text == TaskStatus.FAILED:
        msg = f"采集失败: {task.error or 'unknown error'}"
    elif status_text == TaskStatus.COMPLETED:
        msg = "采集已完成"
    elif status_text == TaskStatus.CANCELLED:
        msg = "采集已取消"
    elif status_text == TaskStatus.RUNNING:
        msg = "采集中"
    else:
        msg = "任务排队中"

    return {
        "task_id": task.id,
        "type": task.type,
        "status": status_text or "unknown",
        "message": msg,
        "error": task.error or "",
        "progress": {"status": status_text or "unknown", "progress": 0},
        "output_dir": str(adapter.data_path),
        "log_path": None,
        "started_at": _dt_iso(task.started_at),
        "completed_at": _dt_iso(task.completed_at),
    }


def _build_task_log_text(task: Task) -> str:
    text_chunks: list[str] = []
    if task.result:
        try:
            payload = json.loads(task.result)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            stdout = str(payload.get("stdout") or "")
            stderr = str(payload.get("stderr") or "")
            message = str(payload.get("message") or "")
            if message:
                text_chunks.append(message.strip())
            if stdout:
                text_chunks.append(stdout)
            if stderr:
                text_chunks.append(stderr)
    if task.error:
        text_chunks.append(str(task.error))
    return "\n".join([c for c in text_chunks if c]).strip()


@router.get("/datasets", response_model=ApiResponse)
async def list_datasets():
    """获取数据集列表"""
    try:
        datasets = adapter.list_datasets()
        return ApiResponse(
            success=True,
            data={"datasets": datasets},
            message=f"找到 {len(datasets)} 个数据集"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acquire", response_model=TaskResponse)
async def start_acquire_task(
    request: AcquireTaskRequest,
    db: Session = Depends(get_db)
):
    """启动数据采集任务"""
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    initial_config = _sanitize_task_config(request.model_dump())
    task = Task(
        id=task_id,
        type="acquire",
        status=TaskStatus.PENDING,
        config=json.dumps(initial_config, ensure_ascii=False)
    )
    db.add(task)
    db.commit()

    try:
        celery_task_id = _dispatch_acquire_task(task_id, request)
    except Exception as e:
        logger.error("提交 Celery 采集任务失败 task_id=%s: %s", task_id, e, exc_info=True)
        task.status = TaskStatus.FAILED
        task.error = f"Celery dispatch failed: {e}"
        db.commit()
        raise HTTPException(status_code=503, detail="任务队列不可用，提交失败")

    config_data = _sanitize_task_config(request.model_dump())
    config_data.update({
        "executor": "celery",
        "celery_task_id": celery_task_id,
    })
    task.config = json.dumps(config_data, ensure_ascii=False)
    db.commit()

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="数据采集任务已提交"
    )


@router.get("/status/{task_id}", response_model=ApiResponse)
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """获取任务状态"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    payload = _build_task_status_payload(task)
    return ApiResponse(
        success=True,
        data=payload,
        message=str(payload.get("message") or ""),
    )


@router.get("/log/{task_id}", response_model=ApiResponse)
async def get_task_log(
    task_id: str,
    offset: int = 0,
    max_bytes: int = 200000,
    db: Session = Depends(get_db),
):
    """获取数据采集任务日志（增量）。"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if offset < 0:
        offset = 0
    if max_bytes <= 0:
        max_bytes = 1
    max_bytes = min(max_bytes, 2_000_000)

    status_payload = _build_task_status_payload(task)
    log_text = _build_task_log_text(task)
    total = len(log_text)
    safe_offset = min(offset, total)
    chunk = log_text[safe_offset : safe_offset + max_bytes]
    new_offset = safe_offset + len(chunk)

    return ApiResponse(
        success=True,
        data={
            **status_payload,
            "log": chunk,
            "offset": new_offset,
            "exists": bool(log_text),
            "eof": new_offset >= total,
        },
        message=str(status_payload.get("message") or ""),
    )


@router.get("/preview/{filename}")
async def preview_data(filename: str, limit: int = 100):
    """预览数据文件"""
    try:
        data = adapter.preview_csv(filename, limit=limit)
        return ApiResponse(
            success=True,
            data={"filename": filename, "preview": data},
            message=f"显示前 {limit} 条数据"
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
