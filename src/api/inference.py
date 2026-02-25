"""
推理服务 API
封装 check_outlier 项目功能
"""
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid

from configs.settings import settings
from src.core.logging_config import get_logger
from src.core.tasks import celery_app
from src.db.database import get_db, Task
from src.models.schemas import (
    InferenceTaskRequest,
    TaskResponse, TaskStatus, ApiResponse
)
from src.adapters.check_outlier import CheckOutlierAdapter
from src.utils.time_utils import utc_now_naive

logger = get_logger(__name__)
router = APIRouter()
adapter = CheckOutlierAdapter()


def _dispatch_inference_task(task_id: str, model: str, algorithm: str, input_files: list, params: dict | None = None) -> str:
    """提交推理任务到 Celery，并返回 celery task id。"""
    async_result = celery_app.send_task(
        "inference.batch",
        kwargs={
            "task_id": task_id,
            "model": model,
            "algorithm": algorithm,
            "input_files": input_files,
            "params": params or {},
        },
    )
    return str(async_result.id)


def _dt_iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _task_output_dir(task: Task) -> str | None:
    try:
        cfg = json.loads(task.config or "{}")
    except Exception:
        cfg = {}
    algorithm = str(cfg.get("algorithm") or "").strip()
    if not algorithm:
        return None
    return str(Path(settings.DATA_INFERENCE_DIR) / algorithm)


def _build_task_status_payload(task: Task) -> dict:
    status_text = str(task.status or "").lower()
    if status_text == TaskStatus.FAILED:
        msg = f"推理失败: {task.error or 'unknown error'}"
    elif status_text == TaskStatus.COMPLETED:
        msg = "推理已完成"
    elif status_text == TaskStatus.CANCELLED:
        msg = "推理已取消"
    elif status_text == TaskStatus.RUNNING:
        msg = "推理执行中"
    else:
        msg = "任务排队中"

    return {
        "task_id": task.id,
        "type": task.type,
        "status": status_text or "unknown",
        "message": msg,
        "error": task.error or "",
        "progress": {"status": status_text or "unknown", "progress": 0},
        "output_dir": _task_output_dir(task),
        "log_path": None,
        "started_at": _dt_iso(task.started_at),
        "completed_at": _dt_iso(task.completed_at),
    }


def _build_task_log_text(task: Task) -> str:
    rows: list[str] = []
    if task.result:
        try:
            payload = json.loads(task.result)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            total = payload.get("total")
            successful = payload.get("successful")
            if total is not None or successful is not None:
                rows.append(f"summary: total={total}, successful={successful}")
            for err in payload.get("errors") or []:
                if not isinstance(err, dict):
                    continue
                file_text = str(err.get("file") or "")
                err_text = str(err.get("error") or "")
                rows.append(f"error: {file_text} | {err_text}".strip())
            for item in payload.get("results") or []:
                if not isinstance(item, dict):
                    continue
                file_text = str(item.get("file") or "")
                ok = bool(item.get("success"))
                rows.append(f"result: {file_text} | success={ok}")
    if task.error:
        rows.append(f"task_error: {task.error}")
    return "\n".join(rows).strip()


@router.get("/algorithms", response_model=ApiResponse)
async def list_algorithms():
    """获取可用算法列表"""
    algorithms = [
        {"id": "chatts", "name": "ChatTS", "description": "ChatTS 大模型检测"},
        {"id": "adtk_hbos", "name": "ADTK-HBOS", "description": "传统统计方法"},
        {"id": "ensemble", "name": "Ensemble", "description": "集成方法"}
    ]
    return ApiResponse(
        success=True,
        data={"algorithms": algorithms},
        message=f"支持 {len(algorithms)} 种算法"
    )


@router.post("/batch", response_model=TaskResponse)
async def start_batch_inference(
    request: InferenceTaskRequest,
    db: Session = Depends(get_db)
):
    """启动批量推理任务"""
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    task = Task(
        id=task_id,
        type="inference",
        status=TaskStatus.PENDING,
        config=request.model_dump_json()
    )
    db.add(task)
    db.commit()

    try:
        celery_task_id = _dispatch_inference_task(
            task_id=task_id,
            model=request.model,
            algorithm=request.algorithm,
            input_files=request.input_files,
            params=request.params or {},
        )
    except Exception as e:
        logger.error("提交 Celery 推理任务失败 task_id=%s: %s", task_id, e, exc_info=True)
        task.status = TaskStatus.FAILED
        task.error = f"Celery dispatch failed: {e}"
        db.commit()
        raise HTTPException(status_code=503, detail="任务队列不可用，提交失败")

    # 在 config 中补充调度元信息，便于排障
    try:
        config_data = request.model_dump()
    except Exception:
        config_data = {}
    config_data.update({
        "executor": "celery",
        "celery_task_id": celery_task_id,
    })
    task.config = json.dumps(config_data, ensure_ascii=False)
    db.commit()

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"推理任务已提交，处理 {len(request.input_files)} 个文件"
    )


@router.post("/cancel/{task_id}", response_model=TaskResponse)
async def cancel_inference_task(task_id: str, db: Session = Depends(get_db)):
    """取消推理任务（best effort）。"""
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
        logger.warning("撤销 Celery 任务失败 task_id=%s celery_id=%s: %s", task_id, celery_task_id, e)

    task.status = TaskStatus.CANCELLED
    task.completed_at = utc_now_naive()
    db.commit()
    return TaskResponse(task_id=task.id, status=TaskStatus.CANCELLED, message="任务已取消")


@router.get("/status/{task_id}", response_model=ApiResponse)
async def get_inference_status(task_id: str, db: Session = Depends(get_db)):
    """获取推理任务状态"""
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
async def get_inference_log(
    task_id: str,
    offset: int = 0,
    max_bytes: int = 200000,
    db: Session = Depends(get_db),
):
    """获取推理任务日志（增量）。"""
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


@router.get("/results/{task_id}", response_model=ApiResponse)
async def get_inference_results(task_id: str, db: Session = Depends(get_db)):
    """获取推理结果"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.COMPLETED:
        return ApiResponse(
            success=False,
            message=f"任务尚未完成，当前状态: {task.status}"
        )
    
    import json
    results = json.loads(task.result) if task.result else {}
    
    return ApiResponse(
        success=True,
        data=results,
        message="推理结果获取成功"
    )


@router.post("/export-to-annotation/{task_id}", response_model=ApiResponse)
async def export_to_annotation(
    task_id: str,
    persist_file: bool = False,
    db: Session = Depends(get_db),
):
    """将推理结果导出为预标注格式（默认返回内存 rows，可选兼容落盘）。"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status != TaskStatus.COMPLETED:
        return ApiResponse(
            success=False,
            data={"task_id": task.id, "status": str(task.status)},
            message=f"任务尚未完成，当前状态: {task.status}",
        )
    
    try:
        rows = adapter.to_annotation_rows(task.result)
        data = {
            "rows": rows,
            "row_count": len(rows),
        }
        if persist_file:
            output_path = adapter.convert_to_annotation_format(task.result)
            data["annotation_file"] = output_path

        message = "已转换为预标注行，可直接导入标注服务"
        if persist_file:
            message = "已导出预标注行并生成兼容文件，可在标注工具中加载"

        return ApiResponse(
            success=True,
            data=data,
            message=message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
