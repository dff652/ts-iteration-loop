"""
数据服务 API
封装 Data-Processing 项目功能
"""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
import uuid

from src.core.logging_config import get_logger
from src.core.tasks import celery_app
from src.db.database import get_db, Task, IotdbSource
from src.models.schemas import (
    AcquireTaskRequest,
    IotdbSourceCreate, IotdbSourceUpdate, IotdbSourceAcquireRequest,
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


def _normalize_dataset_filename(raw_name: str | None, fallback: str = "dataset") -> str:
    text = Path(str(raw_name or fallback)).name.strip()
    if not text:
        text = fallback
    if text.lower().endswith(".csv"):
        text = text[:-4]
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not text:
        text = fallback
    return f"{text}.csv"


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


@router.post("/datasets/upload", response_model=ApiResponse)
async def upload_dataset_csv(
    file: UploadFile = File(...),
    dataset_name: str | None = Form(default=None),
    overwrite: bool = Form(default=False),
):
    """上传 CSV 并创建可用于推理的数据集文件。"""
    filename = str(file.filename or "").strip()
    if not filename or not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持上传 CSV 文件")

    target_filename = _normalize_dataset_filename(dataset_name or filename, fallback="uploaded_dataset")
    target_path = Path(adapter.data_path) / target_filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not overwrite:
        raise HTTPException(status_code=400, detail=f"数据集已存在: {target_filename}（可启用 overwrite）")

    try:
        with target_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    finally:
        await file.close()

    if not target_path.exists() or target_path.stat().st_size <= 0:
        raise HTTPException(status_code=400, detail="上传失败或文件为空")

    return ApiResponse(
        success=True,
        data={
            "filename": target_filename,
            "path": str(target_path),
            "size_bytes": int(target_path.stat().st_size),
        },
        message="CSV 数据集上传成功",
    )


@router.post("/datasets/acquire", response_model=TaskResponse)
async def create_dataset_from_iotdb(
    request: AcquireTaskRequest,
    db: Session = Depends(get_db),
):
    """语义化入口：通过 IoTDB 采集创建数据集。"""
    return await start_acquire_task(request=request, db=db)


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


# ==================== IoTDB 数据源管理 ====================

def _source_to_dict(src: IotdbSource) -> dict:
    return {
        "id": src.id,
        "name": src.name,
        "host": src.host,
        "port": src.port,
        "username": src.username,
        "source_path": src.source_path,
        "point_name": src.point_name,
        "target_points": src.target_points,
        "description": src.description,
        "created_at": _dt_iso(src.created_at),
        "updated_at": _dt_iso(src.updated_at),
    }


@router.get("/sources", response_model=ApiResponse)
async def list_iotdb_sources(db: Session = Depends(get_db)):
    """列出所有 IoTDB 数据源配置"""
    sources = db.query(IotdbSource).order_by(IotdbSource.created_at.desc()).all()
    return ApiResponse(
        success=True,
        data={"sources": [_source_to_dict(s) for s in sources]},
        message=f"找到 {len(sources)} 个数据源配置",
    )


@router.post("/sources", response_model=ApiResponse)
async def create_iotdb_source(
    request: IotdbSourceCreate,
    db: Session = Depends(get_db),
):
    """创建 IoTDB 数据源配置"""
    source = IotdbSource(
        id=str(uuid.uuid4()),
        name=request.name.strip(),
        host=request.host.strip(),
        port=request.port.strip(),
        username=request.username.strip(),
        password=request.password.strip(),
        source_path=request.source_path.strip(),
        point_name=request.point_name.strip(),
        target_points=request.target_points,
        description=(request.description or "").strip() or None,
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    logger.info("创建 IoTDB 数据源: %s (%s)", source.name, source.id)
    return ApiResponse(
        success=True,
        data={"source": _source_to_dict(source)},
        message="数据源创建成功",
    )


@router.put("/sources/{source_id}", response_model=ApiResponse)
async def update_iotdb_source(
    source_id: str,
    request: IotdbSourceUpdate,
    db: Session = Depends(get_db),
):
    """更新 IoTDB 数据源配置"""
    source = db.query(IotdbSource).filter(IotdbSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="数据源不存在")

    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(source, key, value.strip() if isinstance(value, str) else value)
    db.commit()
    db.refresh(source)
    logger.info("更新 IoTDB 数据源: %s (%s)", source.name, source.id)
    return ApiResponse(
        success=True,
        data={"source": _source_to_dict(source)},
        message="数据源更新成功",
    )


@router.delete("/sources/{source_id}", response_model=ApiResponse)
async def delete_iotdb_source(
    source_id: str,
    db: Session = Depends(get_db),
):
    """删除 IoTDB 数据源配置"""
    source = db.query(IotdbSource).filter(IotdbSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="数据源不存在")

    db.delete(source)
    db.commit()
    logger.info("删除 IoTDB 数据源: %s (%s)", source.name, source_id)
    return ApiResponse(success=True, message="数据源删除成功")


@router.post("/sources/{source_id}/acquire", response_model=TaskResponse)
async def acquire_from_source(
    source_id: str,
    request: IotdbSourceAcquireRequest,
    db: Session = Depends(get_db),
):
    """使用指定数据源配置启动采集任务"""
    source = db.query(IotdbSource).filter(IotdbSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="数据源不存在")

    acquire_request = AcquireTaskRequest(
        source=source.source_path,
        host=source.host,
        port=source.port,
        user=source.username,
        password=source.password,
        point_name=source.point_name,
        target_points=request.target_points or source.target_points,
        start_time=request.start_time,
        end_time=request.end_time,
    )
    return await start_acquire_task(request=acquire_request, db=db)
