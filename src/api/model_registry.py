"""
模型注册表 API
提供模型的 CRUD、版本管理、扫描注册和对比功能。
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from configs.settings import settings
from src.core.logging_config import get_logger
from src.db.database import ModelRegistry, ModelEval, get_db
from src.utils.time_utils import utc_now_naive

logger = get_logger(__name__)
router = APIRouter()


# ==================== Request / Response Models ====================

class ModelRegisterRequest(BaseModel):
    name: str
    model_family: str = "chatts"
    model_type: str = "lora"
    version: Optional[str] = None
    model_path: str
    base_model: Optional[str] = None
    config: Optional[dict] = None
    metrics: Optional[dict] = None
    train_loss: Optional[float] = None
    status: str = "active"
    tags: Optional[list] = None
    description: Optional[str] = None
    source_task_id: Optional[str] = None


class ModelUpdateRequest(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[list] = None
    description: Optional[str] = None
    metrics: Optional[dict] = None


# ==================== Helpers ====================

def _json_dumps(obj) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_loads(text: Optional[str], default=None):
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _model_to_dict(m: ModelRegistry) -> dict:
    return {
        "id": m.id,
        "name": m.name,
        "model_family": m.model_family,
        "model_type": m.model_type,
        "version": m.version,
        "model_path": m.model_path,
        "base_model": m.base_model,
        "config": _json_loads(m.config),
        "metrics": _json_loads(m.metrics),
        "train_loss": m.train_loss,
        "status": m.status,
        "tags": _json_loads(m.tags, []),
        "description": m.description,
        "source_task_id": m.source_task_id,
        "created_by": m.created_by,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
    }


def _dt_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


# ==================== Routes ====================

@router.get("/")
def list_models(
    model_family: Optional[str] = Query(None),
    model_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """列表查询模型"""
    query = db.query(ModelRegistry)

    if model_family:
        query = query.filter(ModelRegistry.model_family == model_family)
    if model_type:
        query = query.filter(ModelRegistry.model_type == model_type)
    if status:
        query = query.filter(ModelRegistry.status == status)
    else:
        query = query.filter(ModelRegistry.status != "deleted")
    if keyword:
        kw = f"%{keyword}%"
        query = query.filter(
            ModelRegistry.name.ilike(kw) | ModelRegistry.model_path.ilike(kw)
        )

    total = query.count()
    rows = query.order_by(ModelRegistry.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "success": True,
        "data": {
            "total": total,
            "models": [_model_to_dict(r) for r in rows],
        },
    }


@router.post("/")
def register_model(request: ModelRegisterRequest, db: Session = Depends(get_db)):
    """注册新模型"""
    # 检查路径唯一性
    existing = db.query(ModelRegistry).filter(
        ModelRegistry.model_path == request.model_path,
        ModelRegistry.status != "deleted",
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"模型路径已注册: {request.model_path} (id={existing.id})")

    model_id = str(uuid.uuid4())
    record = ModelRegistry(
        id=model_id,
        name=request.name,
        model_family=request.model_family,
        model_type=request.model_type,
        version=request.version,
        model_path=request.model_path,
        base_model=request.base_model,
        config=_json_dumps(request.config),
        metrics=_json_dumps(request.metrics),
        train_loss=request.train_loss,
        status=request.status,
        tags=_json_dumps(request.tags),
        description=request.description,
        source_task_id=request.source_task_id,
        created_by=settings.DEFAULT_USER,
        created_at=utc_now_naive(),
        updated_at=utc_now_naive(),
    )
    db.add(record)
    db.commit()

    return {
        "success": True,
        "data": {"id": model_id},
        "message": f"模型已注册: {request.name}",
    }


@router.get("/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db)):
    """获取模型详情"""
    record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="模型不存在")

    result = _model_to_dict(record)

    # 附加评估记录
    evals = db.query(ModelEval).filter(ModelEval.model_path == record.model_path).all()
    result["evaluations"] = [
        {
            "id": e.id,
            "dataset_name": e.dataset_name,
            "metrics": _json_loads(e.metrics),
            "created_at": _dt_iso(e.created_at),
        }
        for e in evals
    ]

    return {"success": True, "data": result}


@router.put("/{model_id}")
def update_model(model_id: str, request: ModelUpdateRequest, db: Session = Depends(get_db)):
    """更新模型信息"""
    record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="模型不存在")

    if request.name is not None:
        record.name = request.name
    if request.version is not None:
        record.version = request.version
    if request.status is not None:
        record.status = request.status
    if request.tags is not None:
        record.tags = _json_dumps(request.tags)
    if request.description is not None:
        record.description = request.description
    if request.metrics is not None:
        record.metrics = _json_dumps(request.metrics)

    record.updated_at = utc_now_naive()
    db.commit()

    return {"success": True, "message": "模型信息已更新"}


@router.delete("/{model_id}")
def delete_model(model_id: str, db: Session = Depends(get_db)):
    """归档/删除模型（软删除）"""
    record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="模型不存在")

    record.status = "deleted"
    record.updated_at = utc_now_naive()
    db.commit()

    return {"success": True, "message": "模型已归档"}


@router.get("/{model_id}/loss")
def get_model_loss(model_id: str, db: Session = Depends(get_db)):
    """获取模型 Loss 曲线数据"""
    record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="模型不存在")

    from src.adapters.chatts_training import ChatTSTrainingAdapter
    adapter = ChatTSTrainingAdapter()
    logs = adapter.get_training_log(record.model_path)

    loss_data = []
    if logs:
        for entry in logs:
            if "loss" in entry:
                loss_data.append({
                    "step": entry.get("current_steps", 0),
                    "loss": entry.get("loss"),
                    "epoch": entry.get("epoch"),
                })

    return {"success": True, "data": {"loss_data": loss_data}}


@router.post("/compare")
def compare_models(
    model_ids: list[str] = [],
    db: Session = Depends(get_db),
):
    """多模型对比"""
    if not model_ids or len(model_ids) < 2:
        raise HTTPException(status_code=400, detail="请至少选择两个模型")

    records = db.query(ModelRegistry).filter(ModelRegistry.id.in_(model_ids)).all()
    if len(records) < 2:
        raise HTTPException(status_code=404, detail="未找到足够模型")

    comparison = []
    for record in records:
        item = _model_to_dict(record)
        # 附加最新评估
        eval_record = (
            db.query(ModelEval)
            .filter(ModelEval.model_path == record.model_path)
            .order_by(ModelEval.created_at.desc())
            .first()
        )
        item["latest_eval"] = (
            {
                "dataset_name": eval_record.dataset_name,
                "metrics": _json_loads(eval_record.metrics),
                "created_at": _dt_iso(eval_record.created_at),
            }
            if eval_record
            else None
        )
        comparison.append(item)

    return {"success": True, "data": {"models": comparison}}


@router.post("/scan")
def scan_models(
    model_family: str = "chatts",
    db: Session = Depends(get_db),
):
    """扫描磁盘已有模型，批量注册"""
    from src.adapters.chatts_training import ChatTSTrainingAdapter

    adapter = ChatTSTrainingAdapter()
    if model_family == "qwen":
        try:
            from src.adapters.chatts_training import QwenTrainingAdapter
            adapter = QwenTrainingAdapter()
        except ImportError:
            pass

    disk_models = adapter.list_models()
    saves_prefix = str(adapter.saves_path).replace("\\", "/") + "/"

    # 只扫描 saves 目录下的模型
    disk_models = [
        m for m in disk_models
        if str(m.get("path", "")).replace("\\", "/").startswith(saves_prefix)
    ]

    registered_paths = set()
    existing = db.query(ModelRegistry.model_path).filter(
        ModelRegistry.status != "deleted"
    ).all()
    for row in existing:
        registered_paths.add(str(row[0]))

    registered_count = 0
    skipped_count = 0

    for m in disk_models:
        model_path = str(m.get("path", ""))
        if model_path in registered_paths:
            skipped_count += 1
            continue

        # 过滤 checkpoint 目录
        name = str(m.get("name", ""))
        if name.startswith("checkpoint-") or "/checkpoint-" in model_path:
            continue

        train_results = m.get("train_results", {})

        record = ModelRegistry(
            id=str(uuid.uuid4()),
            name=name,
            model_family=model_family,
            model_type=str(m.get("type", "lora")),
            model_path=model_path,
            train_loss=train_results.get("train_loss") if train_results else None,
            config=_json_dumps({
                "global_step": m.get("global_step"),
                "checkpoints": m.get("checkpoints", []),
            }),
            status="active",
            created_by=settings.DEFAULT_USER,
            created_at=utc_now_naive(),
            updated_at=utc_now_naive(),
        )
        db.add(record)
        registered_count += 1

    db.commit()

    return {
        "success": True,
        "data": {
            "registered": registered_count,
            "skipped": skipped_count,
            "total_on_disk": len(disk_models),
        },
        "message": f"扫描完成: 新注册 {registered_count}, 已存在 {skipped_count}",
    }


@router.get("/versions/history")
def list_versions(
    model_family: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """按 model_family 查看版本历史"""
    query = db.query(ModelRegistry).filter(ModelRegistry.status != "deleted")
    if model_family:
        query = query.filter(ModelRegistry.model_family == model_family)

    rows = query.order_by(ModelRegistry.created_at.desc()).all()

    versions = {}
    for r in rows:
        key = r.model_family or "unknown"
        if key not in versions:
            versions[key] = []
        versions[key].append({
            "id": r.id,
            "name": r.name,
            "version": r.version,
            "model_type": r.model_type,
            "train_loss": r.train_loss,
            "status": r.status,
            "created_at": _dt_iso(r.created_at),
        })

    return {"success": True, "data": {"versions": versions}}
