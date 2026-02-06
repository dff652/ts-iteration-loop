"""
数据资产管理 API
将数据集构建相关逻辑从 UI 直连 DB 下沉到正式接口层。
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from configs.settings import settings
from src.adapters.data_processing import DataProcessingAdapter
from src.adapters.chatts_training import ChatTSTrainingAdapter
from src.db.database import (
    get_db,
    DatasetAsset,
    DatasetItem,
    InferenceResult,
    ReviewQueue,
)
from src.models.schemas import ApiResponse

router = APIRouter()


class AssetSaveRequest(BaseModel):
    name: str
    dataset_type: str = "train"  # train / golden
    items: List[str]
    note: str = ""
    overwrite: bool = False
    freeze: bool = False


class ExportTrainingRequest(BaseModel):
    dataset_id: str
    model_family: str = "chatts"  # chatts / qwen
    output_name: Optional[str] = None
    approved_only: bool = True


def _meta_note(meta_text: Optional[str]) -> str:
    if not meta_text:
        return ""
    try:
        return str(json.loads(meta_text).get("note", ""))
    except Exception:
        return ""


def _normalize_point_name(name: str) -> str:
    text = str(name or "")
    text = text.replace(".csv", "").replace(".json", "")
    if text.startswith("annotations_"):
        text = text.replace("annotations_", "", 1)
    text = text.replace("数据集", "")
    return text


@router.get("/datasets", response_model=ApiResponse)
async def list_assets(
    dataset_type: Optional[str] = Query(None, description="train or golden"),
    db: Session = Depends(get_db),
):
    query = db.query(DatasetAsset)
    if dataset_type:
        query = query.filter(DatasetAsset.dataset_type == dataset_type)
    rows = query.order_by(DatasetAsset.created_at.desc()).all()

    assets = []
    for r in rows:
        assets.append(
            {
                "id": r.id,
                "name": r.name,
                "dataset_type": r.dataset_type,
                "status": r.status,
                "point_count": r.point_count or 0,
                "note": _meta_note(r.meta),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
        )

    return ApiResponse(success=True, data={"assets": assets}, message=f"找到 {len(assets)} 个数据集")


@router.get("/datasets/{dataset_id}", response_model=ApiResponse)
async def get_asset(dataset_id: str, db: Session = Depends(get_db)):
    asset = db.query(DatasetAsset).filter(DatasetAsset.id == dataset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    items = db.query(DatasetItem.point_name).filter(DatasetItem.dataset_id == dataset_id).all()
    points = sorted([str(r[0]) for r in items])

    return ApiResponse(
        success=True,
        data={
            "asset": {
                "id": asset.id,
                "name": asset.name,
                "dataset_type": asset.dataset_type,
                "status": asset.status,
                "point_count": asset.point_count or len(points),
                "note": _meta_note(asset.meta),
                "created_at": asset.created_at.isoformat() if asset.created_at else None,
            },
            "items": points,
        },
        message="获取成功",
    )


@router.post("/datasets/save", response_model=ApiResponse)
async def save_asset(request: AssetSaveRequest, db: Session = Depends(get_db)):
    name = (request.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="必须输入数据集名称")

    if request.dataset_type not in {"train", "golden"}:
        raise HTTPException(status_code=400, detail="dataset_type 必须是 train 或 golden")

    items = sorted(set([_normalize_point_name(x) for x in request.items if str(x).strip()]))
    if not items:
        raise HTTPException(status_code=400, detail="数据集条目为空")

    asset = db.query(DatasetAsset).filter(DatasetAsset.name == name).first()
    if asset and not request.overwrite:
        raise HTTPException(status_code=400, detail="数据集已存在（需启用覆盖）")
    if asset and asset.status == "frozen":
        raise HTTPException(status_code=400, detail="目标数据集已冻结")

    status = "frozen" if (request.dataset_type == "golden" or request.freeze) else "draft"
    meta_text = json.dumps({"note": request.note}, ensure_ascii=False)

    if not asset:
        asset = DatasetAsset(
            id=str(uuid.uuid4()),
            name=name,
            dataset_type=request.dataset_type,
            status=status,
            point_count=len(items),
            meta=meta_text,
        )
        db.add(asset)
    else:
        asset.dataset_type = request.dataset_type
        asset.status = status
        asset.point_count = len(items)
        asset.meta = meta_text
        db.query(DatasetItem).filter(DatasetItem.dataset_id == asset.id).delete()

    for p in items:
        db.add(DatasetItem(dataset_id=asset.id, point_name=p))

    db.commit()

    return ApiResponse(
        success=True,
        data={"id": asset.id, "name": asset.name, "point_count": len(items)},
        message=f"已保存: {asset.name} ({len(items)} pts)",
    )


@router.delete("/datasets/{dataset_id}", response_model=ApiResponse)
async def delete_asset(dataset_id: str, db: Session = Depends(get_db)):
    asset = db.query(DatasetAsset).filter(DatasetAsset.id == dataset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).delete()
    db.delete(asset)
    db.commit()
    return ApiResponse(success=True, data={"id": dataset_id}, message="已删除")


@router.get("/sources", response_model=ApiResponse)
async def list_source_items(
    source_type: str = Query("annotations", description="annotations or inference"),
    method: Optional[str] = Query(None, description="chatts/qwen/timer/adtk_hbos/ensemble"),
    min_score: Optional[float] = Query(None),
    max_score: Optional[float] = Query(None),
    keyword: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    source_type = (source_type or "annotations").strip().lower()
    keyword = (keyword or "").strip().lower()

    if source_type == "annotations":
        ann_dir = Path(settings.ANNOTATIONS_ROOT) / settings.DEFAULT_USER
        items = []
        if ann_dir.exists():
            for f in ann_dir.glob("*.json"):
                try:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                    filename = payload.get("filename") or f.stem
                except Exception:
                    filename = f.stem
                name = _normalize_point_name(filename)
                if keyword and keyword not in name.lower():
                    continue
                items.append(name)

        values = sorted(set(items))
        choices = [{"label": v, "value": v} for v in values]
        return ApiResponse(success=True, data={"choices": choices, "count": len(choices)}, message="ok")

    if source_type != "inference":
        raise HTTPException(status_code=400, detail="source_type 仅支持 annotations 或 inference")

    query = db.query(InferenceResult)
    if method:
        query = query.filter(InferenceResult.method == method)
    if min_score is not None:
        query = query.filter(InferenceResult.score_avg >= min_score)
    if max_score is not None:
        query = query.filter(InferenceResult.score_avg <= max_score)

    rows = query.order_by(InferenceResult.score_avg.desc()).limit(limit).all()
    seen = set()
    choices = []
    for r in rows:
        point_name = (r.point_name or "").strip()
        if not point_name:
            continue
        if keyword and keyword not in point_name.lower():
            continue
        if point_name in seen:
            continue
        seen.add(point_name)
        label = f"{point_name} | Score: {(r.score_avg or 0):.2f} ({r.method or 'unknown'})"
        choices.append({"label": label, "value": point_name})

    return ApiResponse(success=True, data={"choices": choices, "count": len(choices)}, message="ok")


@router.post("/export/training", response_model=ApiResponse)
async def export_training_dataset(request: ExportTrainingRequest, db: Session = Depends(get_db)):
    asset = db.query(DatasetAsset).filter(DatasetAsset.id == request.dataset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if asset.dataset_type != "train":
        raise HTTPException(status_code=400, detail="仅允许导出 train 数据集")

    rows = db.query(DatasetItem.point_name).filter(DatasetItem.dataset_id == request.dataset_id).all()
    point_set = {_normalize_point_name(r[0]) for r in rows if r and r[0]}
    if not point_set:
        raise HTTPException(status_code=400, detail="数据集为空")

    approved_set = None
    if request.approved_only:
        approved_rows = db.query(ReviewQueue.source_id).filter(
            ReviewQueue.source_type == "annotation",
            ReviewQueue.status == "approved",
        ).all()
        approved_set = {_normalize_point_name(r[0]) for r in approved_rows if r and r[0]}

    ann_dir = Path(settings.ANNOTATIONS_ROOT) / settings.DEFAULT_USER
    if not ann_dir.exists():
        raise HTTPException(status_code=404, detail="标注目录不存在")

    adapter = DataProcessingAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        selected = 0
        for json_file in ann_dir.glob("*.json"):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            filename = payload.get("filename") or json_file.stem
            normalized = _normalize_point_name(filename)
            if normalized not in point_set:
                continue
            if approved_set is not None and normalized not in approved_set:
                continue

            out_name = json_file.name if json_file.name.endswith(".json") else f"{json_file.stem}.json"
            (tmp_path / out_name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            selected += 1

        if selected == 0:
            raise HTTPException(status_code=400, detail="未匹配到可导出的标注")

        output_base = (request.output_name or "").strip() or asset.name
        out_dir = settings.DATA_TRAINING_QWEN_DIR if request.model_family == "qwen" else settings.DATA_TRAINING_CHATTS_DIR
        out_path = Path(out_dir) / f"{output_base}.jsonl"

        result = adapter.convert_annotations(
            input_dir=str(tmp_path),
            output_path=str(out_path),
            image_dir=settings.DATA_IMAGES_DIR,
            model_family=request.model_family,
            csv_src_dir=settings.DATA_DOWNSAMPLED_DIR,
        )

    if not result.get("success"):
        error_text = result.get("error") or result.get("stderr") or "导出失败"
        raise HTTPException(status_code=500, detail=error_text)

    # Refresh dataset_info.json for selected model family.
    ChatTSTrainingAdapter(model_family=request.model_family).get_dataset_list()

    return ApiResponse(
        success=True,
        data={"output_path": result.get("output_path"), "selected_count": selected},
        message="导出完成",
    )
