"""
数据资产管理 API
将数据集构建相关逻辑从 UI 直连 DB 下沉到正式接口层。
"""
from __future__ import annotations

import json
import re
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from configs.settings import settings
from src.adapters.data_processing import DataProcessingAdapter
from src.adapters.chatts_training import ChatTSTrainingAdapter
from src.db.database import (
    get_db,
    AnnotationRecord,
    DatasetAsset,
    DatasetItem,
    InferenceResult,
    ReviewQueue,
)
from src.models.schemas import ApiResponse
from src.utils.annotation_store import normalize_point_name, record_to_payload

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
    return normalize_point_name(name)


def _approved_annotation_point_set(db: Session) -> set[str]:
    rows = (
        db.query(ReviewQueue.source_id)
        .filter(
            ReviewQueue.source_type == "annotation",
            ReviewQueue.status == "approved",
        )
        .all()
    )
    return {_normalize_point_name(r[0]) for r in rows if r and r[0]}


_POINT_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9]+)")


def _extract_point_name_from_text(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None

    base = Path(raw).name
    matches = _POINT_PATTERN.findall(base)
    if matches:
        candidates = []
        for m in matches:
            lower = m.lower()
            if lower.endswith((".csv", ".json", ".jpg", ".png", ".jpeg")):
                continue
            candidates.append(m)
        if candidates:
            candidates.sort(key=lambda s: (s.count("_"), len(s)), reverse=True)
            return _normalize_point_name(candidates[0])

    stem = Path(base).stem
    normalized = _normalize_point_name(stem)
    return normalized or None


def _iter_training_records(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    yield item
        return

    if suffix != ".json":
        return

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            for item in data["data"]:
                if isinstance(item, dict):
                    yield item
            return
        yield data


def _extract_training_point_name(record: Dict) -> Optional[str]:
    for key in ("point_name", "source_id", "id", "filename"):
        value = record.get(key)
        if not value:
            continue
        point = _extract_point_name_from_text(str(value))
        if point:
            return point

    image = record.get("image")
    if image:
        point = _extract_point_name_from_text(str(image))
        if point:
            return point

    return None


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

    approved_set = _approved_annotation_point_set(db)
    invalid_items = [p for p in items if p not in approved_set]
    if invalid_items:
        sample = ", ".join(invalid_items[:5])
        raise HTTPException(
            status_code=400,
            detail=f"仅允许保存审核通过点位，未通过数量: {len(invalid_items)}，示例: {sample}",
        )

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
    source_kind: Optional[str] = Query(None, description="auto/human (only for annotations)"),
    model_family: Optional[str] = Query(None, description="chatts/qwen (for training)"),
    method: Optional[str] = Query(None, description="chatts/qwen/timer/adtk_hbos/ensemble"),
    sort_by: str = Query("score_desc", description="score_desc/score_asc/updated_desc/updated_asc/name_asc/name_desc"),
    approved_only: bool = Query(True, description="是否仅保留审核通过点位"),
    min_score: Optional[float] = Query(None),
    max_score: Optional[float] = Query(None),
    keyword: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    source_type = source_type if isinstance(source_type, str) else "annotations"
    source_kind = source_kind if isinstance(source_kind, str) else ""
    sort_by = sort_by if isinstance(sort_by, str) else "score_desc"
    keyword = keyword if isinstance(keyword, str) else ""

    source_type = (source_type or "annotations").strip().lower()
    source_kind = (source_kind or "").strip().lower() or None
    sort_by = (sort_by or "score_desc").strip().lower()
    keyword = (keyword or "").strip().lower()
    if source_kind not in {None, "auto", "human"}:
        raise HTTPException(status_code=400, detail="source_kind 仅支持 auto 或 human")
    if sort_by not in {"score_desc", "score_asc", "updated_desc", "updated_asc", "name_asc", "name_desc"}:
        raise HTTPException(status_code=400, detail="sort_by 非法")

    if source_type == "annotations":
        approved_set = _approved_annotation_point_set(db) if approved_only else None
        rows = (
            db.query(AnnotationRecord)
            .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
            .order_by(AnnotationRecord.updated_at.desc())
            .all()
        )

        inference_query = db.query(InferenceResult)
        if method:
            inference_query = inference_query.filter(InferenceResult.method == method)
        if min_score is not None:
            inference_query = inference_query.filter(InferenceResult.score_avg >= min_score)
        if max_score is not None:
            inference_query = inference_query.filter(InferenceResult.score_avg <= max_score)
        inference_rows = inference_query.order_by(InferenceResult.created_at.desc()).all()

        score_map: Dict[str, InferenceResult] = {}
        for row in inference_rows:
            key = _normalize_point_name(row.point_name or "")
            if not key:
                continue
            # Keep latest row per point.
            if key not in score_map:
                score_map[key] = row

        seen = set()
        entries = []
        for row in rows:
            point_name = _normalize_point_name(row.source_id or row.filename or "")
            if not point_name or point_name in seen:
                continue
            if approved_set is not None and point_name not in approved_set:
                continue
            if keyword and keyword not in point_name.lower():
                continue

            row_source_kind = (row.source_kind or "human").strip().lower()
            if source_kind and row_source_kind != source_kind:
                continue

            score_row = score_map.get(point_name)
            # If score filters or method are applied, only keep rows with matched score metadata.
            if (method or min_score is not None or max_score is not None) and score_row is None:
                continue

            seen.add(point_name)
            score_value = float(score_row.score_avg or 0.0) if score_row is not None and score_row.score_avg is not None else None
            entries.append(
                {
                    "point_name": point_name,
                    "source_kind": row_source_kind,
                    "score_value": score_value,
                    "score_method": (score_row.method or "unknown") if score_row is not None else None,
                    "updated_ts": row.updated_at.timestamp() if row.updated_at else 0.0,
                }
            )

        if sort_by == "score_desc":
            entries.sort(key=lambda e: (1 if e["score_value"] is None else 0, -(e["score_value"] or 0)))
        elif sort_by == "score_asc":
            entries.sort(key=lambda e: (1e18 if e["score_value"] is None else e["score_value"]))
        elif sort_by == "updated_asc":
            entries.sort(key=lambda e: e["updated_ts"])
        elif sort_by == "name_asc":
            entries.sort(key=lambda e: e["point_name"])
        elif sort_by == "name_desc":
            entries.sort(key=lambda e: e["point_name"], reverse=True)
        else:  # updated_desc
            entries.sort(key=lambda e: e["updated_ts"], reverse=True)

        choices = []
        for e in entries[:limit]:
            kind_tag = "[AUTO]" if e["source_kind"] == "auto" else "[HUMAN]"
            suffix = f"{kind_tag}"
            if e["score_value"] is not None:
                suffix += f" | Score: {e['score_value']:.2f} ({e['score_method']})"
            choices.append({"label": f"{e['point_name']} | {suffix}", "value": e["point_name"], "source_kind": e["source_kind"]})

        return ApiResponse(success=True, data={"choices": choices, "count": len(choices)}, message="ok")

    if source_type == "training":
        fam = (model_family or "").strip().lower() or None
        if fam not in {None, "chatts", "qwen", "all"}:
            raise HTTPException(status_code=400, detail="model_family 仅支持 chatts / qwen / all")

        approved_set = _approved_annotation_point_set(db) if approved_only else None

        roots = []
        if fam in {None, "all", "chatts"}:
            roots.append(("chatts", Path(settings.DATA_TRAINING_CHATTS_DIR)))
        if fam in {None, "all", "qwen"}:
            roots.append(("qwen", Path(settings.DATA_TRAINING_QWEN_DIR)))

        point_entries: Dict[str, Dict] = {}
        for model_tag, root in roots:
            if not root.exists():
                continue
            files = sorted(
                [
                    p for p in root.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in {".json", ".jsonl"}
                    and not p.name.startswith(".")
                    and not p.name.startswith("_")
                    and not p.name.startswith("dataset_info")
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for fp in files:
                try:
                    updated_ts = fp.stat().st_mtime
                except Exception:
                    updated_ts = 0.0
                for row in _iter_training_records(fp):
                    point_name = _extract_training_point_name(row)
                    if not point_name:
                        continue
                    if keyword and keyword not in point_name.lower():
                        continue
                    if approved_set is not None and point_name not in approved_set:
                        continue
                    if point_name in point_entries:
                        continue
                    point_entries[point_name] = {
                        "point_name": point_name,
                        "model_family": model_tag,
                        "source_file": fp.name,
                        "updated_ts": updated_ts,
                    }

        entries = list(point_entries.values())
        if sort_by == "updated_asc":
            entries.sort(key=lambda e: e["updated_ts"])
        elif sort_by == "name_asc":
            entries.sort(key=lambda e: e["point_name"])
        elif sort_by == "name_desc":
            entries.sort(key=lambda e: e["point_name"], reverse=True)
        else:
            entries.sort(key=lambda e: e["updated_ts"], reverse=True)

        choices = []
        for e in entries[:limit]:
            label = f"{e['point_name']} | [TRAINING:{e['model_family'].upper()}] | {e['source_file']}"
            choices.append(
                {
                    "label": label,
                    "value": e["point_name"],
                    "source_kind": "training",
                }
            )
        return ApiResponse(success=True, data={"choices": choices, "count": len(choices)}, message="ok")

    if source_type != "inference":
        raise HTTPException(status_code=400, detail="source_type 仅支持 annotations 或 inference 或 training")

    query = db.query(InferenceResult)
    if method:
        query = query.filter(InferenceResult.method == method)
    if min_score is not None:
        query = query.filter(InferenceResult.score_avg >= min_score)
    if max_score is not None:
        query = query.filter(InferenceResult.score_avg <= max_score)

    if sort_by == "score_asc":
        query = query.order_by(InferenceResult.score_avg.asc())
    elif sort_by == "updated_desc":
        query = query.order_by(InferenceResult.created_at.desc())
    elif sort_by == "updated_asc":
        query = query.order_by(InferenceResult.created_at.asc())
    elif sort_by == "name_asc":
        query = query.order_by(InferenceResult.point_name.asc())
    elif sort_by == "name_desc":
        query = query.order_by(InferenceResult.point_name.desc())
    else:
        query = query.order_by(InferenceResult.score_avg.desc())

    rows = query.limit(limit).all()
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
        label = f"{point_name} | [INFERENCE] | Score: {(r.score_avg or 0):.2f} ({r.method or 'unknown'})"
        choices.append({"label": label, "value": point_name, "source_kind": "inference"})

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

    adapter = DataProcessingAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        selected = 0
        ann_rows = (
            db.query(AnnotationRecord)
            .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
            .order_by(AnnotationRecord.updated_at.desc())
            .all()
        )

        for row in ann_rows:
            normalized = _normalize_point_name(row.source_id or row.filename)
            if normalized not in point_set:
                continue
            if approved_set is not None and normalized not in approved_set:
                continue

            payload = record_to_payload(row)
            out_name = f"{_normalize_point_name(row.filename or row.source_id)}.json"
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
