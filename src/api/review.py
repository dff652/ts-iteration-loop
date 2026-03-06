"""
审核队列 API
将审核队列读写能力从 Gradio 内部逻辑下沉到正式接口层。
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import or_
from sqlalchemy.orm import Session

from configs.settings import settings
from src.db.database import AnnotationRecord, InferenceResult, ReviewQueue, get_db
from src.models.schemas import ApiResponse
from src.utils.annotation_store import canonical_point_id
from src.utils.time_utils import utc_now_naive

router = APIRouter()


class ReviewBatchUpdateRequest(BaseModel):
    point_ids: List[str] = Field(default_factory=list)
    status: str = Field(..., min_length=1)
    reviewer: Optional[str] = None

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"pending", "approved", "needs_fix"}:
            raise ValueError("status 仅支持 pending/approved/needs_fix")
        return normalized

    @field_validator("point_ids")
    @classmethod
    def _validate_point_ids(cls, value: List[str]) -> List[str]:
        items: List[str] = []
        for raw in value:
            normalized = canonical_point_id(raw)
            if normalized and normalized not in items:
                items.append(normalized)
        if not items:
            raise ValueError("point_ids 不能为空")
        return items


def _norm_review_status(raw: Optional[str]) -> str:
    text = str(raw or "").strip().lower()
    if text == "rejected":
        return "needs_fix"
    if text in {"pending", "approved", "needs_fix"}:
        return text
    if not text:
        return "unreviewed"
    return text


def _event_ts(value) -> float:
    if value is None:
        return 0.0
    return value.timestamp()


@router.get("/queue", response_model=ApiResponse)
async def list_review_queue(
    status: Optional[str] = Query(None, description="pending/approved/needs_fix/unreviewed"),
    method: Optional[str] = Query(None, description="按推理方法过滤"),
    annotation_kind: str = Query("all", description="all/auto/human"),
    keyword: Optional[str] = Query(None, description="point_id 关键字"),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0, le=100000),
    db: Session = Depends(get_db),
):
    normalized_status = str(status or "").strip().lower() or None
    if normalized_status not in {None, "pending", "approved", "needs_fix", "unreviewed"}:
        raise HTTPException(status_code=400, detail="status 非法")

    normalized_kind = str(annotation_kind or "all").strip().lower()
    if normalized_kind not in {"all", "auto", "human"}:
        raise HTTPException(status_code=400, detail="annotation_kind 非法")

    normalized_method = str(method or "").strip().lower() or None
    normalized_keyword = str(keyword or "").strip().lower()

    ann_rows = (
        db.query(AnnotationRecord)
        .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
        .order_by(AnnotationRecord.updated_at.desc())
        .all()
    )

    point_map: Dict[str, Dict] = {}
    for row in ann_rows:
        point_id = canonical_point_id(getattr(row, "point_id", None), row.source_id, row.filename)
        if not point_id:
            continue
        if point_id in point_map:
            continue
        source_kind = str(row.source_kind or "human").strip().lower()
        if normalized_kind in {"auto", "human"} and source_kind != normalized_kind:
            continue
        if normalized_keyword and normalized_keyword not in point_id.lower():
            continue
        point_map[point_id] = {
            "point_id": point_id,
            "source_kind": source_kind,
            "annotation_count": int(row.annotation_count or 0),
            "segment_count": int(row.segment_count or 0),
            "annotation_updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "annotation_updated_ts": _event_ts(row.updated_at),
            "method": None,
            "score": None,
            "review_status": "unreviewed",
            "reviewer": None,
            "review_updated_at": None,
            "review_updated_ts": 0.0,
        }

    inf_query = db.query(InferenceResult).order_by(InferenceResult.created_at.desc())
    if normalized_method:
        inf_query = inf_query.filter(InferenceResult.method == normalized_method)
    inf_rows = inf_query.all()
    for row in inf_rows:
        point_id = canonical_point_id(getattr(row, "point_id", None), row.point_name)
        if not point_id or point_id not in point_map:
            continue
        entry = point_map[point_id]
        if entry["method"] is None:
            entry["method"] = row.method
            entry["score"] = float(row.score_avg) if row.score_avg is not None else None

    review_rows = (
        db.query(ReviewQueue)
        .filter(ReviewQueue.source_type == "annotation")
        .order_by(ReviewQueue.updated_at.desc())
        .all()
    )
    for row in review_rows:
        point_id = canonical_point_id(getattr(row, "point_id", None), row.source_id, row.point_name)
        if not point_id or point_id not in point_map:
            continue
        entry = point_map[point_id]
        if entry["review_updated_ts"] > 0:
            continue
        entry["review_status"] = _norm_review_status(row.status)
        entry["reviewer"] = str(row.reviewer or "").strip() or None
        entry["review_updated_at"] = row.updated_at.isoformat() if row.updated_at else None
        entry["review_updated_ts"] = _event_ts(row.updated_at)

    items: List[Dict] = []
    stats = {"total": 0, "pending": 0, "approved": 0, "needs_fix": 0, "unreviewed": 0}
    for point_id, entry in point_map.items():
        current_status = str(entry["review_status"] or "unreviewed")
        if normalized_status and current_status != normalized_status:
            continue
        if normalized_method and str(entry["method"] or "").strip().lower() != normalized_method:
            continue
        latest_ts = max(float(entry["annotation_updated_ts"] or 0.0), float(entry["review_updated_ts"] or 0.0))
        items.append(
            {
                "point_id": point_id,
                "source_kind": entry["source_kind"],
                "annotation_count": entry["annotation_count"],
                "segment_count": entry["segment_count"],
                "method": entry["method"],
                "score": entry["score"],
                "status": current_status,
                "reviewer": entry["reviewer"],
                "updated_at": entry["review_updated_at"] or entry["annotation_updated_at"],
                "updated_ts": latest_ts,
            }
        )
        stats["total"] += 1
        if current_status in stats:
            stats[current_status] += 1

    items.sort(key=lambda row: row.get("updated_ts", 0.0), reverse=True)
    total = len(items)
    paged = items[offset : offset + limit]
    for row in paged:
        row.pop("updated_ts", None)

    return ApiResponse(
        success=True,
        data={"items": paged, "total": total, "stats": stats},
        message="ok",
    )


@router.post("/queue/batch-update", response_model=ApiResponse)
async def batch_update_review_status(request: ReviewBatchUpdateRequest, db: Session = Depends(get_db)):
    reviewer = str(request.reviewer or "").strip() or settings.DEFAULT_USER
    now = utc_now_naive()

    updated_rows = 0
    touched_ids: List[str] = []

    for point_id in request.point_ids:
        row = (
            db.query(ReviewQueue)
            .filter(
                ReviewQueue.source_type == "annotation",
                or_(
                    ReviewQueue.point_id == point_id,
                    ReviewQueue.source_id == point_id,
                ),
            )
            .order_by(ReviewQueue.updated_at.desc())
            .first()
        )

        method = None
        score = None
        if row is None:
            inf_row = (
                db.query(InferenceResult)
                .filter(
                    or_(
                        InferenceResult.point_id == point_id,
                        InferenceResult.point_name == point_id,
                    )
                )
                .order_by(InferenceResult.created_at.desc())
                .first()
            )
            if inf_row is not None:
                method = inf_row.method
                score = float(inf_row.score_avg) if inf_row.score_avg is not None else None

            row = ReviewQueue(
                id=str(uuid.uuid4()),
                source_type="annotation",
                source_id=point_id,
                point_id=point_id,
                method=method,
                model=None,
                point_name=point_id,
                score=score,
                strategy="manual_review",
                status=request.status,
                reviewer=reviewer,
                created_at=now,
                updated_at=now,
            )
            db.add(row)
        else:
            row.status = request.status
            row.reviewer = reviewer
            row.updated_at = now

        touched_ids.append(str(row.id))
        updated_rows += 1

    db.commit()

    return ApiResponse(
        success=True,
        data={"updated": updated_rows, "status": request.status, "reviewer": reviewer, "ids": touched_ids},
        message=f"已更新 {updated_rows} 条审核状态",
    )
