"""
Point-first API.

This module exposes a point-centric read model while staying compatible with
existing source_id/filename/point_name based records.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from configs.settings import settings
from src.db.database import (
    AnnotationRecord,
    InferenceResult,
    ReviewQueue,
    get_db,
)
from src.models.schemas import ApiResponse
from src.utils.annotation_store import canonical_point_id

router = APIRouter()


def _event_time(row) -> float:
    ts = getattr(row, "updated_at", None) or getattr(row, "created_at", None)
    if ts is None:
        return 0.0
    return ts.timestamp()


def _point_label(point_id: str, filename: Optional[str]) -> str:
    if filename:
        return f"{point_id} ({filename})"
    return point_id


@router.get("", response_model=ApiResponse)
async def list_points(
    keyword: Optional[str] = Query(None, description="按 point_id 关键字过滤"),
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    key = (keyword or "").strip().lower()
    records: Dict[str, Dict] = {}

    ann_rows = (
        db.query(AnnotationRecord)
        .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
        .order_by(AnnotationRecord.updated_at.desc())
        .all()
    )
    for row in ann_rows:
        point_id = canonical_point_id(getattr(row, "point_id", None), row.source_id, row.filename)
        if not point_id:
            continue
        if key and key not in point_id.lower():
            continue
        if point_id in records:
            continue
        records[point_id] = {
            "point_id": point_id,
            "label": _point_label(point_id, row.filename),
            "source_kind": row.source_kind or "human",
            "annotation_count": int(row.annotation_count or 0),
            "segment_count": int(row.segment_count or 0),
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "updated_ts": _event_time(row),
            "score_avg": None,
            "score_method": None,
            "review_status": None,
        }

    inf_rows = db.query(InferenceResult).order_by(InferenceResult.created_at.desc()).all()
    for row in inf_rows:
        point_id = canonical_point_id(getattr(row, "point_id", None), row.point_name)
        if not point_id:
            continue
        if key and key not in point_id.lower():
            continue
        entry = records.setdefault(
            point_id,
            {
                "point_id": point_id,
                "label": point_id,
                "source_kind": "inference",
                "annotation_count": 0,
                "segment_count": 0,
                "updated_at": row.created_at.isoformat() if row.created_at else None,
                "updated_ts": _event_time(row),
                "score_avg": None,
                "score_method": None,
                "review_status": None,
            },
        )
        if entry["score_avg"] is None:
            entry["score_avg"] = float(row.score_avg) if row.score_avg is not None else None
            entry["score_method"] = row.method or "unknown"
        if _event_time(row) > entry["updated_ts"]:
            entry["updated_ts"] = _event_time(row)
            entry["updated_at"] = row.created_at.isoformat() if row.created_at else None

    review_rows = db.query(ReviewQueue).order_by(ReviewQueue.updated_at.desc()).all()
    for row in review_rows:
        point_id = canonical_point_id(getattr(row, "point_id", None), row.point_name, row.source_id)
        if not point_id:
            continue
        if point_id not in records:
            continue
        if records[point_id]["review_status"] is None:
            records[point_id]["review_status"] = row.status or "pending"
        if _event_time(row) > records[point_id]["updated_ts"]:
            records[point_id]["updated_ts"] = _event_time(row)
            records[point_id]["updated_at"] = row.updated_at.isoformat() if row.updated_at else None

    items = list(records.values())
    items.sort(key=lambda x: x["updated_ts"], reverse=True)
    for item in items:
        item.pop("updated_ts", None)

    return ApiResponse(
        success=True,
        data={"points": items[:limit], "count": min(len(items), limit)},
        message="ok",
    )


@router.get("/{point_id}", response_model=ApiResponse)
async def get_point(point_id: str, db: Session = Depends(get_db)):
    target = canonical_point_id(point_id)
    if not target:
        raise HTTPException(status_code=400, detail="无效 point_id")

    ann_rows = (
        db.query(AnnotationRecord)
        .filter(
            AnnotationRecord.user_id == settings.DEFAULT_USER,
            or_(
                AnnotationRecord.point_id == target,
                AnnotationRecord.source_id == target,
                AnnotationRecord.filename == target,
                AnnotationRecord.filename == f"{target}.csv",
            ),
        )
        .order_by(AnnotationRecord.updated_at.desc())
        .all()
    )

    inf_rows = (
        db.query(InferenceResult)
        .filter(
            or_(
                InferenceResult.point_id == target,
                InferenceResult.point_name == target,
            )
        )
        .order_by(InferenceResult.created_at.desc())
        .all()
    )

    review_rows = (
        db.query(ReviewQueue)
        .filter(
            or_(
                ReviewQueue.point_id == target,
                ReviewQueue.point_name == target,
                ReviewQueue.source_id == target,
            )
        )
        .order_by(ReviewQueue.updated_at.desc())
        .all()
    )

    if not ann_rows and not inf_rows and not review_rows:
        raise HTTPException(status_code=404, detail="点位不存在")

    latest_ann = ann_rows[0] if ann_rows else None
    latest_inf = inf_rows[0] if inf_rows else None
    latest_review = review_rows[0] if review_rows else None

    return ApiResponse(
        success=True,
        data={
            "point_id": target,
            "annotation_summary": {
                "count": len(ann_rows),
                "latest_source_kind": (latest_ann.source_kind if latest_ann else None),
                "latest_annotation_count": int(latest_ann.annotation_count or 0) if latest_ann else 0,
                "latest_segment_count": int(latest_ann.segment_count or 0) if latest_ann else 0,
                "latest_updated_at": latest_ann.updated_at.isoformat() if latest_ann and latest_ann.updated_at else None,
            },
            "inference_summary": {
                "count": len(inf_rows),
                "latest_method": latest_inf.method if latest_inf else None,
                "latest_score_avg": float(latest_inf.score_avg) if latest_inf and latest_inf.score_avg is not None else None,
                "latest_created_at": latest_inf.created_at.isoformat() if latest_inf and latest_inf.created_at else None,
            },
            "review_summary": {
                "count": len(review_rows),
                "latest_status": latest_review.status if latest_review else None,
                "latest_updated_at": latest_review.updated_at.isoformat() if latest_review and latest_review.updated_at else None,
            },
        },
        message="ok",
    )


@router.get("/{point_id}/timeline", response_model=ApiResponse)
async def get_point_timeline(
    point_id: str,
    limit: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    target = canonical_point_id(point_id)
    if not target:
        raise HTTPException(status_code=400, detail="无效 point_id")

    events: List[Dict] = []

    ann_rows = (
        db.query(AnnotationRecord)
        .filter(
            AnnotationRecord.user_id == settings.DEFAULT_USER,
            or_(AnnotationRecord.point_id == target, AnnotationRecord.source_id == target),
        )
        .all()
    )
    for row in ann_rows:
        ts = row.updated_at or row.created_at
        events.append(
            {
                "type": "annotation",
                "time": ts.isoformat() if ts else None,
                "ts": ts.timestamp() if ts else 0.0,
                "data": {
                    "source_kind": row.source_kind,
                    "annotation_count": int(row.annotation_count or 0),
                    "segment_count": int(row.segment_count or 0),
                },
            }
        )

    inf_rows = (
        db.query(InferenceResult)
        .filter(or_(InferenceResult.point_id == target, InferenceResult.point_name == target))
        .all()
    )
    for row in inf_rows:
        ts = row.created_at
        events.append(
            {
                "type": "inference",
                "time": ts.isoformat() if ts else None,
                "ts": ts.timestamp() if ts else 0.0,
                "data": {
                    "method": row.method,
                    "score_avg": float(row.score_avg) if row.score_avg is not None else None,
                    "score_max": float(row.score_max) if row.score_max is not None else None,
                    "segment_count": int(row.segment_count or 0),
                },
            }
        )

    review_rows = (
        db.query(ReviewQueue)
        .filter(or_(ReviewQueue.point_id == target, ReviewQueue.point_name == target, ReviewQueue.source_id == target))
        .all()
    )
    for row in review_rows:
        ts = row.updated_at or row.created_at
        events.append(
            {
                "type": "review",
                "time": ts.isoformat() if ts else None,
                "ts": ts.timestamp() if ts else 0.0,
                "data": {
                    "status": row.status,
                    "strategy": row.strategy,
                    "score": float(row.score) if row.score is not None else None,
                },
            }
        )

    events.sort(key=lambda e: e["ts"], reverse=True)
    for event in events:
        event.pop("ts", None)

    return ApiResponse(
        success=True,
        data={"point_id": target, "events": events[:limit], "count": min(len(events), limit)},
        message="ok",
    )
