"""
Annotation SSOT helpers.

This module centralizes the DB-first read/write path for annotations.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.db.database import AnnotationRecord, AnnotationSegment


def normalize_point_name(name: str) -> str:
    text = str(name or "").strip()
    text = Path(text).name
    lower = text.lower()
    for ext in (".csv", ".json", ".xls", ".xlsx"):
        if lower.endswith(ext):
            text = text[: -len(ext)]
            lower = text.lower()
    if text.startswith("annotations_"):
        text = text.replace("annotations_", "", 1)
    text = text.replace("数据集", "")
    return text.strip()


def _loads_json(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _safe_int(val: Any) -> Optional[int]:
    try:
        if val is None:
            return None
        return int(val)
    except Exception:
        return None


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def infer_source_kind(payload: Dict[str, Any], annotations: List[Dict[str, Any]]) -> str:
    explicit = str(payload.get("source_kind") or "").strip().lower()
    if explicit in {"auto", "human"}:
        return explicit

    saw_auto = False
    saw_human = False
    for ann in annotations:
        source = str(ann.get("source") or "").strip().lower()
        ann_id = str(ann.get("id") or "").strip().lower()
        if source in {"auto", "inference"} or ann_id.startswith("auto_") or ann_id.startswith("infer_"):
            saw_auto = True
        if source in {"human", "manual"}:
            saw_human = True
        if ann.get("expert_output") not in (None, "") or ann.get("expertOutput") not in (None, ""):
            saw_human = True

    if saw_human:
        return "human"
    if saw_auto:
        return "auto"
    return "human"


def _segment_rows(user_id: str, source_id: str, annotation_id: str, annotations: List[Dict[str, Any]]) -> List[AnnotationSegment]:
    rows: List[AnnotationSegment] = []
    for ann_idx, ann in enumerate(annotations):
        label = ann.get("label")
        label_id = None
        label_text = None
        if isinstance(label, dict):
            label_id = str(label.get("id") or "") or None
            label_text = str(label.get("text") or "") or None
        elif label is not None:
            label_text = str(label)

        segments = ann.get("segments") or []
        if not isinstance(segments, list):
            continue
        for seg_idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            start = _safe_int(seg.get("start"))
            end = _safe_int(seg.get("end"))
            count = _safe_int(seg.get("count"))
            if count is None and start is not None and end is not None and end >= start:
                count = end - start + 1
            rows.append(
                AnnotationSegment(
                    annotation_id=annotation_id,
                    user_id=user_id,
                    source_id=source_id,
                    ann_index=ann_idx,
                    seg_index=seg_idx,
                    start=start,
                    end=end,
                    count=count,
                    label_id=label_id,
                    label_text=label_text,
                    score=_safe_float(seg.get("score")),
                    review_status="pending",
                )
            )
    return rows


def upsert_annotation(
    db: Session,
    user_id: str,
    filename: str,
    payload: Dict[str, Any],
) -> AnnotationRecord:
    payload = payload or {}
    filename_in = str(payload.get("filename") or filename or "").strip()
    if not filename_in:
        filename_in = str(filename or "").strip()
    source_id = normalize_point_name(filename_in)
    annotations = payload.get("annotations") or []
    if not isinstance(annotations, list):
        annotations = []
    overall_attribute = payload.get("overall_attribute")
    if overall_attribute is None:
        overall_attribute = payload.get("overall_attributes") or {}
    if not isinstance(overall_attribute, dict):
        overall_attribute = {}

    source_kind = infer_source_kind(payload, annotations)
    is_human_edited = bool(payload.get("is_human_edited")) or source_kind == "human"
    segment_count = sum(len((ann or {}).get("segments") or []) for ann in annotations if isinstance(ann, dict))
    annotation_count = len(annotations)

    record = (
        db.query(AnnotationRecord)
        .filter(AnnotationRecord.user_id == user_id, AnnotationRecord.source_id == source_id)
        .first()
    )
    if record is None:
        record = AnnotationRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            source_id=source_id,
            filename=filename_in,
            created_at=datetime.utcnow(),
        )
        db.add(record)

    meta = {
        "export_time": payload.get("export_time"),
        "last_updated": payload.get("last_updated"),
        "source_kind": source_kind,
    }

    record.filename = filename_in
    record.source_kind = source_kind
    record.source_inference_id = payload.get("source_inference_id")
    record.method = payload.get("method")
    record.status = str(payload.get("status") or "draft")
    record.is_human_edited = is_human_edited
    record.annotation_count = annotation_count
    record.segment_count = segment_count
    record.overall_attribute_json = json.dumps(overall_attribute, ensure_ascii=False)
    record.annotations_json = json.dumps(annotations, ensure_ascii=False)
    record.meta = json.dumps(meta, ensure_ascii=False)
    record.updated_at = datetime.utcnow()
    db.flush()

    db.query(AnnotationSegment).filter(AnnotationSegment.annotation_id == record.id).delete()
    for seg_row in _segment_rows(user_id, source_id, record.id, annotations):
        db.add(seg_row)
    db.flush()
    return record


def get_annotation_record(db: Session, user_id: str, filename: str) -> Optional[AnnotationRecord]:
    normalized = normalize_point_name(filename)
    return (
        db.query(AnnotationRecord)
        .filter(
            AnnotationRecord.user_id == user_id,
            or_(AnnotationRecord.source_id == normalized, AnnotationRecord.filename == filename),
        )
        .order_by(AnnotationRecord.updated_at.desc())
        .first()
    )


def record_to_payload(record: AnnotationRecord, fallback_filename: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "filename": record.filename or fallback_filename or record.source_id,
        "annotations": _loads_json(record.annotations_json, []),
        "overall_attribute": _loads_json(record.overall_attribute_json, {}),
    }
    meta = _loads_json(record.meta, {})
    if meta.get("export_time"):
        payload["export_time"] = meta.get("export_time")
    if meta.get("last_updated"):
        payload["last_updated"] = meta.get("last_updated")
    payload["source_kind"] = record.source_kind
    payload["is_human_edited"] = bool(record.is_human_edited)
    return payload


def list_annotation_records(
    db: Session,
    user_id: str,
    keyword: Optional[str] = None,
    limit: int = 200,
) -> List[AnnotationRecord]:
    query = db.query(AnnotationRecord).filter(AnnotationRecord.user_id == user_id)
    if keyword:
        key = f"%{keyword.strip()}%"
        query = query.filter(
            or_(
                AnnotationRecord.source_id.ilike(key),
                AnnotationRecord.filename.ilike(key),
            )
        )
    return query.order_by(AnnotationRecord.updated_at.desc()).limit(limit).all()

