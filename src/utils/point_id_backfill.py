"""
Helpers for backfilling point_id fields in scheme-1 mode.

Scheme-1 rule: point_id = normalize_point_name(...)
"""

from __future__ import annotations

from typing import Dict

from sqlalchemy.orm import Session

from src.db.database import (
    AnnotationRecord,
    AnnotationSegment,
    DatasetItem,
    InferenceResult,
    ReviewQueue,
)
from src.utils.annotation_store import canonical_point_id


def backfill_point_ids(db: Session) -> Dict[str, int]:
    stats = {
        "annotation_records": 0,
        "annotation_segments": 0,
        "inference_results": 0,
        "review_queue": 0,
        "dataset_items": 0,
        "total": 0,
    }

    ann_rows = db.query(AnnotationRecord).all()
    for row in ann_rows:
        new_id = canonical_point_id(getattr(row, "point_id", None), row.source_id, row.filename)
        if new_id and getattr(row, "point_id", None) != new_id:
            row.point_id = new_id
            stats["annotation_records"] += 1

    seg_rows = db.query(AnnotationSegment).all()
    for row in seg_rows:
        new_id = canonical_point_id(getattr(row, "point_id", None), row.source_id)
        if new_id and getattr(row, "point_id", None) != new_id:
            row.point_id = new_id
            stats["annotation_segments"] += 1

    inf_rows = db.query(InferenceResult).all()
    for row in inf_rows:
        new_id = canonical_point_id(getattr(row, "point_id", None), row.point_name)
        if new_id and getattr(row, "point_id", None) != new_id:
            row.point_id = new_id
            stats["inference_results"] += 1

    review_rows = db.query(ReviewQueue).all()
    for row in review_rows:
        new_id = canonical_point_id(getattr(row, "point_id", None), row.point_name, row.source_id)
        if new_id and getattr(row, "point_id", None) != new_id:
            row.point_id = new_id
            stats["review_queue"] += 1

    dataset_rows = db.query(DatasetItem).all()
    for row in dataset_rows:
        new_id = canonical_point_id(getattr(row, "point_id", None), row.point_name)
        if new_id and getattr(row, "point_id", None) != new_id:
            row.point_id = new_id
            stats["dataset_items"] += 1

    db.commit()
    stats["total"] = sum(v for k, v in stats.items() if k != "total")
    return stats
