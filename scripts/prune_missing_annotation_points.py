#!/usr/bin/env python3
"""
清理“源 CSV 缺失”的标注点位（DB + 本地标注 JSON）。

默认是 dry-run，只输出待清理项；加 `--apply` 才会真正删除。

示例:
  python scripts/prune_missing_annotation_points.py
  python scripts/prune_missing_annotation_points.py --point-id LHS2_20250322_20250325_H2S
  python scripts/prune_missing_annotation_points.py --point-id LHS2_20250322_20250325_H2S --apply
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import and_, or_

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from src.db.database import (
    AnnotationRecord,
    AnnotationSegment,
    ReviewQueue,
    SessionLocal,
)


@dataclass
class MissingRecord:
    row: AnnotationRecord
    candidate_csv_names: list[str]


def _candidate_csv_names(row: AnnotationRecord) -> list[str]:
    names: set[str] = set()
    for raw in (row.filename, row.source_id, row.point_id):
        text = str(raw or "").strip()
        if not text:
            continue
        base = Path(text).name
        if base.endswith(".csv"):
            names.add(base)
        else:
            names.add(f"{base}.csv")
    return sorted(names)


def _csv_exists(csv_name: str, roots: Iterable[Path]) -> bool:
    for root in roots:
        if (root / csv_name).exists():
            return True
    return False


def _find_missing_records(point_id: str | None = None) -> tuple[list[MissingRecord], list[Path]]:
    csv_roots = [Path(settings.DATA_DOWNSAMPLED_DIR), Path(settings.DATA_RAW_DIR)]

    db = SessionLocal()
    try:
        query = db.query(AnnotationRecord)
        if point_id:
            point_text = point_id.strip()
            filename_text = point_text if point_text.endswith(".csv") else f"{point_text}.csv"
            query = query.filter(
                or_(
                    AnnotationRecord.point_id == point_text,
                    AnnotationRecord.source_id == point_text,
                    AnnotationRecord.filename == point_text,
                    AnnotationRecord.filename == filename_text,
                )
            )

        rows = query.order_by(AnnotationRecord.updated_at.desc()).all()
        missing: list[MissingRecord] = []
        for row in rows:
            candidates = _candidate_csv_names(row)
            exists = any(_csv_exists(name, csv_roots) for name in candidates)
            if not exists:
                missing.append(MissingRecord(row=row, candidate_csv_names=candidates))
        return missing, csv_roots
    finally:
        db.close()


def _remove_json_files(missing: list[MissingRecord]) -> int:
    removed = 0
    ann_root = Path(settings.ANNOTATIONS_ROOT)
    for item in missing:
        row = item.row
        user = str(row.user_id or settings.DEFAULT_USER)
        filename = Path(str(row.filename or "")).name
        stem = Path(filename).stem
        source_id = str(row.source_id or "").strip()
        point_id = str(row.point_id or "").strip()

        candidates = [
            ann_root / user / f"{filename}.json",
            ann_root / user / f"{stem}.json",
        ]
        if source_id:
            candidates.append(ann_root / user / f"{source_id}.json")
            candidates.append(ann_root / user / f"{source_id}.csv.json")
        if point_id:
            candidates.append(ann_root / user / f"{point_id}.json")
            candidates.append(ann_root / user / f"{point_id}.csv.json")

        unique_paths = []
        seen = set()
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            unique_paths.append(p)

        for p in unique_paths:
            if p.exists():
                p.unlink()
                removed += 1
                print(f"[delete:file] {p}")
    return removed


def _apply_cleanup(missing: list[MissingRecord], remove_json: bool) -> None:
    if not missing:
        print("没有需要清理的记录。")
        return

    annotation_ids = {m.row.id for m in missing}
    source_ids = {str(m.row.source_id) for m in missing if m.row.source_id}
    point_ids = {str(m.row.point_id) for m in missing if m.row.point_id}

    db = SessionLocal()
    try:
        seg_conditions = [AnnotationSegment.annotation_id.in_(annotation_ids)]
        if source_ids:
            seg_conditions.append(AnnotationSegment.source_id.in_(source_ids))
        if point_ids:
            seg_conditions.append(AnnotationSegment.point_id.in_(point_ids))
        seg_deleted = db.query(AnnotationSegment).filter(or_(*seg_conditions)).delete(synchronize_session=False)

        review_conditions = []
        if source_ids:
            review_conditions.append(ReviewQueue.source_id.in_(source_ids))
        if point_ids:
            review_conditions.append(ReviewQueue.point_id.in_(point_ids))
            review_conditions.append(ReviewQueue.point_name.in_(point_ids))
        review_deleted = 0
        if review_conditions:
            review_deleted = (
                db.query(ReviewQueue)
                .filter(and_(ReviewQueue.source_type == "annotation", or_(*review_conditions)))
                .delete(synchronize_session=False)
            )

        ann_deleted = (
            db.query(AnnotationRecord)
            .filter(AnnotationRecord.id.in_(annotation_ids))
            .delete(synchronize_session=False)
        )

        db.commit()

        print(f"[delete:db] annotation_records={ann_deleted}")
        print(f"[delete:db] annotation_segments={seg_deleted}")
        print(f"[delete:db] review_queue(annotation)={review_deleted}")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    if remove_json:
        json_deleted = _remove_json_files(missing)
        print(f"[delete:file] annotation_json={json_deleted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="清理源 CSV 缺失的标注点位（默认 dry-run）")
    parser.add_argument("--point-id", help="仅处理指定点位（point_id/source_id/filename）")
    parser.add_argument("--apply", action="store_true", help="执行删除（默认仅预览）")
    parser.add_argument("--keep-json", action="store_true", help="执行删除时保留本地标注 JSON 文件")
    args = parser.parse_args()

    missing, csv_roots = _find_missing_records(point_id=args.point_id)

    print("CSV 检查目录:")
    for root in csv_roots:
        print(f"  - {root}")

    print(f"\n待清理记录数: {len(missing)}")
    for idx, item in enumerate(missing, start=1):
        row = item.row
        print(
            f"{idx:>2}. id={row.id} user={row.user_id} point_id={row.point_id} "
            f"source_id={row.source_id} filename={row.filename}"
        )
        print(f"    csv_candidates={item.candidate_csv_names}")

    if not args.apply:
        print("\n当前为 dry-run。确认无误后加 --apply 执行删除。")
        return

    _apply_cleanup(missing, remove_json=not args.keep_json)
    print("\n清理完成。")


if __name__ == "__main__":
    main()
