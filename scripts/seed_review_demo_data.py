#!/usr/bin/env python3
"""Seed demo data for review sampling visualization.

Creates a small, reproducible set of:
- downsampled CSV files
- annotation SSOT records (auto/human)
- inference score rows
- review queue rows (approved/pending/rejected)

Use `--reset` to clean previous demo rows by prefix before seeding.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import settings
from src.db.database import (
    AnnotationRecord,
    AnnotationSegment,
    InferenceResult,
    ReviewQueue,
    SessionLocal,
    init_db,
)
from src.utils.annotation_store import normalize_point_name, upsert_annotation


def _build_series(length: int, seed: int, segs: List[Tuple[int, int]]) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(length):
        baseline = math.sin(i / 12.0) * 0.8 + math.cos(i / 25.0) * 0.35
        noise = rng.uniform(-0.08, 0.08)
        value = baseline + noise
        mask = 0
        for s, e in segs:
            if s <= i <= e:
                mask = 1
                value += 1.6 + rng.uniform(-0.2, 0.25)
                break
        rows.append({"timestamp": i, "value": round(value, 6), "global_mask": mask})
    return pd.DataFrame(rows)


def _annotation_payload(point_name: str, source_kind: str, segs: List[Tuple[int, int]], idx: int) -> Dict:
    ann_id_prefix = "auto" if source_kind == "auto" else "human"
    label_text = "AutoDetect" if source_kind == "auto" else "HumanRefine"
    label_id = "chatts_detected" if idx % 2 == 0 else "qwen_detected"

    segments = []
    for j, (s, e) in enumerate(segs):
        segments.append(
            {
                "start": int(s),
                "end": int(e),
                "count": int(e - s + 1),
                "score": round(0.45 + 0.1 * ((idx + j) % 5), 3),
            }
        )

    return {
        "filename": f"{point_name}.csv",
        "source_kind": source_kind,
        "is_human_edited": source_kind == "human",
        "method": "chatts" if idx % 2 == 0 else "qwen",
        "status": "ready",
        "overall_attribute": {
            "scenario": "demo_review_visual",
            "device": f"demo-device-{idx:02d}",
        },
        "annotations": [
            {
                "id": f"{ann_id_prefix}_{idx:02d}",
                "source": source_kind,
                "label": {
                    "id": label_id,
                    "text": label_text,
                    "color": "#ef4444" if source_kind == "auto" else "#2563eb",
                },
                "segments": segments,
                "local_change": {
                    "trend": "spike",
                    "confidence": "medium",
                    "desc": "demo segments",
                },
            }
        ],
        "export_time": datetime.utcnow().isoformat(),
    }


def _demo_specs(prefix: str) -> List[Dict]:
    specs = []
    statuses = ["approved", "approved", "pending", "approved", "rejected", "approved", "pending", "approved"]
    kinds = ["auto", "human", "auto", "human", "auto", "human", "auto", "human"]
    methods = ["chatts", "qwen", "chatts", "qwen", "chatts", "qwen", "chatts", "qwen"]
    scores = [0.93, 0.87, 0.66, 0.81, 0.49, 0.76, 0.58, 0.72]
    for i in range(8):
        specs.append(
            {
                "point": f"{prefix}{i + 1:02d}",
                "source_kind": kinds[i],
                "method": methods[i],
                "status": statuses[i],
                "score": scores[i],
                "seed": 100 + i,
                "segments": [(35 + i * 2, 48 + i * 2), (130 + i, 142 + i)],
            }
        )
    return specs


def _cleanup_demo_rows(db, prefix: str, user: str, delete_csv: bool, csv_dir: Path) -> Dict[str, int]:
    like_prefix = f"{prefix}%"

    ann_rows = db.query(AnnotationRecord).filter(AnnotationRecord.user_id == user, AnnotationRecord.source_id.like(like_prefix)).all()
    ann_ids = [r.id for r in ann_rows]
    if ann_ids:
        db.query(AnnotationSegment).filter(AnnotationSegment.annotation_id.in_(ann_ids)).delete(synchronize_session=False)
        db.query(AnnotationRecord).filter(AnnotationRecord.id.in_(ann_ids)).delete(synchronize_session=False)

    rq_deleted = db.query(ReviewQueue).filter(
        (ReviewQueue.source_id.like(like_prefix)) | (ReviewQueue.point_name.like(like_prefix))
    ).delete(synchronize_session=False)

    inf_deleted = db.query(InferenceResult).filter(InferenceResult.point_name.like(like_prefix)).delete(synchronize_session=False)

    csv_deleted = 0
    if delete_csv and csv_dir.exists():
        for p in csv_dir.glob(f"{prefix}*.csv"):
            try:
                p.unlink()
                csv_deleted += 1
            except Exception:
                pass

    return {
        "annotation_records": len(ann_rows),
        "review_queue": int(rq_deleted or 0),
        "inference_results": int(inf_deleted or 0),
        "csv_files": csv_deleted,
    }


def seed_demo_data(user: str, prefix: str, reset: bool, delete_csv_on_reset: bool) -> None:
    init_db()
    downsampled_dir = Path(settings.DATA_DOWNSAMPLED_DIR)
    downsampled_dir.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        if reset:
            deleted = _cleanup_demo_rows(db, prefix, user, delete_csv_on_reset, downsampled_dir)
            db.commit()
            print(f"[reset] removed: {deleted}")

        specs = _demo_specs(prefix)
        created_csv = 0
        upserted_ann = 0
        upserted_inf = 0
        upserted_rq = 0

        for i, spec in enumerate(specs):
            point = normalize_point_name(spec["point"])
            csv_path = downsampled_dir / f"{point}.csv"

            df = _build_series(length=240, seed=spec["seed"], segs=spec["segments"])
            df.to_csv(csv_path, index=False)
            created_csv += 1

            payload = _annotation_payload(point, spec["source_kind"], spec["segments"], i)
            upsert_annotation(db, user, f"{point}.csv", payload)
            upserted_ann += 1

            inf = (
                db.query(InferenceResult)
                .filter(InferenceResult.point_name == point, InferenceResult.method == spec["method"])
                .order_by(InferenceResult.created_at.desc())
                .first()
            )
            if inf is None:
                inf = InferenceResult(
                    id=str(uuid.uuid4()),
                    task_id=f"demo-task-{point}",
                    method=spec["method"],
                    model=f"demo-model-{spec['method']}",
                    point_name=point,
                    result_path=str(csv_path),
                    metrics_path="",
                    segments_path="",
                    score_avg=float(spec["score"]),
                    score_max=min(1.0, float(spec["score"]) + 0.06),
                    segment_count=len(spec["segments"]),
                    meta=json.dumps({"demo": True, "source_kind": spec["source_kind"]}, ensure_ascii=False),
                    created_at=datetime.utcnow(),
                )
                db.add(inf)
            else:
                inf.result_path = str(csv_path)
                inf.score_avg = float(spec["score"])
                inf.score_max = min(1.0, float(spec["score"]) + 0.06)
                inf.segment_count = len(spec["segments"])
                inf.meta = json.dumps({"demo": True, "source_kind": spec["source_kind"]}, ensure_ascii=False)
            upserted_inf += 1

            rq = (
                db.query(ReviewQueue)
                .filter(ReviewQueue.source_type == "annotation", ReviewQueue.source_id == point)
                .first()
            )
            if rq is None:
                rq = ReviewQueue(
                    id=str(uuid.uuid4()),
                    source_type="annotation",
                    source_id=point,
                    method=spec["method"],
                    model=f"demo-model-{spec['method']}",
                    point_name=point,
                    score=float(spec["score"]),
                    strategy="topk",
                    status=spec["status"],
                    reviewer=user,
                )
                db.add(rq)
            else:
                rq.method = spec["method"]
                rq.model = f"demo-model-{spec['method']}"
                rq.point_name = point
                rq.score = float(spec["score"])
                rq.strategy = "topk"
                rq.status = spec["status"]
                rq.reviewer = user
                rq.updated_at = datetime.utcnow()
            upserted_rq += 1

        db.commit()

        approved = sum(1 for s in specs if s["status"] == "approved")
        pending = sum(1 for s in specs if s["status"] == "pending")
        rejected = sum(1 for s in specs if s["status"] == "rejected")
        auto_cnt = sum(1 for s in specs if s["source_kind"] == "auto")
        human_cnt = sum(1 for s in specs if s["source_kind"] == "human")

        print("[done] demo review data seeded")
        print(f"  user={user}")
        print(f"  prefix={prefix}")
        print(f"  downsampled_dir={downsampled_dir}")
        print(f"  csv={created_csv}, annotations={upserted_ann}, inference={upserted_inf}, review_queue={upserted_rq}")
        print(f"  status: approved={approved}, pending={pending}, rejected={rejected}")
        print(f"  source_kind: auto={auto_cnt}, human={human_cnt}")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed review demo data for visualization")
    parser.add_argument("--user", default=settings.DEFAULT_USER, help="target user id")
    parser.add_argument("--prefix", default="demo_review_", help="demo point prefix")
    parser.add_argument("--reset", action="store_true", help="cleanup demo rows by prefix before seeding")
    parser.add_argument(
        "--delete-csv-on-reset",
        action="store_true",
        help="delete matching downsampled csv files when used with --reset",
    )
    args = parser.parse_args()
    seed_demo_data(args.user, args.prefix, args.reset, args.delete_csv_on_reset)


if __name__ == "__main__":
    main()
