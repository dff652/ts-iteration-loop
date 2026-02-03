#!/usr/bin/env python3
import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = PROJECT_ROOT / "services" / "inference"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from configs.settings import settings
from src.db.database import SessionLocal, init_db, InferenceResult, SegmentScore
from evaluation import get_evaluator
from evaluation.lb_eval import avg_score


def infer_method_from_path(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    if "inference" in parts:
        idx = parts.index("inference")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def parse_filename(stem: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    parts = stem.split("_")
    if len(parts) < 4:
        return None, None, None, None
    date = parts[-1]
    if len(date) != 8 or not date.isdigit():
        return None, None, None, None
    method = parts[0]
    downsampler = parts[1]
    point_name = "_".join(parts[2:-1])
    return method, downsampler, point_name, date


def find_metrics_files(file_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    folder = file_path.parent
    stem = file_path.stem
    candidates = [
        (folder / f"{stem}_metrics.json", folder / f"{stem}_segments.json"),
        (folder / "metrics.json", folder / "segments.json"),
    ]
    for metrics_path, segments_path in candidates:
        if metrics_path.exists():
            if segments_path.exists():
                return metrics_path, segments_path
            return metrics_path, None
    return None, None


def load_metrics(metrics_path: Path) -> Dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_segments(segments_path: Path) -> List[Dict]:
    with open(segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def safe_float(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def parse_segment_index(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return safe_int(value[0], None), safe_int(value[1], None)
    if value is None:
        return None, None
    text = str(value)
    if "-" in text:
        left, right = text.split("-", 1)
        return safe_int(left, None), safe_int(right, None)
    return None, None


def segments_from_eval_df(df: pd.DataFrame) -> List[Dict]:
    segments: List[Dict] = []
    if df is None or df.empty:
        return segments
    for _, row in df.iterrows():
        start, end = parse_segment_index(row.get("index"))
        if start is None or end is None:
            continue
        segments.append({
            "start": int(start),
            "end": int(end),
            "score": safe_float(row.get("score"), 0.0),
            "raw_p": safe_float(row.get("raw_p"), 0.0),
            "left": safe_float(row.get("left"), 0.0),
            "right": safe_float(row.get("right"), 0.0),
            "length": safe_int(row.get("data_len"), max(0, end - start + 1)),
        })
    return segments


def compute_summary(segments: List[Dict]) -> Dict:
    scores = [seg.get("score", 0.0) for seg in segments if seg.get("score") is not None]
    if scores:
        score_avg = float(avg_score(scores))
        score_max = float(np.max(scores))
        score_min = float(np.min(scores))
        score_p50 = float(np.percentile(scores, 50))
        avg_method = "p_norm_5"
    else:
        score_avg = 0.0
        score_max = 0.0
        score_min = 0.0
        score_p50 = 0.0
        avg_method = "none"
    return {
        "score_avg": score_avg,
        "score_max": score_max,
        "score_min": score_min,
        "score_p50": score_p50,
        "segment_count": len(segments),
        "score_avg_method": avg_method,
    }


def compute_from_csv(file_path: Path, mask_col: Optional[str]) -> Tuple[List[Dict], Dict]:
    df = pd.read_csv(file_path)
    if mask_col is None:
        for candidate in ["global_mask", "outlier_mask", "local_mask", "global_mask_cluster"]:
            if candidate in df.columns:
                mask_col = candidate
                break
    if mask_col is None or mask_col not in df.columns:
        return [], compute_summary([])
    series_col = None
    time_like = {"time", "timestamp", "date", "datetime", "Time", "Date"}
    for col in df.columns:
        if col in time_like:
            continue
        if col not in {"outlier_mask", "global_mask", "local_mask", "global_mask_cluster", "local_mask_cluster", "orig_pos"}:
            if pd.api.types.is_numeric_dtype(df[col]):
                series_col = col
                break
    if series_col is None:
        return [], compute_summary([])

    series = df[series_col].values
    mask = df[mask_col].values
    indices = np.where(mask > 0)[0].astype(int)
    evaluator = get_evaluator("lb_eval")
    result = evaluator.evaluate(series, indices)
    eval_df = result.get("result") if isinstance(result, dict) else None
    segments = segments_from_eval_df(eval_df if isinstance(eval_df, pd.DataFrame) else None)
    summary = compute_summary(segments)
    return segments, summary


def write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_generated_at(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1]
        return datetime.fromisoformat(value)
    except Exception:
        return None


def build_meta(task_name: str, downsampler: str, sensor_path: str, sensor_column: str, date_str: str) -> Dict:
    return {
        "task_name": task_name,
        "downsampler": downsampler,
        "sensor_path": sensor_path,
        "sensor_column": sensor_column,
        "start_time": date_str,
        "end_time": date_str,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill inference results into DB")
    parser.add_argument("--root", type=str, default=settings.DATA_INFERENCE_DIR, help="Root directory to scan")
    parser.add_argument("--method", type=str, default=None, help="Filter by method")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("--mask-col", type=str, default=None, help="Mask column to use if computing")
    parser.add_argument("--write-json", action="store_true", help="Write metrics/segments JSON if missing")
    parser.add_argument("--write-db", action="store_true", help="Write to database")
    parser.add_argument("--recompute", action="store_true", help="Recompute even if JSON exists")
    parser.add_argument("--upsert", action="store_true", help="Update existing DB rows")
    parser.add_argument("--dry-run", action="store_true", help="Do not write JSON or DB")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root not found: {root}")
        return 1

    files = [p for p in root.rglob("*.csv") if p.is_file()]
    if args.method:
        files = [p for p in files if (infer_method_from_path(p) or "").lower() == args.method.lower()]
    if args.limit:
        files = files[: args.limit]

    if not files:
        print("No CSV files found.")
        return 0

    if args.write_db and not args.dry_run:
        init_db()

    created = 0
    updated = 0
    skipped = 0

    for file_path in files:
        method_from_dir = infer_method_from_path(file_path)
        method_from_name, downsampler, point_name, date_str = parse_filename(file_path.stem)
        method = method_from_dir or method_from_name

        rel_parts = []
        try:
            rel_parts = list(file_path.relative_to(root).parts)
        except Exception:
            rel_parts = list(file_path.parts)
        task_name = rel_parts[0] if len(rel_parts) > 1 else ""

        metrics_path, segments_path = find_metrics_files(file_path)
        metrics_payload = None
        segments: List[Dict] = []
        summary = None

        if metrics_path and not args.recompute:
            metrics_payload = load_metrics(metrics_path)
            summary = metrics_payload.get("summary") if isinstance(metrics_payload, dict) else None
            if segments_path:
                segments = load_segments(segments_path)
        else:
            segments, summary = compute_from_csv(file_path, args.mask_col)

        if summary is None:
            summary = compute_summary(segments)

        if args.write_json and not args.dry_run:
            if metrics_path is None:
                metrics_path = file_path.with_name(f"{file_path.stem}_metrics.json")
            if segments_path is None:
                segments_path = file_path.with_name(f"{file_path.stem}_segments.json")

            if metrics_payload is None:
                metrics_payload = {
                    "version": 1,
                    "summary": summary,
                    "method": method or "",
                    "downsampler": downsampler or "",
                    "task_name": task_name,
                    "task_id": None,
                    "point_name": point_name or file_path.stem,
                    "result_csv": file_path.name,
                    "result_path": str(file_path),
                    "metrics_path": str(metrics_path),
                    "segments_path": str(segments_path),
                    "model_path": "",
                    "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
            write_json(metrics_path, metrics_payload)
            write_json(segments_path, segments)

        if args.write_db and not args.dry_run:
            db = SessionLocal()
            try:
                existing = db.query(InferenceResult).filter(InferenceResult.result_path == str(file_path)).first()
                if existing and not args.upsert:
                    skipped += 1
                    db.close()
                    continue

                inference_id = existing.id if existing else str(uuid.uuid4())
                score_avg = safe_float(summary.get("score_avg"), 0.0)
                score_max = safe_float(summary.get("score_max"), 0.0)
                segment_count = safe_int(summary.get("segment_count"), len(segments))

                meta = build_meta(task_name, downsampler or "", str(file_path.parent), point_name or file_path.stem, date_str or "")
                meta.update({
                    "score_min": safe_float(summary.get("score_min"), 0.0),
                    "score_p50": safe_float(summary.get("score_p50"), 0.0),
                    "score_avg_method": summary.get("score_avg_method", ""),
                })

                if existing:
                    existing.method = method or existing.method
                    existing.model = metrics_payload.get("model_path", "") if metrics_payload else existing.model
                    existing.point_name = point_name or existing.point_name
                    existing.result_path = str(file_path)
                    existing.metrics_path = str(metrics_path) if metrics_path else existing.metrics_path
                    existing.segments_path = str(segments_path) if segments_path else existing.segments_path
                    existing.score_avg = score_avg
                    existing.score_max = score_max
                    existing.segment_count = segment_count
                    existing.meta = json.dumps(meta, ensure_ascii=False)
                    db.query(SegmentScore).filter(SegmentScore.inference_id == inference_id).delete()
                    updated += 1
                else:
                    record = InferenceResult(
                        id=inference_id,
                        task_id=metrics_payload.get("task_id") if metrics_payload else None,
                        method=method or "",
                        model=metrics_payload.get("model_path", "") if metrics_payload else "",
                        point_name=point_name or file_path.stem,
                        result_path=str(file_path),
                        metrics_path=str(metrics_path) if metrics_path else "",
                        segments_path=str(segments_path) if segments_path else "",
                        score_avg=score_avg,
                        score_max=score_max,
                        segment_count=segment_count,
                        meta=json.dumps(meta, ensure_ascii=False),
                    )
                    created += 1
                    if metrics_payload and metrics_payload.get("generated_at"):
                        created_at = parse_generated_at(metrics_payload.get("generated_at"))
                        if created_at:
                            record.created_at = created_at
                    db.add(record)

                for seg in segments:
                    db.add(SegmentScore(
                        inference_id=inference_id,
                        start=seg.get("start"),
                        end=seg.get("end"),
                        score=seg.get("score"),
                        raw_p=seg.get("raw_p"),
                        left=seg.get("left"),
                        right=seg.get("right"),
                    ))

                db.commit()
            except Exception as e:
                db.rollback()
                print(f"DB write failed for {file_path}: {e}")
            finally:
                db.close()
        else:
            skipped += 1

    print(f"Total: {len(files)}  Created: {created}  Updated: {updated}  Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
