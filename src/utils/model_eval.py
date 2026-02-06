"""
Model evaluation runner for golden datasets.

Runs inference for each golden point, converts masks to intervals, computes metrics,
writes eval_results.json, and stores summary into ModelEval table.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from configs.settings import settings
from services.inference.evaluation import eval_metrics


MASK_COLUMNS = [
    "global_mask",
    "outlier_mask",
    "local_mask",
    "global_mask_cluster",
    "local_mask_cluster",
]


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_point_names(truth_dir: Path) -> List[str]:
    if not truth_dir.exists():
        return []
    points = []
    for f in truth_dir.glob("*_ds.csv_annotations.json"):
        name = f.name.replace("_ds.csv_annotations.json", "")
        if name:
            points.append(name)
    return sorted(points)


def _load_dataset_points(dataset_name: str, dataset_type: str = "golden") -> Optional[List[str]]:
    if not dataset_name:
        return None
    try:
        from src.db.database import SessionLocal, DatasetAsset, DatasetItem, init_db
        init_db()
        db = SessionLocal()
        asset = db.query(DatasetAsset).filter(
            DatasetAsset.name == dataset_name,
            DatasetAsset.dataset_type == dataset_type,
        ).first()
        if not asset:
            return None
        rows = db.query(DatasetItem.point_name).filter(DatasetItem.dataset_id == asset.id).all()
        points = [str(r[0]) for r in rows]
        return points
    except Exception:
        return None
    finally:
        try:
            db.close()
        except Exception:
            pass


def _resolve_input_csv(point_name: str, data_path: Path) -> Optional[Path]:
    if not point_name:
        return None
    name = point_name
    if name.endswith(".csv"):
        candidate = data_path / name
        if candidate.exists():
            return candidate
    candidate = data_path / f"{name}.csv"
    if candidate.exists():
        return candidate
    candidate = data_path / f"{name}_ds.csv"
    if candidate.exists():
        return candidate
    return None


def _merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


def _intervals_from_annotation(payload: dict) -> List[List[int]]:
    intervals: List[List[int]] = []
    for ann in payload.get("annotations", []):
        for seg in ann.get("segments", []):
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                continue
            intervals.append([int(start), int(end)])
    return _merge_intervals(intervals)


def _build_truth_index(truth_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not truth_dir.exists():
        return index
    for f in truth_dir.glob("*.json"):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            filename = payload.get("filename")
            if filename:
                filename = str(filename).replace(".csv", "").replace(".json", "")
                index[filename] = f
        except Exception:
            pass
        stem = f.stem.replace(".csv", "")
        index.setdefault(stem, f)
    return index


def _load_ground_truth_intervals(point_name: str, truth_dir: Path, truth_index: Dict[str, Path]) -> Optional[List[List[int]]]:
    # 1) eval_metrics format file
    eval_file = truth_dir / f"{point_name}_ds.csv_annotations.json"
    if eval_file.exists():
        return eval_metrics.load_ground_truth(str(eval_file))

    # 2) annotation json lookup
    key = point_name.replace(".csv", "").replace(".json", "")
    candidate = truth_index.get(key)
    if candidate and candidate.exists():
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return _intervals_from_annotation(payload)
        except Exception:
            return None
    return None


def _normalize_point_name(input_csv: Path) -> str:
    name = input_csv.stem
    if name.endswith("_ds"):
        return name[:-3]
    return name


def _strip_date_suffix(name: str) -> str:
    import re
    name = re.sub(r"_\\d{8}(?:_\\d{6})?_to_\\d{8}(?:_\\d{6})?$", "", name)
    name = re.sub(r"_\\d{8}(?:_\\d{6})?$", "", name)
    return name


def _short_point_name(raw: str) -> str:
    import re
    if not raw:
        return raw
    name = os.path.basename(str(raw))
    if name.lower().endswith(".csv"):
        name = name[:-4]
    name = _strip_date_suffix(name)

    match = re.search(r"(?:^|_)([A-Za-z]{1,3}_[A-Za-z0-9]+\\.[A-Za-z0-9]+)$", name)
    if match:
        return match.group(1)

    parts = name.split("_")
    for i in range(len(parts) - 1, -1, -1):
        if "." in parts[i]:
            if i > 0:
                return f"{parts[i - 1]}_{parts[i]}"
            return parts[i]
    return name


def _mask_to_intervals(mask: pd.Series) -> List[List[int]]:
    if mask is None:
        return []
    indices = np.where(mask.astype(int).values > 0)[0].tolist()
    if not indices:
        return []
    groups = _split_continuous_outliers(indices, gp=1)
    intervals: List[List[int]] = []
    for group in groups:
        if not group:
            continue
        intervals.append([int(group[0]), int(group[-1])])
    return intervals


def _split_continuous_outliers(outlier_indices: List[int], min_size: int = 1, gp: int = 1) -> List[List[int]]:
    split_indices: List[List[int]] = []
    current: List[int] = []
    for idx in outlier_indices:
        if not current or idx == current[-1] + gp:
            current.append(idx)
        else:
            if len(current) >= min_size:
                split_indices.append(current)
            current = [idx]
    if len(current) >= min_size:
        split_indices.append(current)
    return split_indices


def _pick_mask_column(df: pd.DataFrame) -> Optional[str]:
    for col in MASK_COLUMNS:
        if col in df.columns:
            return col
    return None


def _resolve_lora_base_model(model_path: Path) -> Tuple[Optional[str], Optional[str]]:
    adapter_config = model_path / "adapter_config.json"
    if not adapter_config.exists():
        return None, None
    try:
        payload = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model = payload.get("base_model_name_or_path")
        if base_model:
            return base_model, str(model_path)
    except Exception:
        return None, None
    return None, None


def _build_inference_cmd(
    input_csv: Path,
    output_dir: Path,
    task_name: str,
    method: str,
    model_path: Path,
    downsampler: str,
    n_downsample: int,
    device: Optional[str],
    skip_db: bool = True,
) -> List[str]:
    python_exe = settings.PYTHON_UNIFIED if settings.USE_LOCAL_MODULES else settings.PYTHON_ILABEL
    run_py = Path(settings.CHECK_OUTLIER_PATH) / "run.py"

    cmd = [
        python_exe,
        str(run_py),
        "--input",
        str(input_csv),
        "--method",
        method,
        "--task_name",
        task_name,
        "--data_path",
        str(output_dir),
        "--downsampler",
        downsampler,
        "--n_downsample",
        str(n_downsample),
    ]

    base_model, lora_adapter = _resolve_lora_base_model(model_path)
    if method in {"chatts", "qwen"}:
        cmd.extend(["--chatts_model_path", base_model or str(model_path)])
        if lora_adapter:
            cmd.extend(["--chatts_lora_adapter_path", lora_adapter])

    if device:
        cmd.extend(["--chatts_device", device])

    if skip_db:
        cmd.append("--skip_db")

    return cmd


def _run_inference(cmd: List[str], timeout: int) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = result.returncode == 0
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return ok, output.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def _locate_output_csv(
    output_dir: Path,
    task_name: str,
    method: str,
    point_name: str,
    downsampler: str,
) -> Optional[Path]:
    folder = output_dir / task_name / method
    if not folder.exists():
        folder = output_dir / task_name
    date_str = "20230101"
    expected = folder / f"{method}_{downsampler}_{point_name}_{date_str}.csv"
    if expected.exists():
        return expected

    candidates = list(folder.glob(f"*_{point_name}_*.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _compute_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    summary: Dict[str, float] = {}
    numeric_keys = []
    for key, value in rows[0].items():
        if isinstance(value, (int, float)) and key != "point_name":
            numeric_keys.append(key)
    for key in numeric_keys:
        values = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        if values:
            summary[key] = round(float(np.mean(values)), 4)
    summary["points"] = len(rows)
    return summary


def evaluate_model_on_golden(
    model_path: str,
    model_family: str,
    truth_dir: str,
    data_dir: str,
    dataset_name: str = "golden",
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    method: Optional[str] = None,
    downsampler: str = "none",
    n_downsample: int = None,
    timeout: int = 900,
) -> Dict:
    model_dir = Path(model_path)
    truth_path = Path(truth_dir)
    data_path = Path(data_dir)

    if not model_dir.exists():
        return {"success": False, "error": f"模型目录不存在: {model_path}"}
    if not truth_path.exists():
        return {"success": False, "error": f"真实标注目录不存在: {truth_dir}"}
    if not data_path.exists():
        return {"success": False, "error": f"数据目录不存在: {data_dir}"}

    method = method or ("qwen" if model_family == "qwen" else "chatts")
    n_downsample = n_downsample or settings.DEFAULT_DOWNSAMPLE_POINTS

    run_id = time.strftime("%Y%m%d_%H%M%S")
    task_name = f"eval_{run_id}"
    output_root = Path(output_dir) if output_dir else (model_dir / "eval_outputs")
    _safe_mkdir(output_root)

    dataset_points = _load_dataset_points(dataset_name, dataset_type="golden")
    if dataset_points is not None and len(dataset_points) == 0:
        return {"success": False, "error": f"数据集为空: {dataset_name}"}

    truth_index = _build_truth_index(truth_path)
    point_names = dataset_points or _list_point_names(truth_path)
    detect_results: Dict[str, List[List[int]]] = {}
    inference_logs: Dict[str, str] = {}
    skipped: List[str] = []

    for point in point_names:
        input_csv = _resolve_input_csv(point, data_path)
        if not input_csv:
            skipped.append(point)
            continue

        cmd = _build_inference_cmd(
            input_csv=input_csv,
            output_dir=output_root,
            task_name=task_name,
            method=method,
            model_path=model_dir,
            downsampler=downsampler,
            n_downsample=n_downsample,
            device=device,
            skip_db=True,
        )
        ok, log = _run_inference(cmd, timeout)
        inference_logs[point] = log
        if not ok:
            skipped.append(point)
            continue

        output_point = _short_point_name(input_csv.stem)
        output_csv = _locate_output_csv(output_root, task_name, method, output_point, downsampler)
        if not output_csv or not output_csv.exists():
            skipped.append(point)
            continue

        try:
            df = pd.read_csv(output_csv)
        except Exception:
            skipped.append(point)
            continue

        mask_col = _pick_mask_column(df)
        if not mask_col:
            skipped.append(point)
            continue

        detect_results[point] = _mask_to_intervals(df[mask_col])

    results: List[Dict] = []
    for point in point_names:
        true_intervals = _load_ground_truth_intervals(point, truth_path, truth_index)
        detected_intervals = detect_results.get(point)
        point_input_csv = _resolve_input_csv(point, data_path)
        csv_data = None
        if point_input_csv and point_input_csv.exists():
            try:
                csv_data = pd.read_csv(point_input_csv)
            except Exception:
                csv_data = None
        if csv_data is None or csv_data.empty:
            continue
        if true_intervals is None or detected_intervals is None:
            continue
        metrics = eval_metrics.calculate_combined_metrics(true_intervals, detected_intervals, len(csv_data))
        metrics["point_name"] = point
        results.append(metrics)

    summary = _compute_summary(results)
    results_csv = output_root / task_name / "eval_results.csv"
    if results:
        try:
            pd.DataFrame(results).to_csv(results_csv, index=False, encoding="utf-8-sig")
        except Exception:
            pass
    eval_payload = {
        "model_path": str(model_dir),
        "model_family": model_family,
        "method": method,
        "dataset_name": dataset_name,
        "truth_dir": str(truth_path),
        "data_dir": str(data_path),
        "output_dir": str(output_root / task_name),
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "results": results,
        "skipped": skipped,
        "results_csv": str(results_csv),
    }

    eval_results_path = model_dir / "eval_results.json"
    try:
        eval_results_path.write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        from src.db.database import SessionLocal, ModelEval, init_db

        init_db()
        db = SessionLocal()
        record = ModelEval(
            id=str(uuid.uuid4()),
            model_path=str(model_dir),
            dataset_name=dataset_name,
            metrics=json.dumps(
                {
                    "summary": summary,
                    "results_path": str(eval_results_path),
                    "results_csv": str(results_csv),
                    "run_id": run_id,
                },
                ensure_ascii=False,
            ),
        )
        db.add(record)
        db.commit()
    except Exception:
        pass
    finally:
        try:
            db.close()
        except Exception:
            pass

    return {
        "success": True,
        "summary": summary,
        "results_path": str(eval_results_path),
        "results_csv": str(results_csv),
        "output_dir": str(output_root / task_name),
        "skipped": skipped,
        "points": len(results),
    }
