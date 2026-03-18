"""
标注工作台 API — 原生 FastAPI 实现（替代 Flask 代理）
提供: 文件浏览、CSV 数据读取(含 M4 降采样)、标注 CRUD、标签配置
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from configs.settings import settings
from src.core.logging_config import get_logger
from src.db.database import AnnotationRecord, get_db
from src.utils.annotation_store import (
    get_annotation_record,
    record_to_payload,
    upsert_annotation,
)
from src.api.auth import verify_token_from_header
from fastapi import Request

router = APIRouter()
logger = get_logger(__name__)

# 允许的数据目录
ALLOWED_DATA_DIRS = [
    settings.DATA_DOWNSAMPLED_DIR,
    settings.DATA_RAW_DIR,
    os.path.join(settings.DATA_INFERENCE_DIR, "chatts"),
    os.path.join(settings.DATA_INFERENCE_DIR, "qwen"),
    os.path.join(settings.DATA_INFERENCE_DIR, "timer"),
    os.path.join(settings.DATA_INFERENCE_DIR, "adtk_hbos"),
    settings.ANNOTATIONS_ROOT,
]

# 默认数据目录
DEFAULT_DATA_DIR = settings.DATA_DOWNSAMPLED_DIR


# ==================== Schemas ====================

class SetPathRequest(BaseModel):
    path: str


class SaveAnnotationRequest(BaseModel):
    filename: str
    annotations: List[dict] = []
    overall_attribute: dict = {}


class LabelConfig(BaseModel):
    overall_attribute: dict = {}
    local_change: dict = {}
    custom_labels: List[dict] = []


# ==================== 文件浏览 ====================

@router.post("/set-path")
def set_path(req: SetPathRequest):
    """设置当前数据目录"""
    target = Path(req.path).resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在: {req.path}")
    return {"success": True, "path": str(target)}


@router.post("/browse")
def browse_directory(req: SetPathRequest):
    """浏览服务器目录"""
    target = Path(req.path).resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在: {req.path}")

    dirs = []
    files = []
    try:
        for entry in sorted(target.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": str(entry), "type": "dir"})
            elif entry.suffix.lower() in (".csv", ".xlsx", ".xls"):
                files.append({
                    "name": entry.name,
                    "path": str(entry),
                    "type": "file",
                    "size": entry.stat().st_size,
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail="无权限访问该目录")

    return {
        "success": True,
        "current_path": str(target),
        "parent_path": str(target.parent),
        "dirs": dirs,
        "files": files,
    }


@router.get("/files")
def list_files(request: Request, data_path: str = Query(default=""), user: str = Query(default=""), db: Session = Depends(get_db)):
    """列出指定目录下的 CSV 文件"""
    base = Path(data_path).resolve() if data_path else Path(DEFAULT_DATA_DIR)
    if not base.exists():
        return {"success": True, "files": [], "path": str(base)}

    # 解析 Token 或使用 Query 中传入的，默认回退
    if not user:
        user = verify_token_from_header(request.headers.get("Authorization")) or "douff"
    if user == "default":
        user = "douff"

    # 查询当前用户的标注记录
    records = db.query(AnnotationRecord).filter(AnnotationRecord.user_id == user).all()
    # 将记录按文件名映射，注意有些旧数据可能没有 source_id
    ann_counts = {}
    for r in records:
        # payload 中可能会覆盖 filename
        payload = record_to_payload(r)
        fname = payload.get("filename", r.source_id)
        ann_counts[fname] = r.annotation_count or 0

    csv_files = []
    for f in sorted(base.glob("*.csv")):
        count = ann_counts.get(f.name, 0)
        csv_files.append({
            "name": f.name,
            "filename": f.name,
            "path": str(f),
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
            "size_bytes": f.stat().st_size,
            "modified_time": f.stat().st_mtime,
            "has_annotations": count > 0,
            "annotation_count": count,
        })
    return {"success": True, "files": csv_files, "path": str(base)}


# ==================== 数据读取 (含 M4 降采样) ====================

def _m4_downsample(x: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
    """M4 降采样: 每个桶保留 first, last, min, max."""
    n = len(x)
    if n <= n_out:
        return np.arange(n)

    n_buckets = n_out // 4
    if n_buckets < 1:
        n_buckets = 1
    bucket_size = n / n_buckets
    indices = set()
    for i in range(n_buckets):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        end = min(end, n)
        if start >= end:
            continue
        bucket_y = y[start:end]
        indices.add(start)              # first
        indices.add(end - 1)            # last
        indices.add(start + int(np.argmin(bucket_y)))  # min
        indices.add(start + int(np.argmax(bucket_y)))  # max
    return np.array(sorted(indices))


@router.get("/data/{filename:path}")
def get_data(filename: str, data_path: str = Query(default="")):
    """读取 CSV 数据, 智能列检测 + M4 降采样"""
    base = Path(data_path) if data_path else Path(DEFAULT_DATA_DIR)
    filepath = base / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    try:
        ext = filepath.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(filepath)
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败: {e}")

    original_len = len(df)
    columns = df.columns.tolist()

    # ---- 智能列检测 ----
    time_col = None
    val_col = None
    label_col = None

    for col in columns:
        if col == "" or str(col).startswith("Unnamed"):
            continue
        # 时间列
        if time_col is None and df[col].dtype == "object":
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors="raise")
                    time_col = col
                except Exception:
                    pass
        # 数值列
        if val_col is None and pd.api.types.is_numeric_dtype(df[col]):
            val_col = col
        # 标签列
        if label_col is None and df[col].dtype == "object" and col != time_col:
            if df[col].nunique() <= 20:
                label_col = col

    if val_col is None and len(columns) >= 2:
        for col in columns:
            if col != time_col and not str(col).startswith("Unnamed"):
                val_col = col
                break
    if val_col is None and columns:
        val_col = columns[-1]

    # 准备数值列
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)

    # ---- M4 降采样 ----
    MAX_ROWS = 10000
    downsampled = False
    if original_len > MAX_ROWS:
        x_arr = np.arange(original_len, dtype=np.float64)
        y_arr = df[val_col].values.astype(np.float64)
        indices = _m4_downsample(x_arr, y_arr, MAX_ROWS)
        df = df.iloc[indices]
        downsampled = True

    # ---- 构建响应 ----
    data_points = []
    series_name = val_col or "value"
    for orig_idx, row in df.iterrows():
        data_points.append({
            "idx": int(orig_idx),
            "val": float(row[val_col]) if pd.notna(row[val_col]) else 0.0,
            "label": str(row[label_col]) if label_col and pd.notna(row[label_col]) else "",
        })

    return {
        "success": True,
        "filename": filename,
        "columns": columns,
        "data": data_points,
        "seriesName": series_name,
        "originalLength": original_len,
        "downsampled": downsampled,
    }


# ==================== 标注 CRUD ====================

@router.get("/annotations/{filename:path}")
def get_annotations(request: Request, filename: str, user: str = Query(default=""), db: Session = Depends(get_db)):
    """获取文件标注"""
    if not user:
        user = verify_token_from_header(request.headers.get("Authorization")) or "douff"
    if user == "default":
        user = "douff"
        
    record = get_annotation_record(db, user, filename)
    if record is None:
        return {"success": True, "filename": filename, "annotations": []}
    payload = record_to_payload(record, fallback_filename=filename)
    return {
        "success": True,
        "filename": payload.get("filename", filename),
        "annotations": payload.get("annotations", []),
        "overall_attribute": payload.get("overall_attribute", {}),
    }


@router.post("/annotations/{filename:path}")
def save_annotations(request: Request, filename: str, req: SaveAnnotationRequest, user: str = Query(default=""), db: Session = Depends(get_db)):
    """保存标注"""
    if not user:
        user = verify_token_from_header(request.headers.get("Authorization")) or "douff"
    if user == "default":
        user = "douff"
        
    save_data = {
        "filename": req.filename or filename,
        "annotations": req.annotations,
        "overall_attribute": req.overall_attribute,
    }
    record = upsert_annotation(db, user, filename, save_data)
    db.commit()
    return {
        "success": True,
        "annotation_id": record.id,
        "annotation_count": int(record.annotation_count or 0),
    }


@router.delete("/annotations/{filename:path}")
def delete_annotation(request: Request, filename: str, annotation_id: str = Query(...), user: str = Query(default=""), db: Session = Depends(get_db)):
    """删除一个标注"""
    if not user:
        user = verify_token_from_header(request.headers.get("Authorization")) or "douff"
    if user == "default":
        user = "douff"
        
    record = get_annotation_record(db, user, filename)
    if record is None:
        raise HTTPException(status_code=404, detail="标注未找到")
    payload = record_to_payload(record, fallback_filename=filename)
    annotations = payload.get("annotations", [])
    payload["annotations"] = [a for a in annotations if (a or {}).get("id") != annotation_id]
    upsert_annotation(db, user, filename, payload)
    db.commit()
    return {"success": True}


@router.get("/annotations-all")
def get_all_annotations(request: Request, user: str = Query(default=""), db: Session = Depends(get_db)):
    """获取用户所有标注"""
    if not user:
        user = verify_token_from_header(request.headers.get("Authorization")) or "douff"
    if user == "default":
        user = "douff"
        
    rows = (
        db.query(AnnotationRecord)
        .filter(AnnotationRecord.user_id == user)
        .order_by(AnnotationRecord.updated_at.desc())
        .all()
    )
    results = []
    for row in rows:
        payload = record_to_payload(row)
        results.append({
            "filename": payload.get("filename", row.source_id),
            "annotations": payload.get("annotations", []),
            "overall_attribute": payload.get("overall_attribute", {}),
        })
    return {"success": True, "annotations": results, "count": len(results)}


# ==================== 标签配置 ====================

LABELS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           "services", "annotator", "backend", "config", "labels.json")


@router.get("/labels")
def get_labels():
    """获取标签配置"""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return {"success": True, "labels": labels}
    return {
        "success": True,
        "labels": {
            "overall_attribute": {
                "趋势": {"options": ["上升", "下降", "平稳", "其他"]},
                "噪声水平": {"options": ["高", "中", "低"]},
                "周期性": {"options": ["有周期", "无周期", "总体有周期"]},
            },
            "local_change": {
                "趋势": {"options": ["上升", "下降", "平稳", "其他"]},
                "置信度": {"options": ["high", "medium", "low"]},
            },
            "custom_labels": [],
        },
    }


@router.post("/labels")
def save_labels(config: LabelConfig):
    """保存标签配置"""
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, ensure_ascii=False, indent=2)
    return {"success": True}
