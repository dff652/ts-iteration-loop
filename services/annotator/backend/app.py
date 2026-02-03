import os
import json
import time
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tsdownsample import M4Downsampler
from configs.settings import settings
from .file_registry import init_db, sync_directory, list_files, rebuild_directory

# Import authentication module
# Fix import path for sibling modules when run from project root
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import authentication module
from auth import login_required, verify_password, generate_token, load_users

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)

# Configuration directories
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
# 使用标准化标注目录 (从 settings 导入)
ANNOTATIONS_DIR = settings.ANNOTATIONS_ROOT
LABELS_FILE = os.path.join(BASE_DIR, 'config', 'labels.json')
REGISTRY_DB = os.path.join(DATA_DIR, 'file_registry.db')

# Allowed data directories (whitelist)
ALLOWED_DATA_DIRS = [
    settings.DATA_DOWNSAMPLED_DIR,
    settings.DATA_RAW_DIR,
    os.path.join(settings.DATA_INFERENCE_DIR, "chatts"),
    os.path.join(settings.DATA_INFERENCE_DIR, "qwen"),
    os.path.join(settings.DATA_INFERENCE_DIR, "timer"),
    os.path.join(settings.DATA_INFERENCE_DIR, "adtk_hbos"),
    os.path.join(settings.DATA_INFERENCE_DIR, "ensemble"),
]

_ALLOWED_ROOTS = [Path(p).resolve() for p in ALLOWED_DATA_DIRS]
_CSV_PATH_CACHE = {}


def _normalize_annotation_name(filename: str) -> str:
    name = filename or ""
    lower = name.lower()
    for ext in (".csv", ".xls", ".xlsx"):
        if lower.endswith(ext):
            return name[: -len(ext)]
    return name

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
init_db(REGISTRY_DB)

# Current working data path
CURRENT_DATA_PATH = DATA_DIR

# ==================== Level 3 Fallback Helpers ====================
import re
import sys

# 添加项目根目录到 path 以便导入共享模块
_PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_standard_filename(filename: str) -> dict:
    """
    解析标准化文件名: {PointID}_{StartTime}_{EndTime}_{Algorithm}.csv
    
    Returns:
        解析结果字典或 None
    """
    # 格式: AI_20405E.PV_20230101_000000_20231231_235959_chatts.csv
    pattern = r'^(.+?)_(\d{8}_\d{6})_(\d{8}_\d{6})_(\w+)\.csv$'
    match = re.match(pattern, filename)
    if match:
        try:
            return {
                'point_id': match.group(1),
                'start_time': datetime.strptime(match.group(2), '%Y%m%d_%H%M%S'),
                'end_time': datetime.strptime(match.group(3), '%Y%m%d_%H%M%S'),
                'algorithm': match.group(4)
            }
        except ValueError:
            return None
    return None


def infer_method_from_path(path: str) -> str:
    """Infer method from inference directory path, if any."""
    try:
        parts = Path(path).parts
    except Exception:
        return ""
    if "inference" in parts:
        idx = parts.index("inference")
        if idx + 1 < len(parts):
            method = parts[idx + 1].lower()
            if method in {"chatts", "qwen", "timer", "adtk_hbos", "ensemble"}:
                return method
    return ""


def _is_allowed_path(path: str) -> bool:
    try:
        p = Path(path).resolve()
    except Exception:
        return False
    for root in _ALLOWED_ROOTS:
        try:
            if root == p or root in p.parents:
                return True
        except Exception:
            continue
    return False


def _allowed_root_dirs():
    dirs = []
    for root in _ALLOWED_ROOTS:
        if root.exists() and root.is_dir():
            dirs.append({
                'name': root.name,
                'path': str(root),
                'is_dir': True,
                'has_data_files': True,
            })
    return dirs


def _annotation_csv_candidates(source_id: str) -> list[str]:
    candidates = []
    if not source_id:
        return candidates
    name = str(source_id)
    if name.endswith(".csv"):
        candidates.append(name)
    if name.endswith(".json"):
        candidates.append(name[:-5] + ".csv")
    if name.startswith("annotations_"):
        base = name.replace("annotations_", "", 1)
        if base.endswith(".json"):
            base = base[:-5]
        if not base.endswith(".csv"):
            base = base + ".csv"
        candidates.append(base)
    if not name.endswith(".csv") and not name.endswith(".json"):
        candidates.append(name + ".csv")
    if os.path.sep in name or "/" in name:
        base = os.path.basename(name)
        if base and base != name:
            candidates.append(base)
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def _find_csv_path_for_candidates(candidates: list[str]) -> Optional[Path]:
    if not candidates:
        return None
    safe_candidates: list[str] = []
    for name in candidates:
        cached = _CSV_PATH_CACHE.get(name)
        if cached:
            return Path(cached)
        if os.path.isabs(name):
            if _is_allowed_path(name):
                safe_candidates.append(name)
            continue
        safe_candidates.append(name)
    if not safe_candidates:
        return None

    # Direct match at root
    for root in _ALLOWED_ROOTS:
        if not root.exists():
            continue
        for name in safe_candidates:
            if os.path.isabs(name):
                p = Path(name)
            else:
                p = root / name
            if p.exists():
                _CSV_PATH_CACHE[name] = str(p)
                return p

    name_set = set(safe_candidates)
    stems = {Path(n).stem for n in safe_candidates}

    for root in _ALLOWED_ROOTS:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f in name_set:
                    p = Path(dirpath) / f
                    _CSV_PATH_CACHE[f] = str(p)
                    return p
            # fallback: stem include
            for f in filenames:
                if not f.endswith(".csv"):
                    continue
                if any(stem in f for stem in stems):
                    return Path(dirpath) / f
    return None


def _normalize_method(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"all", "any", "none", ""}:
        return None
    return normalized


def _parse_float_arg(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_inference_index_map(
    path: str,
    method: Optional[str],
    min_score: Optional[float],
    max_score: Optional[float],
    score_by: str = "score_avg",
    strategy: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        from src.db.database import SessionLocal, InferenceResult
    except Exception as e:
        return None, f"DB import failed: {e}"

    score_by = (score_by or "score_avg").strip().lower()
    score_col = InferenceResult.score_avg if score_by in {"avg", "score_avg", "mean"} else InferenceResult.score_max
    strategy = (strategy or "").strip().lower()

    db = SessionLocal()
    try:
        query = db.query(InferenceResult)
        if path:
            query = query.filter(InferenceResult.result_path.like(f"{path}%"))
        if method:
            query = query.filter(InferenceResult.method == method)
        if min_score is not None:
            query = query.filter(score_col >= min_score)
        if max_score is not None:
            query = query.filter(score_col <= max_score)
        if strategy in {"topk", "high", "score_desc"}:
            query = query.order_by(score_col.desc(), InferenceResult.created_at.desc())
        elif strategy in {"low_score", "low", "score_asc"}:
            query = query.order_by(score_col.asc(), InferenceResult.created_at.desc())
        elif strategy == "random":
            try:
                from sqlalchemy import func as _sa_func
                query = query.order_by(_sa_func.random())
            except Exception:
                query = query.order_by(InferenceResult.created_at.desc())
        else:
            query = query.order_by(InferenceResult.created_at.desc())
        if limit is not None and limit > 0:
            query = query.limit(limit)

        rows = query.all()
        index: Dict[str, Any] = {}
        for row in rows:
            filename = os.path.basename(row.result_path or "")
            if not filename:
                continue
            if filename in index:
                continue
            meta = {}
            if row.meta:
                try:
                    meta = json.loads(row.meta)
                except Exception:
                    meta = {}
            index[filename] = {
                "id": row.id,
                "method": row.method,
                "model": row.model,
                "result_path": row.result_path,
                "metrics_path": row.metrics_path,
                "segments_path": row.segments_path,
                "score_avg": row.score_avg,
                "score_max": row.score_max,
                "segment_count": row.segment_count,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "meta": meta,
            }
        return index, None
    except Exception as e:
        return None, str(e)
    finally:
        db.close()


def _sample_inference_rows(
    method: Optional[str],
    min_score: Optional[float],
    max_score: Optional[float],
    score_by: str,
    strategy: str,
    limit: int,
):
    try:
        from src.db.database import SessionLocal, InferenceResult
        from sqlalchemy import func as _sa_func
    except Exception as e:
        return None, f"DB import failed: {e}"

    score_by = (score_by or "score_avg").strip().lower()
    score_col = InferenceResult.score_avg if score_by in {"avg", "score_avg", "mean"} else InferenceResult.score_max
    strategy = (strategy or "topk").strip().lower()

    db = SessionLocal()
    try:
        query = db.query(InferenceResult)
        if method:
            query = query.filter(InferenceResult.method == method)
        if min_score is not None:
            query = query.filter(score_col >= min_score)
        if max_score is not None:
            query = query.filter(score_col <= max_score)

        if strategy in {"topk", "high", "score_desc"}:
            query = query.order_by(score_col.desc(), InferenceResult.created_at.desc())
        elif strategy in {"low_score", "low", "score_asc"}:
            query = query.order_by(score_col.asc(), InferenceResult.created_at.desc())
        elif strategy == "random":
            query = query.order_by(_sa_func.random())
        else:
            query = query.order_by(InferenceResult.created_at.desc())

        if limit and limit > 0:
            query = query.limit(limit)
        rows = query.all()
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        db.close()


def fetch_from_iotdb(metadata: dict) -> pd.DataFrame:
    """
    从 IoTDB 获取数据
    
    Args:
        metadata: 包含 point_id, start_time, end_time 的字典
        
    Returns:
        DataFrame 或 None
    """
    try:
        # 尝试导入共享配置
        try:
            from src.utils.iotdb_config import load_iotdb_config
            config = load_iotdb_config()
        except ImportError:
            # 使用默认配置
            config = {
                "host": "192.168.199.185",
                "port": "6667",
                "user": "root",
                "password": "root"
            }
        
        from iotdb.Session import Session
        
        session = Session(
            config['host'], 
            config['port'], 
            config['user'], 
            config['password'],
            fetch_size=2000000
        )
        session.open(False)
        
        st = metadata['start_time'].strftime('%Y-%m-%d %H:%M:%S')
        et = metadata['end_time'].strftime('%Y-%m-%d %H:%M:%S')
        point_id = metadata['point_id']
        
        # 注意：这里假设 path 需要从某处获取，暂时使用默认值
        # 实际使用时可能需要更复杂的路径推断逻辑
        default_path = config.get('default_path', 'root.supcon.nb.whlj.LJSJ')
        
        query = f"select `{point_id}` from {default_path} where time >= {st.replace(' ', 'T')} and time <= {et.replace(' ', 'T')}"
        
        result = session.execute_query_statement(query)
        df = result.todf()
        
        if len(df) > 0:
            df.set_index('Time', inplace=True)
            df.index = pd.to_datetime(df.index.astype('int64')).tz_localize('UTC').tz_convert('Asia/Shanghai')
        
        session.close()
        return df
        
    except Exception as e:
        print(f"[fetch_from_iotdb] Error: {e}")
        return None


# ==================== Static Files ====================
@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory(app.static_folder, 'index.html')


# ==================== Path Management (User-specific) ====================
@app.route('/api/set-path', methods=['POST'])
@login_required
def set_data_path(current_user):
    """Set custom data directory path for current user"""
    try:
        from auth import load_users, save_users
        
        data = request.get_json()
        path = data.get('path', '')
        
        if path and os.path.isdir(path) and _is_allowed_path(path):
            # Save path to user config
            users = load_users()
            if current_user in users:
                users[current_user]['data_path'] = path
                save_users(users)
            
            return jsonify({'success': True, 'path': path})
        return jsonify({
            'success': False,
            'error': 'Invalid or not allowed directory path',
            'allowed_roots': [str(p) for p in _ALLOWED_ROOTS]
        }), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/current-path', methods=['GET'])
@login_required
def get_current_path(current_user):
    """Get current data directory path for current user"""
    try:
        from auth import load_users
        
        users = load_users()
        user_path = users.get(current_user, {}).get('data_path', DATA_DIR)
        
        return jsonify({'success': True, 'path': user_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/browse-dir', methods=['GET'])
def browse_directory():
    """Browse server directory structure"""
    try:
        # Default to /home instead of user home directory
        path = request.args.get('path', '/home')
        
        if not _is_allowed_path(path):
            return jsonify({
                'success': True,
                'current_path': '/',
                'parent_path': None,
                'directories': _allowed_root_dirs(),
                'has_data_files': False,
                'allowed_roots': [str(p) for p in _ALLOWED_ROOTS]
            })
        
        if not os.path.exists(path):
            return jsonify({'success': False, 'error': 'Path does not exist'}), 404
        
        if not os.path.isdir(path):
            path = os.path.dirname(path)
        
        path = os.path.abspath(path)
        parent_path = os.path.dirname(path)
        
        directories = []
        try:
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                try:
                    if os.path.isdir(item_path):
                        has_data_files = False
                        try:
                            for f in os.listdir(item_path):
                                if f.endswith(('.csv', '.xls', '.xlsx')):
                                    has_data_files = True
                                    break
                        except PermissionError:
                            pass
                        
                        directories.append({
                            'name': item,
                            'path': item_path,
                            'is_dir': True,
                            'has_data_files': has_data_files
                        })
                except PermissionError:
                    continue
        except PermissionError:
            return jsonify({'success': False, 'error': 'Permission denied'}), 403
        
        current_has_data = any(
            f.endswith(('.csv', '.xls', '.xlsx')) 
            for f in os.listdir(path) 
            if os.path.isfile(os.path.join(path, f))
        )
        
        return jsonify({
            'success': True,
            'current_path': path,
            'parent_path': parent_path if parent_path != path else None,
            'directories': directories,
            'has_data_files': current_has_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rebuild-index', methods=['POST'])
@login_required
def rebuild_index(current_user):
    """Rebuild file registry for current user's path (or provided path)."""
    try:
        from auth import load_users

        data = request.get_json(silent=True) or {}
        users = load_users()
        user_path = users.get(current_user, {}).get('data_path', DATA_DIR)
        path = data.get('path') or user_path

        if not path or not os.path.isdir(path):
            return jsonify({'success': False, 'error': 'Path does not exist'}), 404
        if not _is_allowed_path(path):
            return jsonify({
                'success': False,
                'error': 'Invalid or not allowed directory path',
                'allowed_roots': [str(p) for p in _ALLOWED_ROOTS]
            }), 400

        rebuild_directory(REGISTRY_DB, path)
        method_filter = infer_method_from_path(path)
        files = list_files(REGISTRY_DB, path, method_filter or None)

        return jsonify({
            'success': True,
            'path': path,
            'count': len(files)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== File Management (User-specific path) ====================
@app.route('/api/files', methods=['GET'])
@login_required
def get_files(current_user):
    """Get all CSV and Excel files in current user's directory"""
    try:
        from auth import load_users
        
        min_score = _parse_float_arg(request.args.get('min_score'))
        max_score = _parse_float_arg(request.args.get('max_score'))
        score_by = request.args.get('score_by', 'score_avg')
        method_param = _normalize_method(request.args.get('method'))
        strategy = request.args.get('strategy')
        limit = request.args.get('limit', type=int)
        use_score_filter = (
            min_score is not None
            or max_score is not None
            or method_param is not None
            or strategy is not None
            or limit is not None
        )

        # Get user's data path
        users = load_users()
        user_path = users.get(current_user, {}).get('data_path', DATA_DIR)
        
        print(f"=== get_files for user: {current_user} ===")
        print(f"User path: {user_path}")
        print(f"Path exists: {os.path.exists(user_path)}")
        print(f"Is directory: {os.path.isdir(user_path)}")
        
        if not os.path.exists(user_path):
            return jsonify({'success': False, 'error': 'Path does not exist'}), 404
        
        # Sync and query file registry for this directory
        sync_directory(REGISTRY_DB, user_path)
        method_filter = method_param or infer_method_from_path(user_path)
        registry_files = list_files(REGISTRY_DB, user_path, method_filter or None)

        inference_index = None
        if use_score_filter:
            inference_index, err = _fetch_inference_index_map(
                user_path,
                method_filter,
                min_score,
                max_score,
                score_by=score_by,
                strategy=strategy,
                limit=limit,
            )
            if inference_index is None:
                return jsonify({'success': False, 'error': f'Inference index unavailable: {err}'}), 500
        
        files = []
        for entry in registry_files:
            f = entry["name"]
            full_path = os.path.join(user_path, f)
            if not os.path.isfile(full_path):
                continue
            if inference_index is not None and f not in inference_index:
                continue

            # Check for annotations in user's annotation directory
            user_ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
            
            # Try multiple annotation file name patterns (ordered by frequency)
            annotation_file = None
            annotation_count = 0
            has_annotations = False
            
            # Pattern 5: Direct replacement .csv -> .json (most common for current files)
            # This matches files like: 数据集zhlh_100_XXX.PV.json for CSV files like: 数据集zhlh_100_XXX.PV.csv
            pattern5 = os.path.join(user_ann_dir, f.replace('.csv', '.json'))
            # Pattern 1: filename.csv.json (standard auto-save format)
            pattern1 = os.path.join(user_ann_dir, f"{f}.json")
            # Pattern 4: 数据集filename.json (for CSV files without 数据集 prefix)
            # This matches files like: 数据集zhlh_100_XXX.json for CSV files like: zhlh_100_XXX.csv
            pattern4 = os.path.join(user_ann_dir, f"数据集{f.replace('.csv', '')}.json")
            # Pattern 3: annotations_filename.json (old export without 数据集)
            pattern3 = os.path.join(user_ann_dir, f"annotations_{f.replace('.csv', '')}.json")
            # Pattern 2: annotations_数据集filename.json (old export format)
            pattern2 = os.path.join(user_ann_dir, f"annotations_数据集{f.replace('.csv', '')}.json")
            
            for pattern in [pattern5, pattern1, pattern4, pattern3, pattern2]:
                if os.path.exists(pattern):
                    annotation_file = pattern
                    print(f"  -> Found annotation: {os.path.basename(pattern)}")
                    break
            
            if annotation_file and os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r', encoding='utf-8') as af:
                        ann_data = json.load(af)
                        # Support both formats
                        annotations = ann_data.get('annotations', [])
                        
                        # Has annotations if there are any annotations (including no-label)
                        if annotations:
                            has_annotations = True
                        
                        # Only count annotations with segments (exclude no-label annotations)
                        annotations_with_segments = [
                            ann for ann in annotations 
                            if ann.get('segments') and len(ann.get('segments', [])) > 0
                        ]
                        annotation_count = len(annotations_with_segments)
                except Exception as e:
                    print(f"  -> Error reading annotation: {e}")
                    pass
            
            file_info = {
                'name': f,
                'has_annotations': has_annotations,
                'annotation_count': annotation_count
            }

            if inference_index is not None:
                idx_info = inference_index.get(f, {})
                file_info.update({
                    'score_avg': idx_info.get('score_avg'),
                    'score_max': idx_info.get('score_max'),
                    'segment_count': idx_info.get('segment_count'),
                    'inference_id': idx_info.get('id'),
                    'method': idx_info.get('method'),
                    'metrics_path': idx_info.get('metrics_path'),
                    'segments_path': idx_info.get('segments_path'),
                })

            files.append(file_info)

        print(f"Matched files: {[f['name'] for f in files]}")
        
        return jsonify({
            'success': True,
            'files': files,
            'path': user_path,
            'filtered': use_score_filter,
            'score_by': score_by if use_score_filter else None,
            'strategy': strategy if use_score_filter else None,
            'limit': limit if use_score_filter else None
        })
    except Exception as e:
        print(f"Error in get_files: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review/sample', methods=['POST'])
@login_required
def review_sample(current_user):
    """Sample inference results into review queue."""
    try:
        data = request.get_json(silent=True) or {}
        source_type = data.get('source_type', 'inference')
        strategy = data.get('strategy', 'topk')
        limit = int(data.get('limit', 50) or 50)
        score_by = data.get('score_by', 'score_avg')
        method = _normalize_method(data.get('method'))
        min_score = _parse_float_arg(data.get('min_score'))
        max_score = _parse_float_arg(data.get('max_score'))

        from src.db.database import SessionLocal, ReviewQueue, InferenceResult, init_db
        init_db()

        rows = []
        err = None
        if source_type == 'annotation':
            ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
            if not os.path.isdir(ann_dir):
                return jsonify({'success': True, 'created': 0, 'skipped': 0, 'total': 0})
            entries = []
            for name in os.listdir(ann_dir):
                if not name.endswith('.json'):
                    continue
                full_path = os.path.join(ann_dir, name)
                try:
                    stat = os.stat(full_path)
                except OSError:
                    continue
                filename = None
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        filename = data.get('filename')
                except Exception:
                    filename = None
                if not filename:
                    filename = name.replace('.json', '')
                entries.append((filename, stat.st_mtime))

            if strategy == 'random':
                np.random.shuffle(entries)
            elif strategy in {'low_score', 'low'}:
                entries.sort(key=lambda x: x[1])
            else:
                entries.sort(key=lambda x: x[1], reverse=True)

            if limit and limit > 0:
                entries = entries[:limit]
            rows = entries
        else:
            rows, err = _sample_inference_rows(
                method=method,
                min_score=min_score,
                max_score=max_score,
                score_by=score_by,
                strategy=strategy,
                limit=limit,
            )
            if rows is None:
                return jsonify({'success': False, 'error': err}), 500

        db = SessionLocal()
        created = 0
        skipped = 0
        try:
            if source_type == 'annotation':
                for filename, _mtime in rows:
                    exists = db.query(ReviewQueue).filter(
                        ReviewQueue.source_type == 'annotation',
                        ReviewQueue.source_id == filename
                    ).first()
                    if exists:
                        skipped += 1
                        continue
                    db.add(ReviewQueue(
                        id=str(uuid.uuid4()),
                        source_type='annotation',
                        source_id=filename,
                        method=method,
                        model=None,
                        point_name=filename,
                        score=None,
                        strategy=strategy,
                        status='pending',
                        reviewer=None,
                    ))
                    created += 1
            else:
                for row in rows:
                    exists = db.query(ReviewQueue).filter(
                        ReviewQueue.source_type == 'inference',
                        ReviewQueue.source_id == row.id
                    ).first()
                    if exists:
                        skipped += 1
                        continue

                    score = row.score_avg if score_by in {"avg", "score_avg", "mean"} else row.score_max
                    db.add(ReviewQueue(
                        id=str(uuid.uuid4()),
                        source_type='inference',
                        source_id=row.id,
                        method=row.method,
                        model=row.model,
                        point_name=row.point_name,
                        score=score,
                        strategy=strategy,
                        status='pending',
                        reviewer=None,
                    ))
                    created += 1
            db.commit()
        except Exception as e:
            db.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            db.close()

        return jsonify({
            'success': True,
            'created': created,
            'skipped': skipped,
            'total': len(rows),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review/queue', methods=['GET'])
@login_required
def review_queue(current_user):
    """List review queue items with filters."""
    try:
        source_type = request.args.get('source_type')
        status = request.args.get('status')
        method = _normalize_method(request.args.get('method'))
        reviewer = request.args.get('reviewer')
        limit = request.args.get('limit', type=int) or 200

        from src.db.database import SessionLocal, ReviewQueue, InferenceResult

        db = SessionLocal()
        try:
            query = db.query(ReviewQueue, InferenceResult).outerjoin(
                InferenceResult, ReviewQueue.source_id == InferenceResult.id
            )
            if source_type:
                query = query.filter(ReviewQueue.source_type == source_type)
            if status:
                query = query.filter(ReviewQueue.status == status)
            if method:
                query = query.filter(ReviewQueue.method == method)
            if reviewer:
                query = query.filter(ReviewQueue.reviewer == reviewer)
            query = query.order_by(ReviewQueue.updated_at.desc())
            if limit and limit > 0:
                query = query.limit(limit)

            items = []
            for review, inference in query.all():
                result_path = inference.result_path if inference else None
                filename = os.path.basename(result_path) if result_path else None
                result_dir = os.path.dirname(result_path) if result_path else None
                if review.source_type == 'annotation':
                    if not filename or not result_dir:
                        candidates = _annotation_csv_candidates(review.source_id)
                        found = _find_csv_path_for_candidates(candidates)
                        if found:
                            result_path = str(found)
                            result_dir = str(found.parent)
                            filename = found.name
                        elif not filename:
                            filename = candidates[0] if candidates else review.source_id
                items.append({
                    'id': review.id,
                    'source_id': review.source_id,
                    'source_type': review.source_type,
                    'method': review.method,
                    'model': review.model,
                    'point_name': review.point_name,
                    'score': review.score,
                    'strategy': review.strategy,
                    'status': review.status,
                    'reviewer': review.reviewer,
                    'created_at': review.created_at.isoformat() if review.created_at else None,
                    'updated_at': review.updated_at.isoformat() if review.updated_at else None,
                    'result_path': result_path,
                    'result_dir': result_dir,
                    'filename': filename,
                })
        finally:
            db.close()

        return jsonify({'success': True, 'items': items})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review/queue/<item_id>', methods=['PATCH'])
@login_required
def review_queue_update(current_user, item_id):
    """Update review queue item status."""
    try:
        data = request.get_json(silent=True) or {}
        status = data.get('status')
        reviewer = data.get('reviewer') or current_user
        if status not in {'pending', 'approved', 'rejected', 'needs_fix'}:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400

        from src.db.database import SessionLocal, ReviewQueue

        db = SessionLocal()
        try:
            item = db.query(ReviewQueue).filter(ReviewQueue.id == item_id).first()
            if not item:
                return jsonify({'success': False, 'error': 'Item not found'}), 404
            item.status = status
            item.reviewer = reviewer
            item.updated_at = datetime.utcnow()
            db.commit()
        except Exception as e:
            db.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            db.close()

        return jsonify({'success': True, 'id': item_id, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review/queue/batch', methods=['PATCH'])
@login_required
def review_queue_batch_update(current_user):
    """Batch update review queue status."""
    try:
        data = request.get_json(silent=True) or {}
        status = data.get('status')
        reviewer = data.get('reviewer') or current_user
        ids = data.get('ids') or []
        if status not in {'pending', 'approved', 'rejected', 'needs_fix'}:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400
        if not isinstance(ids, list) or not ids:
            return jsonify({'success': False, 'error': 'No ids provided'}), 400

        from src.db.database import SessionLocal, ReviewQueue
        db = SessionLocal()
        try:
            updated = db.query(ReviewQueue).filter(ReviewQueue.id.in_(ids)).update(
                {
                    ReviewQueue.status: status,
                    ReviewQueue.reviewer: reviewer,
                    ReviewQueue.updated_at: datetime.utcnow(),
                },
                synchronize_session=False
            )
            db.commit()
        except Exception as e:
            db.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            db.close()

        return jsonify({'success': True, 'updated': updated, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review/stats', methods=['GET'])
@login_required
def review_stats(current_user):
    """Get review queue stats."""
    try:
        source_type = request.args.get('source_type')
        method = _normalize_method(request.args.get('method'))

        from src.db.database import SessionLocal, ReviewQueue
        from sqlalchemy import func as _sa_func

        db = SessionLocal()
        try:
            query = db.query(ReviewQueue.status, _sa_func.count(ReviewQueue.id))
            if source_type:
                query = query.filter(ReviewQueue.source_type == source_type)
            if method:
                query = query.filter(ReviewQueue.method == method)
            rows = query.group_by(ReviewQueue.status).all()
            stats = {status: count for status, count in rows}
            total = sum(stats.values())
        finally:
            db.close()

        return jsonify({
            'success': True,
            'stats': stats,
            'total': total
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/data/<filename>', methods=['GET'])
@login_required
def get_data(filename, current_user):
    """Read CSV or Excel file data with smart column detection and M4 downsampling"""
    try:
        from auth import load_users
        
        # Get user's data path
        users = load_users()
        user_path = users.get(current_user, {}).get('data_path', DATA_DIR)
        
        filepath = os.path.join(user_path, filename)
        
        # Level 3 Fallback: 如果文件不存在，尝试从文件名解析元数据并从 IoTDB 重新拉取
        if not os.path.exists(filepath):
            metadata = parse_standard_filename(filename)
            if metadata:
                print(f"[Level 3 Fallback] File not found, attempting to fetch from IoTDB: {filename}")
                try:
                    df = fetch_from_iotdb(metadata)
                    if df is not None and len(df) > 0:
                        # 缓存到本地
                        os.makedirs(user_path, exist_ok=True)
                        df.to_csv(filepath, index=True)
                        print(f"[Level 3 Fallback] Successfully fetched and cached: {filepath}")
                    else:
                        return jsonify({'success': False, 'error': 'File not found and IoTDB fetch failed'}), 404
                except Exception as e:
                    print(f"[Level 3 Fallback] IoTDB fetch error: {e}")
                    return jsonify({'success': False, 'error': f'File not found: {filename}'}), 404
            else:
                return jsonify({'success': False, 'error': 'File not found'}), 404
        
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath, engine='openpyxl' if file_ext == '.xlsx' else None)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format'}), 400
        
        original_len = len(df)
        columns = df.columns.tolist()
        
        # ============ Smart Column Detection (by data type) ============
        time_col = None
        val_col = None
        series_col = None
        label_col = None
        
        for col in columns:
            # Skip unnamed/index columns
            if col == '' or str(col).startswith('Unnamed'):
                continue
            
            # Detect datetime column
            if time_col is None:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    time_col = col
                elif df[col].dtype == 'object':
                    # Try parsing first few non-null values as datetime
                    sample = df[col].dropna().head(5)
                    if len(sample) > 0:
                        try:
                            pd.to_datetime(sample, errors='raise')
                            time_col = col
                        except:
                            pass
            
            # Detect numeric columns for value
            if pd.api.types.is_numeric_dtype(df[col]):
                if val_col is None:
                    val_col = col
            
            # Detect string columns that might be series/label
            elif df[col].dtype == 'object' and col != time_col:
                unique_count = df[col].nunique()
                if unique_count <= 10 and series_col is None:
                    series_col = col
                elif label_col is None:
                    label_col = col
        
        # Fallback: use second column as value if not detected
        if val_col is None and len(columns) >= 2:
            for col in columns:
                if col != time_col and not str(col).startswith('Unnamed'):
                    val_col = col
                    break
        
        # Fallback: use first column as value for single-column files
        if val_col is None and len(columns) >= 1:
            val_col = columns[-1]  # Use last column
        
        print(f"=== [DEBUG] get_data for {filename} ===")
        print(f"Columns: {columns}")
        print(f"[Column Detection] time={time_col}, value={val_col}, series={series_col}, label={label_col}")
        
        # ============ Parse Time Column ============
        use_index_mode = True  # Default to index mode
        if time_col is not None:
            try:
                # Use pandas to parse datetime with timezone support
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
                valid_times = df[time_col].notna().sum()
                if valid_times > 0:
                    use_index_mode = False
                    print(f"[Time Parse] Successfully parsed {valid_times}/{len(df)} time values")
            except Exception as e:
                print(f"[Time Parse] Failed: {e}, falling back to index mode")
        
        # ============ Prepare Value Array (handle NaN) ============
        if val_col is not None:
            df[val_col] = pd.to_numeric(df[val_col], errors='coerce').fillna(0.0)
        else:
            return jsonify({'success': False, 'error': 'No numeric value column found'}), 400
        
        # ============ M4 Downsampling (preserves min/max) ============
        MAX_ROWS = 10000
        downsampled = False
        if original_len > MAX_ROWS:
            try:
                # Prepare arrays for M4 downsampler
                x_arr = np.arange(original_len, dtype=np.float64)
                y_arr = df[val_col].values.astype(np.float64)
                
                # M4 returns positions that preserve local min/max
                downsampler = M4Downsampler()
                indices = downsampler.downsample(x_arr, y_arr, n_out=MAX_ROWS)
                
                # Keep original index after filtering
                df = df.iloc[indices] # Don't reset index here
                downsampled = True
                print(f"[M4 Downsampling] {original_len} rows -> {len(df)} rows")
            except Exception as e:
                print(f"[M4 Downsampling] Error: {e}, using uniform sampling fallback")
                step = original_len // MAX_ROWS
                df = df.iloc[::step].reset_index(drop=True)
                downsampled = True
        
        # ============ Build Response Data ============
        data = []
        series_set = set()
        
        for original_idx, row in df.iterrows():
            # Use the original DataFrame index (preserves link to raw file row)
            index_value = int(original_idx)
            
            # Value
            val_value = float(row[val_col]) if pd.notna(row[val_col]) else 0.0
            
            # Series
            if series_col and pd.notna(row[series_col]):
                series_value = str(row[series_col])
            else:
                series_value = val_col if val_col else 'value'
            series_set.add(series_value)
            
            # Label
            if label_col and pd.notna(row[label_col]):
                label_value = str(row[label_col])
            else:
                label_value = ''
            
            data.append({
                'idx': index_value,           # Original row index
                'time': index_value,          # Original row index
                'val': val_value,
                'series': series_value,
                'label': label_value
            })
        
        return jsonify({
            'success': True,
            'filename': filename,
            'columns': columns,
            'data': data,
            'seriesList': list(series_set),
            'labelList': [],
            'useIndexMode': True,           # Always use index mode for X-axis
            'originalLength': original_len,
            'downsampled': downsampled,
            'detectedColumns': {
                'time': time_col,
                'value': val_col,
                'series': series_col,
                'label': label_col
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


# ==================== User Authentication ====================
@app.route('/api/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400
        
        if verify_password(username, password):
            token = generate_token(username)
            users = load_users()
            user_info = users.get(username, {})
            
            return jsonify({
                'success': True,
                'token': token,
                'username': username,
                'name': user_info.get('name', username)
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user', methods=['GET'])
@login_required
def get_current_user(current_user):
    """Get current logged in user info"""
    try:
        users = load_users()
        user_info = users.get(current_user, {})
        return jsonify({
            'success': True,
            'username': current_user,
            'name': user_info.get('name', current_user)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Annotations (Multi-user Support) ====================
@app.route('/api/annotations/<filename>', methods=['GET'])
@login_required
def get_annotations(filename, current_user):
    """Get annotations for a file (user-specific)"""
    try:
        print(f"=== [DEBUG] get_annotations for {filename} (user: {current_user}) ===")
        with open('/tmp/debug_anno.log', 'a') as debug_f:
             debug_f.write(f"\n[DEBUG] Request for file: {filename}\n")
        
        # User-specific annotation directory
        user_ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
        os.makedirs(user_ann_dir, exist_ok=True)
        
        # Try multiple patterns to find the annotation file
        annotation_file = None
        annotation_data = None
        
        base_name = _normalize_annotation_name(filename)

        # Pattern 1: Standard format (filename.json)
        pattern1 = os.path.join(user_ann_dir, f"{base_name}.json")
        if os.path.exists(pattern1):
            annotation_file = pattern1
        elif base_name != filename:
            pattern1_alt = os.path.join(user_ann_dir, f"{filename}.json")
            if os.path.exists(pattern1_alt):
                annotation_file = pattern1_alt
        
        # Pattern 2 & 3: Search all JSON files and match by filename field
        if not annotation_file:
            for json_file in os.listdir(user_ann_dir):
                if not json_file.endswith('.json'):
                    continue
                json_path = os.path.join(user_ann_dir, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if this JSON's filename matches
                        if data.get('filename') == filename:
                            annotation_file = json_path
                            annotation_data = data
                            break
                except:
                    continue
        
        if not annotation_file:
            # Check for global_mask in CSV and auto-generate annotations
            auto_annotations = []
            try:
                from auth import load_users
                users = load_users()
                user_path = users.get(current_user, {}).get('data_path', DATA_DIR)
                csv_path = os.path.join(user_path, filename)
                
                if os.path.exists(csv_path):
                     try:
                         # Read only necessary columns to speed up
                         df_head = pd.read_csv(csv_path, nrows=1)
                         if 'global_mask' in df_head.columns:
                             df = pd.read_csv(csv_path, usecols=['global_mask'])
                             # Find continuous regions of 1s
                             mask = df['global_mask'].fillna(0).astype(int)
                             
                             # Logic to find start/end indices of 1s sequences
                             # 0 1 1 1 0 -> diff: 1 0 0 -1
                             # Pad with 0 to handle edge cases
                             padded = np.concatenate(([0], mask.values, [0]))
                             diff = np.diff(padded)
                             starts = np.where(diff == 1)[0]
                             ends = np.where(diff == -1)[0] - 1
                             
                             # Merge all segments into a single annotation
                             # Check algorithm type from filename
                             if 'qwen' in filename.lower():
                                 label_obj = {
                                     "id": "qwen_detected",
                                     "text": "Qwen",
                                     "color": "#6366f1",
                                     "categoryId": "algorithm_results"
                                 }
                                 algo_desc = "Qwen Auto-Detection"
                                 algo_id_suffix = "qwen"
                             else:
                                 label_obj = {
                                     "id": "chatts_detected",
                                     "text": "ChatTS",
                                     "color": "#ec4899",
                                     "categoryId": "algorithm_results"
                                 }
                                 algo_desc = "ChatTS Auto-Detection"
                                 algo_id_suffix = "chatts"
                             all_segments = []
                             for i, (start, end) in enumerate(zip(starts, ends)):
                                 all_segments.append({
                                     "start": int(start),
                                     "end": int(end),
                                     "count": int(end) - int(start) + 1,
                                     "label": label_obj
                                 })
                             
                             if all_segments:
                                 auto_annotations.append({
                                     "id": f"auto_{int(time.time())}_{algo_id_suffix}",
                                     "label": label_obj,
                                     "segments": all_segments,
                                     "local_change": {
                                          "trend": "其他",
                                          "confidence": "high",
                                          "desc": algo_desc
                                     }
                                 })
                             print(f"[DEBUG] Auto-generated 1 annotation with {len(all_segments)} segments from global_mask")
                             if all_segments:
                                print(f"[DEBUG] First segment sample: {all_segments[0]}")
                     except Exception as e:
                         print(f"Error reading CSV for mask: {e}")
            except Exception as e:
                print(f"Error in auto-annotation: {e}")


            return jsonify({
                'success': True,
                'filename': filename,
                'annotations': auto_annotations
            })
        
        # Load data if not already loaded
        if not annotation_data:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'annotations': annotation_data.get('annotations', []),
            'overall_attribute': annotation_data.get('overall_attribute', {})
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotations/<filename>', methods=['POST'])
@login_required
def save_annotations(filename, current_user):
    """Save annotations for a file (user-specific directory)"""
    try:
        from flask import request
        from auth import load_users
        
        data = request.get_json()
        
        # Support both old and new formats
        # New format: {filename, overall_attribute, annotations, export_time}
        # Old format: {annotations, overall_attributes}
        if 'filename' in data:
            # New unified format
            save_data = data
        else:
            # Old format - convert to new format
            save_data = {
                'filename': filename,
                'overall_attribute': data.get('overall_attributes', {}),
                'annotations': data.get('annotations', []),
                'export_time': datetime.now().isoformat()
            }
        
        # Save to user's annotation directory
        user_ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
        os.makedirs(user_ann_dir, exist_ok=True)
        
        base_name = _normalize_annotation_name(filename)
        annotation_file = os.path.join(user_ann_dir, f"{base_name}.json")
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Annotations saved for {filename}'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotations/<filename>', methods=['DELETE'])
def delete_annotation(filename):
    """Delete a specific annotation"""
    try:
        data = request.get_json()
        annotation_id = data.get('annotation_id')
        
        if not annotation_id:
            return jsonify({'success': False, 'error': 'Annotation ID is required'}), 400
        
        base_name = _normalize_annotation_name(filename)
        annotation_file = os.path.join(ANNOTATIONS_DIR, f"{base_name}.json")
        if not os.path.exists(annotation_file) and base_name != filename:
            annotation_file = os.path.join(ANNOTATIONS_DIR, f"{filename}.json")
        
        if not os.path.exists(annotation_file):
            return jsonify({'success': False, 'error': 'Annotation file not found'}), 404
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', [])
        annotations = [a for a in annotations if a.get('id') != annotation_id]
        
        data['annotations'] = annotations
        data['last_updated'] = datetime.now().isoformat()
        
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'message': 'Annotation deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download-annotations/<filename>', methods=['GET'])
def download_annotations(filename):
    """Download annotations in target JSON format"""
    try:
        base_name = _normalize_annotation_name(filename)
        annotation_file = os.path.join(ANNOTATIONS_DIR, f"{base_name}.json")
        if not os.path.exists(annotation_file) and base_name != filename:
            annotation_file = os.path.join(ANNOTATIONS_DIR, f"{filename}.json")
        
        if not os.path.exists(annotation_file):
            return jsonify({
                'annotations': [],
                'export_time': datetime.now().isoformat(),
                'filename': filename
            })
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format for export
        export_annotations = []
        for ann in data.get('annotations', []):
            export_ann = {
                'categories': ann.get('categories', {}),
                'local_change': ann.get('local_change', {})
            }
            export_annotations.append(export_ann)
        
        return jsonify({
            'annotations': export_annotations,
            'export_time': datetime.now().isoformat(),
            'filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotations/all', methods=['GET'])
@login_required
def get_all_annotations(current_user):
    """Get all annotations for current user (for training data export)"""
    try:
        user_ann_dir = os.path.join(ANNOTATIONS_DIR, current_user)
        
        if not os.path.exists(user_ann_dir):
            return jsonify({
                'success': True,
                'annotations': [],
                'count': 0
            })
        
        all_annotations = []
        
        for json_file in os.listdir(user_ann_dir):
            if not json_file.endswith('.json'):
                continue
            
            json_path = os.path.join(user_ann_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 确保包含必要字段
                    annotation_entry = {
                        'filename': data.get('filename', json_file.replace('.json', '')),
                        'annotations': data.get('annotations', []),
                        'overall_attribute': data.get('overall_attribute', {}),
                        'export_time': data.get('export_time', '')
                    }
                    all_annotations.append(annotation_entry)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'annotations': all_annotations,
            'count': len(all_annotations),
            'user': current_user
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Label Configuration ====================
@app.route('/api/labels', methods=['GET'])
def get_labels():
    """Get label configuration"""
    try:
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            return jsonify({'success': True, 'labels': labels})
        else:
            return jsonify({
                'success': True,
                'labels': {
                    'overall_attribute': {},
                    'local_change': {},
                    'custom_labels': []
                }
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/labels', methods=['POST'])
def save_labels():
    """Save label configuration"""
    try:
        data = request.get_json()
        
        with open(LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'message': 'Labels saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/labels/custom', methods=['POST'])
def add_custom_label():
    """Add a custom label"""
    try:
        data = request.get_json()
        label_text = data.get('label', '')
        label_color = data.get('color', '#3b82f6')
        
        if not label_text:
            return jsonify({'success': False, 'error': 'Label cannot be empty'}), 400
        
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        else:
            labels = {'overall_attribute': {}, 'local_change': {}, 'custom_labels': []}
        
        new_label = {
            'id': f'custom_{int(time.time() * 1000)}',
            'text': label_text,
            'color': label_color
        }
        
        existing_texts = [l.get('text', l) if isinstance(l, dict) else l 
                         for l in labels.get('custom_labels', [])]
        
        if label_text not in existing_texts:
            labels.setdefault('custom_labels', []).append(new_label)
            
            with open(LABELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'custom_labels': labels['custom_labels']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Time Series Annotator v2 - Backend Starting...")
    print("API Server: http://localhost:5000")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Annotations Directory: {ANNOTATIONS_DIR}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
