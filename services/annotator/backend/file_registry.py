"""
Simple SQLite-backed file registry for annotator.
"""

import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional


def _ensure_db_dir(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)


def init_db(db_path: str) -> None:
    _ensure_db_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                dir_path TEXT,
                filename TEXT,
                method TEXT,
                size_bytes INTEGER,
                mtime REAL,
                updated_at REAL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_files_dir ON files(dir_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_files_method ON files(method)")
        conn.commit()


def infer_method_from_filename(filename: str) -> Optional[str]:
    name = filename.lower()
    if "qwen" in name:
        return "qwen"
    if "chatts" in name:
        return "chatts"
    if "timer" in name:
        return "timer"
    if "adtk_hbos" in name:
        return "adtk_hbos"
    if "ensemble" in name:
        return "ensemble"
    return None


def sync_directory(db_path: str, dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        return

    allowed_exts = (".csv", ".xls", ".xlsx")
    entries = []
    for name in os.listdir(dir_path):
        if not name.lower().endswith(allowed_exts):
            continue
        full_path = os.path.join(dir_path, name)
        if not os.path.isfile(full_path):
            continue
        try:
            stat = os.stat(full_path)
        except OSError:
            continue

        entries.append({
            "path": full_path,
            "dir_path": dir_path,
            "filename": name,
            "method": infer_method_from_filename(name),
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "updated_at": time.time(),
        })

    with sqlite3.connect(db_path) as conn:
        # Upsert current files
        for e in entries:
            conn.execute(
                """
                INSERT INTO files (path, dir_path, filename, method, size_bytes, mtime, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    dir_path=excluded.dir_path,
                    filename=excluded.filename,
                    method=excluded.method,
                    size_bytes=excluded.size_bytes,
                    mtime=excluded.mtime,
                    updated_at=excluded.updated_at
                """,
                (
                    e["path"], e["dir_path"], e["filename"], e["method"],
                    e["size_bytes"], e["mtime"], e["updated_at"],
                ),
            )

        # Remove stale files for this directory
        existing = conn.execute(
            "SELECT path FROM files WHERE dir_path = ?",
            (dir_path,),
        ).fetchall()
        existing_paths = {row[0] for row in existing}
        current_paths = {e["path"] for e in entries}
        stale = existing_paths - current_paths
        if stale:
            conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in stale])

        conn.commit()


def list_files(db_path: str, dir_path: str, method: Optional[str] = None) -> List[Dict]:
    if not os.path.isdir(dir_path):
        return []

    with sqlite3.connect(db_path) as conn:
        if method:
            rows = conn.execute(
                """
                SELECT filename, method, mtime
                FROM files
                WHERE dir_path = ? AND method = ?
                ORDER BY mtime DESC
                """,
                (dir_path, method),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT filename, method, mtime
                FROM files
                WHERE dir_path = ?
                ORDER BY mtime DESC
                """,
                (dir_path,),
            ).fetchall()

    return [
        {"name": r[0], "method": r[1], "mtime": r[2]}
        for r in rows
    ]


def rebuild_directory(db_path: str, dir_path: str) -> None:
    """Clear and rebuild registry entries for a directory."""
    if not os.path.isdir(dir_path):
        return
    _ensure_db_dir(db_path)
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM files WHERE dir_path = ?", (dir_path,))
        conn.commit()
    sync_directory(db_path, dir_path)
