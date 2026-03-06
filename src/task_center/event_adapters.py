"""
Task Center 事件源适配器（Webhook 以外的拉模式入口）。

当前实现：
- 文件到达扫描（file_watch）
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from configs.settings import settings
from src.api import task_center as task_center_api
from src.core.logging_config import get_logger
from src.db.database import SessionLocal, TaskCenterDefinition
from src.models.schemas import TaskCenterEventTriggerRequest


logger = get_logger(__name__)


@dataclass
class FileEventScanStats:
    scanned_definitions: int = 0
    scanned_files: int = 0
    created_runs: int = 0
    deduped: int = 0
    moved_files: int = 0
    deleted_files: int = 0
    post_action_errors: int = 0
    skipped: int = 0
    errors: int = 0


def _is_file_watch_definition(cfg: dict) -> bool:
    source = str(cfg.get("source") or cfg.get("event_source") or cfg.get("adapter") or "").strip().lower()
    return source in {"file_watch", "file", "filesystem"}


def _safe_json_file(path: Path) -> dict | None:
    try:
        raw = path.read_text(encoding="utf-8")
        loaded = json.loads(raw)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _archive_target_path(archive_dir: Path, file_path: Path) -> Path:
    target = archive_dir / file_path.name
    if not target.exists():
        return target
    stem = file_path.stem
    suffix = file_path.suffix
    ts = int(time.time())
    candidate = archive_dir / f"{stem}.{ts}{suffix}"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = archive_dir / f"{stem}.{ts}.{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _archive_base_dir(cfg: dict, file_path: Path) -> Path:
    base = Path(str(cfg.get("archive_dir") or "").strip())
    if not bool(cfg.get("archive_by_date")):
        return base
    st = file_path.stat()
    day = time.strftime("%Y-%m-%d", time.localtime(st.st_mtime))
    return base / day


def _apply_post_action(file_path: Path, cfg: dict, stats: FileEventScanStats) -> None:
    action = str(cfg.get("post_action") or "keep").strip().lower()
    if action in {"", "keep", "none"}:
        return

    if action == "move":
        archive_dir_raw = str(cfg.get("archive_dir") or "").strip()
        if not archive_dir_raw:
            stats.post_action_errors += 1
            logger.warning("file_watch move skipped: archive_dir missing | file=%s", file_path)
            return
        archive_dir = _archive_base_dir(cfg, file_path)
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = _archive_target_path(archive_dir, file_path)
        shutil.move(str(file_path), str(target))
        stats.moved_files += 1
        return

    if action == "delete":
        file_path.unlink()
        stats.deleted_files += 1
        return

    stats.post_action_errors += 1
    logger.warning("file_watch unknown post_action=%s | file=%s", action, file_path)


def run_file_event_scan(execute_mode: str | None = None) -> FileEventScanStats:
    mode = str(execute_mode or getattr(settings, "TASK_CENTER_EXECUTION_MODE", "dispatch")).strip().lower()
    stats = FileEventScanStats()

    db = SessionLocal()
    try:
        definitions = (
            db.query(TaskCenterDefinition)
            .filter(
                TaskCenterDefinition.enabled == True,  # noqa: E712
                TaskCenterDefinition.trigger_mode == "event",
            )
            .order_by(TaskCenterDefinition.created_at.asc())
            .all()
        )
        for definition in definitions:
            stats.scanned_definitions += 1
            cfg = task_center_api._json_loads(definition.config, {})  # noqa: SLF001
            if not isinstance(cfg, dict):
                cfg = {}
            if not _is_file_watch_definition(cfg):
                stats.skipped += 1
                continue

            watch_dir = Path(str(cfg.get("watch_dir") or "").strip())
            if not str(watch_dir):
                stats.errors += 1
                continue
            pattern = str(cfg.get("file_glob") or "*.json").strip() or "*.json"
            event_key = str(cfg.get("event_key") or "file.arrived").strip().lower()
            include_json_payload = bool(cfg.get("include_json_payload"))

            files = sorted([p for p in watch_dir.glob(pattern) if p.is_file()])
            stats.scanned_files += len(files)

            for file_path in files:
                req = None
                payload: dict = {}
                dedupe_key = ""
                try:
                    st = file_path.stat()
                    dedupe_key = f"{file_path.resolve()}:{int(st.st_mtime)}:{st.st_size}"
                    payload = {
                        "file_path": str(file_path.resolve()),
                        "file_name": file_path.name,
                        "file_size": int(st.st_size),
                        "file_mtime": int(st.st_mtime),
                    }
                    if include_json_payload:
                        loaded = _safe_json_file(file_path)
                        if loaded is not None:
                            payload["file_payload"] = loaded

                    req = TaskCenterEventTriggerRequest(
                        event_key=event_key,
                        payload=payload,
                        definition_id=definition.id,
                        dedupe_key=dedupe_key,
                        execute_mode=mode,
                    )
                    resp = task_center_api._trigger_event_impl(db, req)  # noqa: SLF001
                    data = resp.data or {}
                    created_count = int(data.get("created_count") or 0)
                    stats.created_runs += created_count
                    deduped_ids = data.get("deduped_definition_ids")
                    deduped_count = 0
                    if isinstance(deduped_ids, list):
                        deduped_count = len(deduped_ids)
                        stats.deduped += deduped_count
                    if created_count > 0 or deduped_count > 0:
                        _apply_post_action(file_path, cfg, stats)
                except Exception as exc:
                    stats.errors += 1
                    try:
                        req_payload = req.model_dump() if req is not None else {
                            "event_key": event_key,
                            "definition_id": definition.id,
                            "execute_mode": mode,
                            "payload": payload,
                            "dedupe_key": dedupe_key,
                        }
                        task_center_api._record_dead_letter(  # noqa: SLF001
                            "file_watch",
                            req_payload,
                            str(exc),
                            {"file_path": str(file_path)},
                        )
                    except Exception:
                        pass
                    logger.exception("file event trigger failed | definition_id=%s | file=%s", definition.id, file_path)
    finally:
        db.close()

    return stats


def run_file_event_loop() -> None:
    interval = int(getattr(settings, "TASK_CENTER_FILE_WATCH_INTERVAL_SEC", 15) or 15)
    interval = max(2, interval)
    logger.info("Task Center file event loop started | interval=%ss", interval)
    while True:
        try:
            stats = run_file_event_scan()
            logger.info(
                "file event scan | definitions=%s files=%s created=%s deduped=%s moved=%s deleted=%s post_action_errors=%s skipped=%s errors=%s",
                stats.scanned_definitions,
                stats.scanned_files,
                stats.created_runs,
                stats.deduped,
                stats.moved_files,
                stats.deleted_files,
                stats.post_action_errors,
                stats.skipped,
                stats.errors,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("file event scan failed: %s", exc, exc_info=True)
        time.sleep(interval)


if __name__ == "__main__":
    run_file_event_loop()
