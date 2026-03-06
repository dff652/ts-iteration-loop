"""
Task Center 事件处理：事件触发、Webhook、死信、速率限制。
"""
import hmac
import json
import threading
import time
import uuid
from hashlib import sha256
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from configs.settings import settings
from src.db.database import TaskCenterDefinition, TaskCenterRun
from src.models.schemas import ApiResponse, TaskCenterEventTriggerRequest
from src.utils.time_utils import utc_now_naive

from src.task_center.helpers import (
    _TASK_CENTER_META_KEY,
    _int_value,
    _json_dumps,
    _json_loads,
)

# ==================== 模块级状态 ====================
_EVENT_METRICS_LOCK = threading.Lock()
_EVENT_RATE_LIMIT_LOCK = threading.Lock()
_EVENT_METRICS: dict[str, int] = {}
_EVENT_RATE_LIMIT_STATE: dict[str, tuple[int, int]] = {}


# ==================== 指标 ====================
def _metric_inc(name: str, delta: int = 1) -> None:
    key = str(name or "").strip().lower()
    if not key:
        return
    with _EVENT_METRICS_LOCK:
        _EVENT_METRICS[key] = int(_EVENT_METRICS.get(key, 0)) + int(delta)


def _metrics_snapshot() -> dict[str, int]:
    with _EVENT_METRICS_LOCK:
        return {k: int(v) for k, v in _EVENT_METRICS.items()}


# ==================== 死信 ====================
def _dead_letter_file_path() -> Path:
    raw = str(getattr(settings, "TASK_CENTER_DEAD_LETTER_PATH", "") or "").strip()
    if not raw:
        raw = "data/task_center_dead_letter.jsonl"
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_dead_letter_rows() -> list[dict]:
    path = _dead_letter_file_path()
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line or "").strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _record_dead_letter(source: str, request_payload: dict, error: str, context: dict | None = None) -> str:
    event_id = str(uuid.uuid4())
    row = {
        "id": event_id,
        "source": str(source or "").strip().lower(),
        "event_key": str(request_payload.get("event_key") or "").strip().lower(),
        "definition_id": str(request_payload.get("definition_id") or "").strip(),
        "dedupe_key": str(request_payload.get("dedupe_key") or "").strip(),
        "execute_mode": str(request_payload.get("execute_mode") or "").strip().lower(),
        "request": request_payload,
        "error": str(error or "").strip(),
        "created_at": utc_now_naive().isoformat(),
    }
    if isinstance(context, dict) and context:
        row["context"] = context
    path = _dead_letter_file_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _metric_inc("dead_letter_total", 1)
    return event_id


def _dead_letter_view(limit: int, offset: int) -> tuple[int, list[dict]]:
    rows = _read_dead_letter_rows()
    rows.reverse()
    total = len(rows)
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    return total, rows[start:end]


def _find_dead_letter_row(event_id: str) -> dict | None:
    target = str(event_id or "").strip()
    if not target:
        return None
    rows = _read_dead_letter_rows()
    for row in reversed(rows):
        if str(row.get("id") or "").strip() == target:
            return row
    return None


# ==================== 速率限制 ====================
def _rate_limit_per_min() -> int:
    return _int_value(getattr(settings, "TASK_CENTER_EVENT_RATE_LIMIT_PER_MIN", 300), default=300, min_value=0, max_value=1000000)


def _events_enabled() -> bool:
    return bool(getattr(settings, "TASK_CENTER_EVENTS_ENABLED", False))


def _ensure_events_enabled() -> None:
    if _events_enabled():
        return
    raise HTTPException(status_code=404, detail="事件触发能力未启用（TASK_CENTER_EVENTS_ENABLED=false）")


def _check_event_rate_limit(bucket: str) -> tuple[bool, int, int]:
    limit = _rate_limit_per_min()
    if limit <= 0:
        return True, 0, 0
    now_minute = int(time.time() // 60)
    key = str(bucket or "").strip().lower() or "default"
    with _EVENT_RATE_LIMIT_LOCK:
        state = _EVENT_RATE_LIMIT_STATE.get(key)
        if not state or int(state[0]) != now_minute:
            count = 0
        else:
            count = int(state[1])
        if count >= limit:
            return False, count, limit
        count += 1
        _EVENT_RATE_LIMIT_STATE[key] = (now_minute, count)
        return True, count, limit


# ==================== Webhook ====================
def _webhook_secrets() -> dict[str, str]:
    raw = str(getattr(settings, "TASK_CENTER_WEBHOOK_SECRETS", "") or "").strip()
    if not raw:
        return {}
    items: dict[str, str] = {}
    for token in raw.split(","):
        part = str(token or "").strip()
        if not part:
            continue
        if "=" in part:
            key, secret = part.split("=", 1)
            k = str(key or "").strip().lower()
            v = str(secret or "").strip()
            if k and v:
                items[k] = v
        else:
            items["default"] = part
    return items


def _webhook_secret_for_source(source: str) -> str:
    mapping = _webhook_secrets()
    key = str(source or "").strip().lower()
    if key and key in mapping:
        return mapping[key]
    return str(mapping.get("default") or "")


def _normalize_hmac_signature(signature: str | None) -> str:
    text = str(signature or "").strip()
    if text.lower().startswith("sha256="):
        text = text.split("=", 1)[1]
    return text.strip().lower()


def _verify_webhook_signature(secret: str, body: bytes, signature: str | None) -> bool:
    sec = str(secret or "").strip()
    if not sec:
        return True
    provided = _normalize_hmac_signature(signature)
    if not provided:
        return False
    expected = hmac.new(sec.encode("utf-8"), body, sha256).hexdigest().lower()
    return hmac.compare_digest(expected, provided)


# ==================== 事件去重 ====================
def _find_event_dedupe_run(db: Session, definition_id: str, event_key: str, dedupe_key: str) -> TaskCenterRun | None:
    if not definition_id or not dedupe_key:
        return None

    from src.task_center.helpers import _run_payload
    from src.task_center.state_machine import _extract_run_control

    rows = (
        db.query(TaskCenterRun)
        .filter(TaskCenterRun.definition_id == definition_id)
        .order_by(TaskCenterRun.created_at.desc())
        .limit(300)
        .all()
    )
    event_key_norm = str(event_key or "").strip().lower()
    dedupe_key_norm = str(dedupe_key or "").strip()
    for row in rows:
        control = _extract_run_control(_run_payload(row))
        if control["event_key"] != event_key_norm:
            continue
        if control["event_dedupe_key"] == dedupe_key_norm:
            return row
    return None


# ==================== 事件触发核心 ====================
def _trigger_event_impl(db: Session, request: TaskCenterEventTriggerRequest) -> ApiResponse:
    _ensure_events_enabled()
    event_key = str(request.event_key or "").strip().lower()
    execute_mode = str(request.execute_mode or "dispatch").strip().lower()
    _metric_inc("event_trigger_requests_total", 1)
    allowed, current, limit = _check_event_rate_limit(f"event:{event_key}")
    if not allowed:
        _metric_inc("event_trigger_rate_limited_total", 1)
        raise HTTPException(status_code=429, detail=f"事件触发超过限流阈值: {current}/{limit} per min")

    if request.definition_id:
        definition = (
            db.query(TaskCenterDefinition)
            .filter(TaskCenterDefinition.id == request.definition_id)
            .first()
        )
        if not definition:
            raise HTTPException(status_code=404, detail="任务定义不存在")
        if not bool(definition.enabled):
            raise HTTPException(status_code=409, detail="任务定义已禁用")
        if str(definition.trigger_mode or "").strip().lower() != "event":
            raise HTTPException(status_code=409, detail="任务定义不是 event 触发模式")
        definitions = [definition]
    else:
        definitions = (
            db.query(TaskCenterDefinition)
            .filter(
                TaskCenterDefinition.enabled == True,  # noqa: E712
                TaskCenterDefinition.trigger_mode == "event",
            )
            .order_by(TaskCenterDefinition.created_at.asc())
            .all()
        )

    # 延迟导入避免循环依赖
    from src.task_center.run_operations import _create_run_record, _execute_run_impl

    created_run_ids: list[str] = []
    deduped_definition_ids: list[str] = []
    skipped_definition_ids: list[str] = []
    rate_limited_definition_ids: list[str] = []
    dispatched_count = 0
    simulated_count = 0

    for definition in definitions:
        cfg = _json_loads(definition.config, {})
        if not isinstance(cfg, dict):
            cfg = {}
        cfg_event_key = str(cfg.get("event_key") or "").strip().lower()
        if cfg_event_key and cfg_event_key != event_key:
            skipped_definition_ids.append(definition.id)
            continue

        dedupe_key = str(request.dedupe_key or "").strip()
        if dedupe_key and _find_event_dedupe_run(db, definition.id, event_key, dedupe_key):
            deduped_definition_ids.append(definition.id)
            continue

        payload = {}
        base_payload = cfg.get("input_payload")
        if isinstance(base_payload, dict):
            payload.update(base_payload)
        if isinstance(request.payload, dict):
            payload.update(request.payload)

        # 并发检查
        from src.models.schemas import TaskStatus
        max_concurrent = cfg.get("max_concurrent_runs")
        if isinstance(max_concurrent, int) and max_concurrent > 0:
            running_count = (
                db.query(TaskCenterRun.id)
                .filter(
                    TaskCenterRun.definition_id == definition.id,
                    TaskCenterRun.status.in_([TaskStatus.PENDING, TaskStatus.RUNNABLE, TaskStatus.RUNNING, TaskStatus.BLOCKED]),
                )
                .count()
            )
            if running_count >= max_concurrent:
                rate_limited_definition_ids.append(definition.id)
                # Event 触发如果不希望被丢弃，也可以仅 create_run 然后不分发，让 sweeper 兜底，
                # 但更安全的做法是直接拦截以保护后端。我们将返回信息告知客户端
                continue

        run = _create_run_record(
            db,
            definition_id=definition.id,
            task_type=str(cfg.get("task_type") or definition.task_type or "acquire_inference"),
            trigger_mode="event",
            input_payload=payload,
            steps=cfg.get("steps") if isinstance(cfg.get("steps"), list) else None,
            step_specs=cfg.get("step_specs") if isinstance(cfg.get("step_specs"), list) else None,
            max_retries=cfg.get("max_retries"),
            retry_policy=cfg.get("retry_policy"),
            retry_delay_sec=cfg.get("retry_delay_sec"),
            retry_backoff_factor=cfg.get("retry_backoff_factor"),
            retry_max_delay_sec=cfg.get("retry_max_delay_sec"),
            retry_on_errors=cfg.get("retry_on_errors") if isinstance(cfg.get("retry_on_errors"), list) else None,
            timeout_sec=cfg.get("timeout_sec"),
            event_key=event_key,
            event_dedupe_key=dedupe_key or None,
        )
        created_run_ids.append(run.id)

        if execute_mode == "simulate":
            _execute_run_impl(db, run, simulate=True)
            dispatched_count += 1
            simulated_count += 1
        elif execute_mode == "dispatch":
            _execute_run_impl(db, run, simulate=False)
            dispatched_count += 1

    if created_run_ids:
        _metric_inc("event_trigger_runs_created_total", len(created_run_ids))
    if deduped_definition_ids:
        _metric_inc("event_trigger_deduped_total", len(deduped_definition_ids))
    if skipped_definition_ids:
        _metric_inc("event_trigger_skipped_total", len(skipped_definition_ids))
    if rate_limited_definition_ids:
        _metric_inc("event_trigger_rate_limited_total", len(rate_limited_definition_ids))

    return ApiResponse(
        success=True,
        data={
            "event_key": event_key,
            "execute_mode": execute_mode,
            "matched_definitions": len(definitions),
            "created_count": len(created_run_ids),
            "dispatched_count": dispatched_count,
            "simulated_count": simulated_count,
            "run_ids": created_run_ids,
            "deduped_definition_ids": deduped_definition_ids,
            "skipped_definition_ids": skipped_definition_ids,
            "rate_limited_definition_ids": rate_limited_definition_ids,
        },
        message=f"事件触发完成：创建 {len(created_run_ids)} 个运行实例，限流拦截 {len(rate_limited_definition_ids)} 个",
    )
