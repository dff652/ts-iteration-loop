"""
Task Center 状态机：DAG 依赖计算、重试/超时、控制参数管理。
"""
from datetime import datetime

from sqlalchemy.orm import Session

from src.db.database import Task, TaskCenterRun, TaskCenterStepRun
from src.models.schemas import TaskStatus
from src.utils.time_utils import utc_now_naive

from src.task_center.helpers import (
    _TASK_CENTER_META_KEY,
    _append_step_log,
    _float_value,
    _int_value,
    _json_dumps,
    _json_loads,
    _normalize_task_status,
    _parse_depends_on,
    _run_payload,
)


def _extract_run_control(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {
            "retry_attempt": 0,
            "max_retries": 0,
            "retry_policy": "fixed",
            "retry_delay_sec": 0,
            "retry_backoff_factor": 2.0,
            "retry_max_delay_sec": None,
            "retry_on_errors": [],
            "timeout_sec": None,
            "retry_of_run_id": "",
            "event_key": "",
            "event_dedupe_key": "",
        }
    raw_meta = payload.get(_TASK_CENTER_META_KEY)
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    retry_attempt = _int_value(raw_meta.get("retry_attempt"), default=0, min_value=0, max_value=100)
    max_retries = _int_value(raw_meta.get("max_retries"), default=0, min_value=0, max_value=10)
    retry_policy = str(raw_meta.get("retry_policy") or "fixed").strip().lower()
    if retry_policy not in {"fixed", "exponential"}:
        retry_policy = "fixed"
    retry_delay_sec = _int_value(raw_meta.get("retry_delay_sec"), default=0, min_value=0, max_value=3600)
    retry_backoff_factor = _float_value(raw_meta.get("retry_backoff_factor"), default=2.0, min_value=1.0, max_value=10.0)
    retry_max_delay_sec = _int_value(raw_meta.get("retry_max_delay_sec"), default=0, min_value=0, max_value=86400)
    if retry_max_delay_sec <= 0:
        retry_max_delay_sec = None
    retry_on_errors = []
    retry_on_errors_raw = raw_meta.get("retry_on_errors")
    if isinstance(retry_on_errors_raw, list):
        for item in retry_on_errors_raw:
            text = str(item or "").strip().lower()
            if text and text not in retry_on_errors:
                retry_on_errors.append(text)
    timeout_sec = _int_value(raw_meta.get("timeout_sec"), default=0, min_value=0, max_value=86400)
    if timeout_sec <= 0:
        timeout_sec = None
    retry_of_run_id = str(raw_meta.get("retry_of_run_id") or "").strip()
    event_key = str(raw_meta.get("event_key") or "").strip().lower()
    event_dedupe_key = str(raw_meta.get("event_dedupe_key") or "").strip()
    return {
        "retry_attempt": retry_attempt,
        "max_retries": max_retries,
        "retry_policy": retry_policy,
        "retry_delay_sec": retry_delay_sec,
        "retry_backoff_factor": retry_backoff_factor,
        "retry_max_delay_sec": retry_max_delay_sec,
        "retry_on_errors": retry_on_errors,
        "timeout_sec": timeout_sec,
        "retry_of_run_id": retry_of_run_id,
        "event_key": event_key,
        "event_dedupe_key": event_dedupe_key,
    }


def _merge_run_control(
    input_payload: dict | None,
    *,
    max_retries: int | None = None,
    retry_policy: str | None = None,
    retry_delay_sec: int | None = None,
    retry_backoff_factor: float | None = None,
    retry_max_delay_sec: int | None = None,
    retry_on_errors: list[str] | None = None,
    timeout_sec: int | None = None,
    retry_attempt: int | None = None,
    retry_of_run_id: str | None = None,
    event_key: str | None = None,
    event_dedupe_key: str | None = None,
) -> dict:
    payload = dict(input_payload or {})
    if _TASK_CENTER_META_KEY in payload and not isinstance(payload.get(_TASK_CENTER_META_KEY), dict):
        payload.pop(_TASK_CENTER_META_KEY, None)

    existing = _extract_run_control(payload)
    merged_max_retries = existing["max_retries"] if max_retries is None else _int_value(max_retries, default=0, min_value=0, max_value=10)
    policy = existing["retry_policy"] if retry_policy is None else str(retry_policy or "").strip().lower()
    if policy not in {"fixed", "exponential"}:
        policy = "fixed"
    merged_retry_delay_sec = existing["retry_delay_sec"] if retry_delay_sec is None else _int_value(retry_delay_sec, default=0, min_value=0, max_value=3600)
    merged_retry_backoff_factor = (
        existing["retry_backoff_factor"]
        if retry_backoff_factor is None
        else _float_value(retry_backoff_factor, default=2.0, min_value=1.0, max_value=10.0)
    )
    max_delay_base = (
        existing["retry_max_delay_sec"]
        if retry_max_delay_sec is None
        else _int_value(retry_max_delay_sec, default=0, min_value=0, max_value=86400)
    )
    merged_retry_max_delay_sec = max_delay_base if max_delay_base and max_delay_base > 0 else None
    if retry_on_errors is None:
        merged_retry_on_errors = existing["retry_on_errors"]
    else:
        merged_retry_on_errors = []
        for item in retry_on_errors:
            text = str(item or "").strip().lower()
            if text and text not in merged_retry_on_errors:
                merged_retry_on_errors.append(text)
    merged_retry_attempt = existing["retry_attempt"] if retry_attempt is None else _int_value(retry_attempt, default=0, min_value=0, max_value=100)
    timeout_base = existing["timeout_sec"] if timeout_sec is None else _int_value(timeout_sec, default=0, min_value=0, max_value=86400)
    merged_timeout_sec = timeout_base if timeout_base and timeout_base > 0 else None
    merged_retry_of = existing["retry_of_run_id"] if retry_of_run_id is None else str(retry_of_run_id or "").strip()
    merged_event_key = existing["event_key"] if event_key is None else str(event_key or "").strip().lower()
    merged_event_dedupe_key = existing["event_dedupe_key"] if event_dedupe_key is None else str(event_dedupe_key or "").strip()

    meta = {}
    if merged_max_retries > 0:
        meta["max_retries"] = merged_max_retries
    if policy != "fixed":
        meta["retry_policy"] = policy
    if merged_retry_delay_sec > 0:
        meta["retry_delay_sec"] = merged_retry_delay_sec
    if merged_retry_backoff_factor != 2.0:
        meta["retry_backoff_factor"] = merged_retry_backoff_factor
    if merged_retry_max_delay_sec:
        meta["retry_max_delay_sec"] = merged_retry_max_delay_sec
    if merged_retry_on_errors:
        meta["retry_on_errors"] = merged_retry_on_errors
    if merged_retry_attempt > 0:
        meta["retry_attempt"] = merged_retry_attempt
    if merged_timeout_sec:
        meta["timeout_sec"] = merged_timeout_sec
    if merged_retry_of:
        meta["retry_of_run_id"] = merged_retry_of
    if merged_event_key:
        meta["event_key"] = merged_event_key
    if merged_event_dedupe_key:
        meta["event_dedupe_key"] = merged_event_dedupe_key

    if meta:
        payload[_TASK_CENTER_META_KEY] = meta
    else:
        payload.pop(_TASK_CENTER_META_KEY, None)
    return payload


def _refresh_step_runnable_states(step_rows: list[TaskCenterStepRun]) -> bool:
    changed = False
    status_by_name: dict[str, str] = {}
    for step in step_rows:
        name = str(step.step_name or "").strip().lower()
        if name:
            status_by_name[name] = _normalize_task_status(step.status)

    for step in step_rows:
        current = _normalize_task_status(step.status)
        if current in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT, TaskStatus.RUNNING}:
            continue
        deps = _parse_depends_on(step)
        deps_ready = True
        for dep in deps:
            if status_by_name.get(dep) != TaskStatus.COMPLETED:
                deps_ready = False
                break
        target = TaskStatus.RUNNABLE if deps_ready else TaskStatus.BLOCKED
        if current != target:
            step.status = target
            changed = True
        status_by_name[str(step.step_name or "").strip().lower()] = target
    return changed


def _step_specs_from_rows(step_rows: list[TaskCenterStepRun]) -> list[dict]:
    specs: list[dict] = []
    for step in step_rows:
        name = str(step.step_name or "").strip().lower()
        if not name:
            continue
        specs.append({"name": name, "depends_on": _parse_depends_on(step)})
    return specs


def _latest_step_completed_at(step_rows: list[TaskCenterStepRun]) -> datetime | None:
    values = [step.completed_at for step in step_rows if step.completed_at]
    if not values:
        return None
    return max(values)


def _find_retry_child_run(db: Session, source_run: TaskCenterRun, retry_attempt: int) -> TaskCenterRun | None:
    q = db.query(TaskCenterRun).order_by(TaskCenterRun.created_at.desc())
    if source_run.definition_id:
        q = q.filter(TaskCenterRun.definition_id == source_run.definition_id)
    else:
        q = q.filter(TaskCenterRun.task_type == source_run.task_type)
    candidates = q.limit(300).all()
    for row in candidates:
        row_control = _extract_run_control(_run_payload(row))
        if row_control["retry_of_run_id"] == source_run.id and row_control["retry_attempt"] == retry_attempt:
            return row
    return None


def _retry_delay_for_next_attempt(control: dict) -> int:
    base_delay = _int_value(control.get("retry_delay_sec"), default=0, min_value=0, max_value=3600)
    policy = str(control.get("retry_policy") or "fixed").strip().lower()
    if policy == "exponential":
        attempt = _int_value(control.get("retry_attempt"), default=0, min_value=0, max_value=100)
        factor = _float_value(control.get("retry_backoff_factor"), default=2.0, min_value=1.0, max_value=10.0)
        computed = int(round(base_delay * (factor ** attempt)))
    else:
        computed = base_delay
    max_delay = control.get("retry_max_delay_sec")
    if isinstance(max_delay, int) and max_delay > 0:
        computed = min(computed, max_delay)
    return max(0, computed)


def _should_retry_by_error(run: TaskCenterRun, control: dict) -> bool:
    keywords = control.get("retry_on_errors") if isinstance(control.get("retry_on_errors"), list) else []
    if not keywords:
        return True
    run_status = _normalize_task_status(run.status)
    text = str(run.error or "").strip().lower()
    if run_status == TaskStatus.TIMEOUT:
        text = f"timeout {text}".strip()
    for key in keywords:
        token = str(key or "").strip().lower()
        if token and token in text:
            return True
    return False


def _step_legacy_task(db: Session, step: TaskCenterStepRun) -> Task | None:
    step_result = _json_loads(step.result, {})
    if not isinstance(step_result, dict):
        return None
    legacy_task_id = str(step_result.get("legacy_task_id") or "").strip()
    if not legacy_task_id:
        return None
    return db.query(Task).filter(Task.id == legacy_task_id).first()


def _maybe_timeout_run(db: Session, run: TaskCenterRun, step_rows: list[TaskCenterStepRun]) -> bool:
    control = _extract_run_control(_run_payload(run))
    timeout_sec = control["timeout_sec"]
    if not timeout_sec or not run.started_at:
        return False

    now = utc_now_naive()
    if (now - run.started_at).total_seconds() < timeout_sec:
        return False

    changed = False
    terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}
    for step in step_rows:
        step_status = _normalize_task_status(step.status)
        if step_status in terminal:
            continue
        step.status = TaskStatus.TIMEOUT
        step.completed_at = step.completed_at or now
        step.message = f"step timeout after {timeout_sec}s"
        _append_step_log(step, f"[{now.isoformat()}] step timeout after {timeout_sec}s")
        legacy_task = _step_legacy_task(db, step)
        if legacy_task and _normalize_task_status(legacy_task.status) not in terminal:
            legacy_task.status = TaskStatus.TIMEOUT
            legacy_task.completed_at = now
        changed = True
    if changed:
        run.error = f"run timeout after {timeout_sec}s"
    return changed


def _maybe_auto_retry_run(db: Session, run: TaskCenterRun, step_rows: list[TaskCenterStepRun]) -> None:
    run_status = _normalize_task_status(run.status)
    if run_status not in {TaskStatus.FAILED, TaskStatus.TIMEOUT}:
        return
    if not step_rows:
        return

    control = _extract_run_control(_run_payload(run))
    max_retries = control["max_retries"]
    retry_attempt = control["retry_attempt"]
    if max_retries <= 0 or retry_attempt >= max_retries:
        return
    if not _should_retry_by_error(run, control):
        return

    now = utc_now_naive()
    due_after = _retry_delay_for_next_attempt(control)
    base_time = run.completed_at or now
    if (now - base_time).total_seconds() < due_after:
        return

    next_attempt = retry_attempt + 1
    if _find_retry_child_run(db, run, next_attempt):
        return

    # 延迟导入避免循环依赖
    from src.task_center.run_operations import _create_run_record, _execute_run_impl

    new_run = _create_run_record(
        db,
        definition_id=run.definition_id,
        task_type=run.task_type,
        trigger_mode=run.trigger_mode,
        input_payload=_run_payload(run),
        steps=None,
        step_specs=_step_specs_from_rows(step_rows),
        max_retries=max_retries,
        retry_policy=control["retry_policy"],
        retry_delay_sec=control["retry_delay_sec"],
        retry_backoff_factor=control["retry_backoff_factor"],
        retry_max_delay_sec=control["retry_max_delay_sec"],
        retry_on_errors=control["retry_on_errors"],
        timeout_sec=control["timeout_sec"],
        retry_attempt=next_attempt,
        retry_of_run_id=run.id,
    )
    _execute_run_impl(db, new_run, simulate=False)
