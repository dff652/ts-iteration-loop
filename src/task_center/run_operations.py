"""
Task Center Run 核心操作：创建/执行/同步/索引。
"""
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from src.db.database import (
    InferenceResult,
    Task,
    TaskCenterResultIndex,
    TaskCenterRun,
    TaskCenterStepRun,
)
from src.models.schemas import TaskStatus
from src.utils.time_utils import utc_now_naive

from src.task_center.helpers import (
    _append_step_log,
    _dt_iso,
    _json_dumps,
    _json_loads,
    _normalize_task_status,
    _parse_depends_on,
    _run_payload,
)
from src.task_center.state_machine import (
    _extract_run_control,
    _latest_step_completed_at,
    _maybe_auto_retry_run,
    _maybe_timeout_run,
    _merge_run_control,
    _refresh_step_runnable_states,
    _step_legacy_task,
    _step_specs_from_rows,
)
from src.task_center.helpers import _build_step_specs


def _ensure_task_center_index_from_inference(
    db: Session,
    run_id: str,
    legacy_task_id: str,
) -> None:
    has_existing = (
        db.query(TaskCenterResultIndex)
        .filter(TaskCenterResultIndex.run_id == run_id)
        .count()
    ) > 0
    if has_existing:
        return
    rows = (
        db.query(InferenceResult)
        .filter(InferenceResult.task_id == legacy_task_id)
        .order_by(InferenceResult.created_at.desc())
        .all()
    )
    for row in rows:
        db.add(
            TaskCenterResultIndex(
                run_id=run_id,
                point_id=row.point_id or "",
                model_version=row.model or "",
                result_path=row.result_path or "",
                status=TaskStatus.COMPLETED,
                meta=_json_dumps(
                    {
                        "source": "inference_results",
                        "legacy_task_id": legacy_task_id,
                        "inference_result_id": row.id,
                    }
                ),
            )
        )


def _list_step_payloads(db: Session, run_id: str) -> list[dict]:
    rows = (
        db.query(TaskCenterStepRun)
        .filter(TaskCenterStepRun.run_id == run_id)
        .order_by(TaskCenterStepRun.id.asc())
        .all()
    )
    payload: list[dict] = []
    for row in rows:
        payload.append(
            {
                "id": row.id,
                "step_name": row.step_name,
                "status": row.status,
                "depends_on": _parse_depends_on(row),
                "message": row.message or "",
                "started_at": _dt_iso(row.started_at),
                "completed_at": _dt_iso(row.completed_at),
            }
        )
    return payload


def _create_run_record(
    db: Session,
    *,
    definition_id: str | None,
    task_type: str,
    trigger_mode: str,
    input_payload: dict | None,
    steps: list[str] | None,
    step_specs: list[dict] | None,
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
    created_at: datetime | None = None,
) -> TaskCenterRun:
    payload_with_control = _merge_run_control(
        input_payload,
        max_retries=max_retries,
        retry_policy=retry_policy,
        retry_delay_sec=retry_delay_sec,
        retry_backoff_factor=retry_backoff_factor,
        retry_max_delay_sec=retry_max_delay_sec,
        retry_on_errors=retry_on_errors,
        timeout_sec=timeout_sec,
        retry_attempt=retry_attempt,
        retry_of_run_id=retry_of_run_id,
        event_key=event_key,
        event_dedupe_key=event_dedupe_key,
    )
    run = TaskCenterRun(
        id=str(uuid.uuid4()),
        definition_id=definition_id,
        task_type=task_type,
        trigger_mode=trigger_mode,
        status=TaskStatus.PENDING,
        input_payload=_json_dumps(payload_with_control),
        created_at=created_at or utc_now_naive(),
    )
    db.add(run)
    db.flush()

    resolved_specs = _build_step_specs(step_specs, steps)
    for spec in resolved_specs:
        step_name = str(spec.get("name") or "").strip().lower()
        depends_on = spec.get("depends_on") if isinstance(spec.get("depends_on"), list) else []
        status = TaskStatus.RUNNABLE if not depends_on else TaskStatus.BLOCKED
        db.add(
            TaskCenterStepRun(
                run_id=run.id,
                step_name=step_name,
                status=status,
                depends_on=_json_dumps(depends_on),
            )
        )

    db.commit()
    db.refresh(run)
    _sync_run_from_legacy_tasks(db, run)
    return run


def _sync_run_from_legacy_tasks(db: Session, run: TaskCenterRun) -> None:
    step_rows = (
        db.query(TaskCenterStepRun)
        .filter(TaskCenterStepRun.run_id == run.id)
        .order_by(TaskCenterStepRun.id.asc())
        .all()
    )
    if not step_rows:
        return

    changed = False
    run_errors: list[str] = []

    for step in step_rows:
        step_result = _json_loads(step.result, {})
        if not isinstance(step_result, dict):
            step_result = {}
        legacy_task_id = str(step_result.get("legacy_task_id") or "").strip()

        if legacy_task_id:
            legacy_task = db.query(Task).filter(Task.id == legacy_task_id).first()
            if legacy_task:
                normalized = _normalize_task_status(legacy_task.status)
                current = _normalize_task_status(step.status)
                # Legacy task 刚入队时通常仍为 pending，不应覆盖已分发 step 的 running 状态。
                if normalized == TaskStatus.PENDING and current == TaskStatus.RUNNING:
                    normalized = TaskStatus.RUNNING
                if current != normalized:
                    step.status = normalized
                    changed = True
                if normalized in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}:
                    if not step.completed_at and legacy_task.completed_at:
                        step.completed_at = legacy_task.completed_at
                        changed = True
                    elif not step.completed_at:
                        step.completed_at = utc_now_naive()
                        changed = True
                    if normalized == TaskStatus.COMPLETED and str(step.step_name or "").lower() == "inference":
                        _ensure_task_center_index_from_inference(db, run.id, legacy_task_id)
                        changed = True
                    if normalized == TaskStatus.FAILED:
                        if legacy_task.error:
                            run_errors.append(str(legacy_task.error))
                        elif legacy_task.result:
                            run_errors.append(str(legacy_task.result))

    if _refresh_step_runnable_states(step_rows):
        changed = True
    if _maybe_timeout_run(db, run, step_rows):
        changed = True

    statuses = [_normalize_task_status(step.status) for step in step_rows]
    status_set = set(statuses)
    has_failed = TaskStatus.FAILED in status_set
    has_timeout = TaskStatus.TIMEOUT in status_set
    has_running = TaskStatus.RUNNING in status_set
    has_pending = TaskStatus.PENDING in status_set
    has_runnable = TaskStatus.RUNNABLE in status_set
    has_blocked = TaskStatus.BLOCKED in status_set
    has_cancelled = TaskStatus.CANCELLED in status_set
    all_done = all(
        status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}
        for status in statuses
    )

    if has_failed:
        if _normalize_task_status(run.status) != TaskStatus.FAILED:
            run.status = TaskStatus.FAILED
            changed = True
        run.error = "; ".join([e for e in run_errors if e][:3])
        if not run.completed_at:
            run.completed_at = _latest_step_completed_at(step_rows) or utc_now_naive()
            changed = True
    elif has_timeout:
        if _normalize_task_status(run.status) != TaskStatus.TIMEOUT:
            run.status = TaskStatus.TIMEOUT
            changed = True
        if not run.completed_at:
            run.completed_at = _latest_step_completed_at(step_rows) or utc_now_naive()
            changed = True
    elif all_done and has_cancelled:
        if _normalize_task_status(run.status) != TaskStatus.CANCELLED:
            run.status = TaskStatus.CANCELLED
            changed = True
        if not run.completed_at:
            run.completed_at = _latest_step_completed_at(step_rows) or utc_now_naive()
            changed = True
    elif all_done:
        if _normalize_task_status(run.status) != TaskStatus.COMPLETED:
            run.status = TaskStatus.COMPLETED
            changed = True
        if not run.completed_at:
            run.completed_at = _latest_step_completed_at(step_rows) or utc_now_naive()
            changed = True
        run.result = _json_dumps(
            {
                "success": True,
                "run_id": run.id,
                "step_count": len(step_rows),
            }
        )
        changed = True
    elif has_running:
        if _normalize_task_status(run.status) != TaskStatus.RUNNING:
            run.status = TaskStatus.RUNNING
            changed = True
        if not run.started_at:
            run.started_at = utc_now_naive()
            changed = True
    elif has_runnable:
        if _normalize_task_status(run.status) != TaskStatus.RUNNABLE:
            run.status = TaskStatus.RUNNABLE
            changed = True
    elif has_pending:
        if _normalize_task_status(run.status) != TaskStatus.PENDING:
            run.status = TaskStatus.PENDING
            changed = True
    elif has_blocked:
        if _normalize_task_status(run.status) != TaskStatus.BLOCKED:
            run.status = TaskStatus.BLOCKED
            changed = True

    if changed:
        db.commit()
        db.refresh(run)
    _maybe_auto_retry_run(db, run, step_rows)


def _execute_run_impl(db: Session, run: TaskCenterRun, simulate: bool) -> TaskCenterRun:
    if run.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}:
        return run

    _sync_run_from_legacy_tasks(db, run)
    if run.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}:
        return run

    step_rows = (
        db.query(TaskCenterStepRun)
        .filter(TaskCenterStepRun.run_id == run.id)
        .order_by(TaskCenterStepRun.id.asc())
        .all()
    )
    if not step_rows:
        for i, step_name in enumerate(["acquire", "inference"]):
            depends_on = [] if i == 0 else ["acquire"]
            step_rows.append(
                TaskCenterStepRun(
                    run_id=run.id,
                    step_name=step_name,
                    status=TaskStatus.RUNNABLE if not depends_on else TaskStatus.BLOCKED,
                    depends_on=_json_dumps(depends_on),
                )
            )
        db.add_all(step_rows)
        db.flush()

    if _refresh_step_runnable_states(step_rows):
        db.commit()
        db.refresh(run)

    now = utc_now_naive()
    if not run.started_at:
        run.started_at = now
    run.status = TaskStatus.RUNNING

    if not simulate:
        from src.task_center.dispatch import _dispatch_step

        dispatched = 0
        for step in step_rows:
            status_now = _normalize_task_status(step.status)
            if status_now not in {TaskStatus.RUNNABLE, TaskStatus.PENDING}:
                continue
            try:
                _dispatch_step(db, run, step)
                dispatched += 1
            except Exception as exc:
                step.status = TaskStatus.FAILED
                step.completed_at = utc_now_naive()
                step.message = f"dispatch failed: {exc}"
                _append_step_log(step, f"[{utc_now_naive().isoformat()}] dispatch failed: {exc}")
                run.status = TaskStatus.FAILED
                run.error = str(exc)
                run.completed_at = utc_now_naive()
                db.commit()
                db.refresh(run)
                return run
        if dispatched == 0:
            _sync_run_from_legacy_tasks(db, run)
            return run
        db.commit()
        db.refresh(run)
        return run

    input_payload = _json_loads(run.input_payload, {}) if isinstance(run.input_payload, str) else {}

    for step in step_rows:
        step_started = utc_now_naive()
        step.status = TaskStatus.RUNNING
        step.started_at = step_started
        step.logs = ((step.logs or "") + f"[{step_started.isoformat()}] step={step.step_name} started\n").strip()

        step_done = utc_now_naive()
        step.status = TaskStatus.COMPLETED
        step.completed_at = step_done
        step.message = f"{step.step_name} completed (simulated)"
        step.logs = (step.logs + f"\n[{step_done.isoformat()}] step={step.step_name} completed").strip()
        step.result = _json_dumps(
            {
                "success": True,
                "simulated": True,
                "step_name": step.step_name,
            }
        )

    run.status = TaskStatus.COMPLETED
    run.completed_at = utc_now_naive()
    run.result = _json_dumps(
        {
            "success": True,
            "simulated": True,
            "run_id": run.id,
            "step_count": len(step_rows),
        }
    )

    has_index = (
        db.query(TaskCenterResultIndex)
        .filter(TaskCenterResultIndex.run_id == run.id)
        .count()
    ) > 0
    if not has_index:
        db.add(
            TaskCenterResultIndex(
                run_id=run.id,
                point_id=str(input_payload.get("point_id") or ""),
                model_version=str(input_payload.get("model_version") or "simulated"),
                result_path=str(input_payload.get("result_path") or ""),
                status=TaskStatus.COMPLETED,
                meta=_json_dumps({"source": "task_center_simulated"}),
            )
        )

    db.commit()
    db.refresh(run)
    return run
