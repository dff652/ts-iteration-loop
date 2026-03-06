"""
任务中心 API（新增模块，旧模块保持不变）

本文件仅包含 API 路由定义和向后兼容的重导出。
业务逻辑已拆分至 src/task_center/ 子模块：
  - helpers.py       — 工具函数、序列化、值解析
  - state_machine.py — 状态机流转、DAG 依赖、重试/超时
  - dispatch.py      — 执行器分发（acquire/inference/training）
  - events.py        — 事件触发、Webhook、死信、速率限制
  - run_operations.py — Run 创建/执行/同步/索引
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from configs.settings import settings
from src.core.tasks import celery_app  # noqa: F401 — 测试通过 task_center_api.celery_app 访问
from src.db.database import (
    get_db,
    TaskCenterDefinition,
    TaskCenterRun,
    TaskCenterStepRun,
    TaskCenterResultIndex,
)
from src.models.schemas import (
    ApiResponse,
    TaskCenterDeadLetterReplayRequest,
    TaskCenterDefinitionRequest,
    TaskCenterEventTriggerRequest,
    TaskCenterRunCreateRequest,
    TaskCenterRunExecuteRequest,
    TaskStatus,
)

# ==================== 从子模块重导出所有符号 ====================
# 保持向后兼容：scheduler.py / event_adapters.py / tests 通过
# src.api.task_center 访问这些函数和变量

from src.task_center.helpers import (  # noqa: F401
    _TASK_CENTER_META_KEY,
    _json_dumps,
    _json_loads,
    _dt_iso,
    _int_value,
    _float_value,
    _normalize_steps,
    _normalize_task_status,
    _append_step_log,
    _run_payload,
    _build_run_message,
    _parse_depends_on,
    _build_step_specs,
)
from src.task_center.state_machine import (  # noqa: F401
    _extract_run_control,
    _merge_run_control,
    _refresh_step_runnable_states,
    _step_specs_from_rows,
    _latest_step_completed_at,
    _find_retry_child_run,
    _retry_delay_for_next_attempt,
    _should_retry_by_error,
    _step_legacy_task,
    _maybe_timeout_run,
    _maybe_auto_retry_run,
)
from src.task_center.dispatch import (  # noqa: F401
    _dispatch_inference_legacy_task,
    _dispatch_training_legacy_task,
    _dispatch_acquire_legacy_task,
    _dispatch_step,
)
from src.task_center.events import (  # noqa: F401
    _EVENT_METRICS_LOCK,
    _EVENT_RATE_LIMIT_LOCK,
    _EVENT_METRICS,
    _EVENT_RATE_LIMIT_STATE,
    _metric_inc,
    _metrics_snapshot,
    _dead_letter_file_path,
    _read_dead_letter_rows,
    _record_dead_letter,
    _dead_letter_view,
    _find_dead_letter_row,
    _rate_limit_per_min,
    _events_enabled,
    _ensure_events_enabled,
    _check_event_rate_limit,
    _webhook_secrets,
    _webhook_secret_for_source,
    _normalize_hmac_signature,
    _verify_webhook_signature,
    _find_event_dedupe_run,
    _trigger_event_impl,
)
from src.task_center.run_operations import (  # noqa: F401
    _ensure_task_center_index_from_inference,
    _list_step_payloads,
    _create_run_record,
    _sync_run_from_legacy_tasks,
    _execute_run_impl,
)
from src.utils.time_utils import utc_now_naive  # noqa: F401 — 测试通过 task_center_api.utc_now_naive 访问

# ==================== API 路由 ====================
router = APIRouter()


@router.post("/definitions", response_model=ApiResponse)
async def create_definition(request: TaskCenterDefinitionRequest, db: Session = Depends(get_db)):
    if str(request.trigger_mode.value or "").strip().lower() == "event" and not _events_enabled():
        raise HTTPException(status_code=409, detail="事件触发定义未启用，请设置 TASK_CENTER_EVENTS_ENABLED=true")

    definition = TaskCenterDefinition(
        id=str(uuid.uuid4()),
        name=request.name.strip(),
        task_type=request.task_type,
        trigger_mode=request.trigger_mode.value,
        schedule_cron=request.schedule_cron,
        enabled=bool(request.enabled),
        config=_json_dumps(request.config),
        created_by=request.created_by or settings.DEFAULT_USER,
        updated_by=request.created_by or settings.DEFAULT_USER,
    )
    db.add(definition)
    db.commit()
    db.refresh(definition)

    return ApiResponse(
        success=True,
        data={
            "id": definition.id,
            "name": definition.name,
            "task_type": definition.task_type,
            "trigger_mode": definition.trigger_mode,
            "schedule_cron": definition.schedule_cron,
            "enabled": bool(definition.enabled),
            "config": _json_loads(definition.config, {}),
            "created_at": _dt_iso(definition.created_at),
        },
        message="任务定义已创建",
    )


@router.get("/definitions", response_model=ApiResponse)
async def list_definitions(db: Session = Depends(get_db)):
    rows = db.query(TaskCenterDefinition).order_by(TaskCenterDefinition.created_at.desc()).all()
    data = []
    for row in rows:
        if str(row.trigger_mode or "").strip().lower() == "event" and not _events_enabled():
            continue
        data.append(
            {
                "id": row.id,
                "name": row.name,
                "task_type": row.task_type,
                "trigger_mode": row.trigger_mode,
                "schedule_cron": row.schedule_cron,
                "enabled": bool(row.enabled),
                "config": _json_loads(row.config, {}),
                "created_at": _dt_iso(row.created_at),
            }
        )
    return ApiResponse(success=True, data={"definitions": data}, message=f"共 {len(data)} 条")


@router.post("/events/trigger", response_model=ApiResponse)
async def trigger_event(request: TaskCenterEventTriggerRequest, db: Session = Depends(get_db)):
    try:
        return _trigger_event_impl(db, request)
    except HTTPException as exc:
        _metric_inc("event_trigger_http_error_total", 1)
        if int(exc.status_code) >= 500:
            _record_dead_letter(
                "api_trigger",
                request.model_dump(),
                f"http_error:{exc.status_code}:{exc.detail}",
                {"status_code": int(exc.status_code)},
            )
        raise
    except Exception as exc:
        _metric_inc("event_trigger_exception_total", 1)
        dead_id = _record_dead_letter("api_trigger", request.model_dump(), str(exc))
        raise HTTPException(status_code=500, detail=f"事件触发失败，已写入死信: {dead_id}") from exc


@router.post("/events/webhook/{source}", response_model=ApiResponse)
async def trigger_event_webhook(source: str, request: Request, db: Session = Depends(get_db)):
    _ensure_events_enabled()
    body = await request.body()
    secret = _webhook_secret_for_source(source)
    signature = request.headers.get(getattr(settings, "TASK_CENTER_WEBHOOK_SIGNATURE_HEADER", "X-TaskCenter-Signature"))
    if signature is None:
        signature = request.headers.get("X-TaskCenter-Signature")
    if not _verify_webhook_signature(secret, body, signature):
        _metric_inc("event_webhook_auth_failed_total", 1)
        raise HTTPException(status_code=401, detail="Webhook signature 校验失败")

    payload = _json_loads(body.decode("utf-8"), {}) if body else {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Webhook body 必须是 JSON object")

    event_key = str(payload.get("event_key") or source).strip().lower()
    definition_id = str(payload.get("definition_id") or "").strip() or None
    dedupe_key = (
        str(payload.get("dedupe_key") or "").strip()
        or str(request.headers.get("X-Event-Id") or "").strip()
        or str(payload.get("event_id") or "").strip()
        or None
    )
    execute_mode = str(payload.get("execute_mode") or "dispatch").strip().lower()

    event_payload_raw = payload.get("payload")
    if isinstance(event_payload_raw, dict):
        event_payload = dict(event_payload_raw)
    else:
        event_payload = {
            k: v
            for k, v in payload.items()
            if k not in {"event_key", "definition_id", "dedupe_key", "event_id", "execute_mode", "payload"}
        }
    event_payload["event_source"] = str(source or "").strip().lower()

    trigger_req = TaskCenterEventTriggerRequest(
        event_key=event_key,
        payload=event_payload,
        definition_id=definition_id,
        dedupe_key=dedupe_key,
        execute_mode=execute_mode,
    )
    try:
        return _trigger_event_impl(db, trigger_req)
    except HTTPException:
        _metric_inc("event_webhook_http_error_total", 1)
        raise
    except Exception as exc:
        _metric_inc("event_webhook_exception_total", 1)
        dead_id = _record_dead_letter(
            f"webhook:{source}",
            trigger_req.model_dump(),
            str(exc),
            {"header_event_id": str(request.headers.get("X-Event-Id") or "").strip()},
        )
        raise HTTPException(status_code=500, detail=f"Webhook 处理失败，已写入死信: {dead_id}") from exc


@router.get("/events/dead-letters", response_model=ApiResponse)
async def list_dead_letters(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    _ensure_events_enabled()
    total, items = _dead_letter_view(limit, offset)
    return ApiResponse(
        success=True,
        data={
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
        },
        message=f"共 {total} 条，返回 {len(items)} 条",
    )


@router.post("/events/dead-letters/replay", response_model=ApiResponse)
async def replay_dead_letter(
    request: TaskCenterDeadLetterReplayRequest,
    db: Session = Depends(get_db),
):
    _ensure_events_enabled()
    event_id = str(request.event_id or "").strip()
    if not event_id:
        raise HTTPException(status_code=422, detail="event_id 不能为空")
    row = _find_dead_letter_row(event_id)
    if not row:
        raise HTTPException(status_code=404, detail="死信事件不存在")

    req_raw = row.get("request")
    if not isinstance(req_raw, dict):
        raise HTTPException(status_code=409, detail="死信数据缺少 request 内容，无法重放")

    event_key = str(req_raw.get("event_key") or "").strip().lower()
    if not event_key:
        raise HTTPException(status_code=409, detail="死信数据缺少 event_key，无法重放")

    trigger_req = TaskCenterEventTriggerRequest(
        event_key=event_key,
        payload=req_raw.get("payload") if isinstance(req_raw.get("payload"), dict) else {},
        definition_id=str(req_raw.get("definition_id") or "").strip() or None,
        dedupe_key=str(req_raw.get("dedupe_key") or "").strip() or None,
        execute_mode=str(request.execute_mode or "dispatch").strip().lower(),
    )

    try:
        resp = _trigger_event_impl(db, trigger_req)
        _metric_inc("dead_letter_replay_success_total", 1)
        return ApiResponse(
            success=True,
            data={
                "event_id": event_id,
                "execute_mode": trigger_req.execute_mode,
                "result": resp.data or {},
            },
            message="死信重放已执行",
        )
    except HTTPException:
        _metric_inc("dead_letter_replay_http_error_total", 1)
        raise
    except Exception as exc:
        _metric_inc("dead_letter_replay_exception_total", 1)
        new_dead_id = _record_dead_letter("dead_letter_replay", trigger_req.model_dump(), str(exc), {"source_event_id": event_id})
        raise HTTPException(status_code=500, detail=f"死信重放失败，已写入新死信: {new_dead_id}") from exc


@router.get("/events/metrics", response_model=ApiResponse)
async def get_event_metrics():
    _ensure_events_enabled()
    metrics = _metrics_snapshot()
    dead_letter_rows = _read_dead_letter_rows()
    data = {
        "counters": metrics,
        "dead_letter_total": len(dead_letter_rows),
        "rate_limit_per_min": _rate_limit_per_min(),
        "rate_limit_bucket_count": len(_EVENT_RATE_LIMIT_STATE),
    }
    return ApiResponse(success=True, data=data, message="ok")


@router.post("/runs", response_model=ApiResponse)
async def create_run(request: TaskCenterRunCreateRequest, db: Session = Depends(get_db)):
    cfg = {}
    if request.definition_id:
        definition = (
            db.query(TaskCenterDefinition)
            .filter(TaskCenterDefinition.id == request.definition_id)
            .first()
        )
        if not definition:
            raise HTTPException(status_code=404, detail="任务定义不存在")
        if definition.config:
            cfg = _json_loads(definition.config, {})

    payload = {}
    base_payload = cfg.get("input_payload")
    if isinstance(base_payload, dict):
        payload.update(base_payload)
    if isinstance(request.input_payload, dict):
        payload.update(request.input_payload)

    run = _create_run_record(
        db,
        definition_id=request.definition_id,
        task_type=request.task_type,
        trigger_mode=request.trigger_mode.value,
        input_payload=payload,
        steps=request.steps or cfg.get("steps") if isinstance(cfg.get("steps"), list) else request.steps,
        step_specs=(
            request.step_specs if isinstance(request.step_specs, list) 
            else cfg.get("step_specs") if isinstance(cfg.get("step_specs"), list) 
            else None
        ),
        max_retries=request.max_retries if request.max_retries is not None else cfg.get("max_retries"),
        retry_policy=request.retry_policy if request.retry_policy is not None else cfg.get("retry_policy"),
        retry_delay_sec=request.retry_delay_sec if request.retry_delay_sec is not None else cfg.get("retry_delay_sec"),
        retry_backoff_factor=request.retry_backoff_factor if request.retry_backoff_factor is not None else cfg.get("retry_backoff_factor"),
        retry_max_delay_sec=request.retry_max_delay_sec if request.retry_max_delay_sec is not None else cfg.get("retry_max_delay_sec"),
        retry_on_errors=(
            request.retry_on_errors if isinstance(request.retry_on_errors, list)
            else cfg.get("retry_on_errors") if isinstance(cfg.get("retry_on_errors"), list)
            else None
        ),
        timeout_sec=request.timeout_sec if request.timeout_sec is not None else cfg.get("timeout_sec"),
    )
    if request.auto_execute:
        run = _execute_run_impl(db, run, simulate=True)

    status_text = str(run.status or "").lower()
    control = _extract_run_control(_run_payload(run))
    return ApiResponse(
        success=True,
        data={
            "run_id": run.id,
            "definition_id": run.definition_id,
            "task_type": run.task_type,
            "trigger_mode": run.trigger_mode,
            "status": status_text,
            "retry_attempt": control["retry_attempt"],
            "max_retries": control["max_retries"],
            "retry_policy": control["retry_policy"],
            "retry_delay_sec": control["retry_delay_sec"],
            "retry_backoff_factor": control["retry_backoff_factor"],
            "retry_max_delay_sec": control["retry_max_delay_sec"],
            "retry_on_errors": control["retry_on_errors"],
            "timeout_sec": control["timeout_sec"],
            "event_key": control["event_key"],
            "event_dedupe_key": control["event_dedupe_key"],
            "steps": _list_step_payloads(db, run.id),
            "created_at": _dt_iso(run.created_at),
        },
        message="任务运行实例已创建",
    )


@router.get("/runs", response_model=ApiResponse)
async def list_runs(
    run_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    trigger_mode: str | None = Query(default=None),
    definition_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    # Direct function calls in tests may keep FastAPI Query objects as defaults.
    run_id = run_id if isinstance(run_id, str) else None
    status = status if isinstance(status, str) else None
    task_type = task_type if isinstance(task_type, str) else None
    trigger_mode = trigger_mode if isinstance(trigger_mode, str) else None
    definition_id = definition_id if isinstance(definition_id, str) else None
    limit = int(limit) if isinstance(limit, int) else 20
    offset = int(offset) if isinstance(offset, int) else 0

    q = db.query(TaskCenterRun)
    if run_id:
        q = q.filter(TaskCenterRun.id == str(run_id).strip())
    if status:
        q = q.filter(TaskCenterRun.status == str(status).strip().lower())
    if task_type:
        q = q.filter(TaskCenterRun.task_type == str(task_type).strip())
    if trigger_mode:
        q = q.filter(TaskCenterRun.trigger_mode == str(trigger_mode).strip().lower())
    if definition_id:
        q = q.filter(TaskCenterRun.definition_id == str(definition_id).strip())

    total = q.count()
    rows = q.order_by(TaskCenterRun.created_at.desc()).offset(offset).limit(limit).all()

    data_rows = []
    for row in rows:
        _sync_run_from_legacy_tasks(db, row)
        control = _extract_run_control(_run_payload(row))
        step_rows = (
            db.query(TaskCenterStepRun)
            .filter(TaskCenterStepRun.run_id == row.id)
            .order_by(TaskCenterStepRun.id.asc())
            .all()
        )
        step_status_counts: dict[str, int] = {}
        for step in step_rows:
            s = _normalize_task_status(step.status)
            step_status_counts[s] = int(step_status_counts.get(s, 0)) + 1
        data_rows.append(
            {
                "run_id": row.id,
                "definition_id": row.definition_id,
                "task_type": row.task_type,
                "trigger_mode": row.trigger_mode,
                "status": _normalize_task_status(row.status),
                "error": row.error or "",
                "retry_attempt": control["retry_attempt"],
                "max_retries": control["max_retries"],
                "retry_policy": control["retry_policy"],
                "retry_delay_sec": control["retry_delay_sec"],
                "retry_backoff_factor": control["retry_backoff_factor"],
                "retry_max_delay_sec": control["retry_max_delay_sec"],
                "retry_on_errors": control["retry_on_errors"],
                "timeout_sec": control["timeout_sec"],
                "event_key": control["event_key"],
                "event_dedupe_key": control["event_dedupe_key"],
                "step_count": len(step_rows),
                "step_status_counts": step_status_counts,
                "created_at": _dt_iso(row.created_at),
                "started_at": _dt_iso(row.started_at),
                "completed_at": _dt_iso(row.completed_at),
            }
        )

    return ApiResponse(
        success=True,
        data={
            "total": total,
            "limit": limit,
            "offset": offset,
            "runs": data_rows,
        },
        message=f"共 {total} 条，返回 {len(data_rows)} 条",
    )


@router.post("/runs/{run_id}/execute", response_model=ApiResponse)
async def execute_run(
    run_id: str,
    request: TaskCenterRunExecuteRequest,
    db: Session = Depends(get_db),
):
    run = db.query(TaskCenterRun).filter(TaskCenterRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="任务运行实例不存在")

    run = _execute_run_impl(db, run, simulate=bool(request.simulate))
    status_text = str(run.status or "").lower()
    control = _extract_run_control(_run_payload(run))
    return ApiResponse(
        success=True,
        data={
            "run_id": run.id,
            "status": status_text,
            "retry_attempt": control["retry_attempt"],
            "max_retries": control["max_retries"],
            "retry_policy": control["retry_policy"],
            "retry_delay_sec": control["retry_delay_sec"],
            "retry_backoff_factor": control["retry_backoff_factor"],
            "retry_max_delay_sec": control["retry_max_delay_sec"],
            "retry_on_errors": control["retry_on_errors"],
            "timeout_sec": control["timeout_sec"],
            "event_key": control["event_key"],
            "event_dedupe_key": control["event_dedupe_key"],
            "started_at": _dt_iso(run.started_at),
            "completed_at": _dt_iso(run.completed_at),
            "steps": _list_step_payloads(db, run.id),
            "simulated": bool(request.simulate),
        },
        message=_build_run_message(status_text),
    )


@router.post("/runs/{run_id}/cancel", response_model=ApiResponse)
async def cancel_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(TaskCenterRun).filter(TaskCenterRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="任务运行实例不存在")

    _sync_run_from_legacy_tasks(db, run)
    status_now = _normalize_task_status(run.status)
    if status_now in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}:
        return ApiResponse(
            success=True,
            data={
                "run_id": run.id,
                "status": status_now,
                "cancelled_steps": 0,
                "revoke_errors": [],
            },
            message="任务已结束",
        )

    step_rows = (
        db.query(TaskCenterStepRun)
        .filter(TaskCenterStepRun.run_id == run.id)
        .order_by(TaskCenterStepRun.id.asc())
        .all()
    )

    revoke_errors: list[str] = []
    cancelled_steps = 0
    from src.utils.time_utils import utc_now_naive as _utc_now
    now = _utc_now()

    for step in step_rows:
        step_status = _normalize_task_status(step.status)
        if step_status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}:
            continue

        legacy_task = _step_legacy_task(db, step)
        if legacy_task:
            cfg = _json_loads(legacy_task.config, {})
            if not isinstance(cfg, dict):
                cfg = {}
            celery_task_id = str(cfg.get("celery_task_id") or "").strip()
            if celery_task_id:
                try:
                    celery_app.control.revoke(celery_task_id, terminate=True)
                except Exception as exc:  # pragma: no cover - best effort logging branch
                    revoke_errors.append(f"step={step.id}, celery_task_id={celery_task_id}, err={exc}")
                    _append_step_log(step, f"[{now.isoformat()}] revoke failed celery_task_id={celery_task_id}: {exc}")
            legacy_task.status = TaskStatus.CANCELLED
            legacy_task.completed_at = now

        step.status = TaskStatus.CANCELLED
        step.completed_at = now
        step.message = "step cancelled by user"
        _append_step_log(step, f"[{now.isoformat()}] step cancelled by user")
        cancelled_steps += 1

    run.status = TaskStatus.CANCELLED
    run.completed_at = now
    db.commit()
    db.refresh(run)

    return ApiResponse(
        success=True,
        data={
            "run_id": run.id,
            "status": str(run.status or "").lower(),
            "cancelled_steps": cancelled_steps,
            "revoke_errors": revoke_errors,
        },
        message="任务已取消",
    )


@router.post("/runs/{run_id}/retry", response_model=ApiResponse)
async def retry_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(TaskCenterRun).filter(TaskCenterRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="任务运行实例不存在")

    _sync_run_from_legacy_tasks(db, run)
    status_now = _normalize_task_status(run.status)
    if status_now in {TaskStatus.PENDING, TaskStatus.RUNNING}:
        raise HTTPException(status_code=409, detail="任务仍在执行中，不能重试")
    if status_now == TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="任务已完成，不能重试")

    old_steps = (
        db.query(TaskCenterStepRun)
        .filter(TaskCenterStepRun.run_id == run.id)
        .order_by(TaskCenterStepRun.id.asc())
        .all()
    )
    if not old_steps:
        raise HTTPException(status_code=409, detail="原任务无步骤定义，不能重试")

    source_payload = _run_payload(run)
    source_control = _extract_run_control(source_payload)
    track_retry_chain = source_control["max_retries"] > 0 or source_control["retry_attempt"] > 0
    retry_attempt = source_control["retry_attempt"] + 1 if track_retry_chain else None

    new_run = _create_run_record(
        db,
        definition_id=run.definition_id,
        task_type=run.task_type,
        trigger_mode=run.trigger_mode,
        input_payload=source_payload,
        steps=None,
        step_specs=_step_specs_from_rows(old_steps),
        max_retries=source_control["max_retries"] if source_control["max_retries"] > 0 else None,
        retry_policy=source_control["retry_policy"],
        retry_delay_sec=source_control["retry_delay_sec"],
        retry_backoff_factor=source_control["retry_backoff_factor"],
        retry_max_delay_sec=source_control["retry_max_delay_sec"],
        retry_on_errors=source_control["retry_on_errors"],
        timeout_sec=source_control["timeout_sec"],
        retry_attempt=retry_attempt,
        retry_of_run_id=run.id if track_retry_chain else None,
    )

    return ApiResponse(
        success=True,
        data={
            "source_run_id": run.id,
            "new_run_id": new_run.id,
            "status": str(new_run.status or "").lower(),
            "retry_attempt": _extract_run_control(_run_payload(new_run))["retry_attempt"],
            "steps": _list_step_payloads(db, new_run.id),
        },
        message="重试任务已创建",
    )


@router.get("/runs/{run_id}/status", response_model=ApiResponse)
async def get_run_status(run_id: str, db: Session = Depends(get_db)):
    run = db.query(TaskCenterRun).filter(TaskCenterRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="任务运行实例不存在")

    _sync_run_from_legacy_tasks(db, run)
    status_text = str(run.status or "").lower()
    control = _extract_run_control(_run_payload(run))
    return ApiResponse(
        success=True,
        data={
            "run_id": run.id,
            "definition_id": run.definition_id,
            "task_type": run.task_type,
            "trigger_mode": run.trigger_mode,
            "status": status_text,
            "error": run.error or "",
            "retry_attempt": control["retry_attempt"],
            "max_retries": control["max_retries"],
            "retry_policy": control["retry_policy"],
            "retry_delay_sec": control["retry_delay_sec"],
            "retry_backoff_factor": control["retry_backoff_factor"],
            "retry_max_delay_sec": control["retry_max_delay_sec"],
            "retry_on_errors": control["retry_on_errors"],
            "timeout_sec": control["timeout_sec"],
            "event_key": control["event_key"],
            "event_dedupe_key": control["event_dedupe_key"],
            "steps": _list_step_payloads(db, run.id),
            "created_at": _dt_iso(run.created_at),
            "started_at": _dt_iso(run.started_at),
            "completed_at": _dt_iso(run.completed_at),
        },
        message=_build_run_message(status_text),
    )


@router.get("/runs/{run_id}/log", response_model=ApiResponse)
async def get_run_log(
    run_id: str,
    offset: int = 0,
    max_bytes: int = 200000,
    db: Session = Depends(get_db),
):
    run = db.query(TaskCenterRun).filter(TaskCenterRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="任务运行实例不存在")

    _sync_run_from_legacy_tasks(db, run)
    if offset < 0:
        offset = 0
    if max_bytes <= 0:
        max_bytes = 1
    max_bytes = min(max_bytes, 2_000_000)

    log_rows: list[str] = []
    step_rows = (
        db.query(TaskCenterStepRun)
        .filter(TaskCenterStepRun.run_id == run_id)
        .order_by(TaskCenterStepRun.id.asc())
        .all()
    )
    for step in step_rows:
        header = f"[{step.step_name}] status={step.status}"
        log_rows.append(header)
        if step.message:
            log_rows.append(f"[{step.step_name}] message={step.message}")
        if step.logs:
            log_rows.append(str(step.logs))
    if run.error:
        log_rows.append(f"[run] error={run.error}")

    log_text = "\n".join(log_rows).strip()
    total = len(log_text)
    safe_offset = min(offset, total)
    chunk = log_text[safe_offset : safe_offset + max_bytes]
    new_offset = safe_offset + len(chunk)

    status_text = str(run.status or "").lower()
    return ApiResponse(
        success=True,
        data={
            "run_id": run.id,
            "status": status_text,
            "log": chunk,
            "offset": new_offset,
            "exists": bool(log_text),
            "eof": new_offset >= total,
        },
        message=_build_run_message(status_text),
    )


@router.get("/runs/{run_id}/results", response_model=ApiResponse)
async def get_run_results(run_id: str, db: Session = Depends(get_db)):
    run = db.query(TaskCenterRun).filter(TaskCenterRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="任务运行实例不存在")

    _sync_run_from_legacy_tasks(db, run)
    rows = (
        db.query(TaskCenterResultIndex)
        .filter(TaskCenterResultIndex.run_id == run_id)
        .order_by(TaskCenterResultIndex.created_at.desc())
        .all()
    )
    indexed = []
    for row in rows:
        indexed.append(
            {
                "id": row.id,
                "run_id": row.run_id,
                "point_id": row.point_id or "",
                "model_version": row.model_version or "",
                "result_path": row.result_path or "",
                "status": row.status or "",
                "meta": _json_loads(row.meta, {}),
                "created_at": _dt_iso(row.created_at),
            }
        )

    return ApiResponse(
        success=True,
        data={
            "run_id": run.id,
            "status": str(run.status or "").lower(),
            "result": _json_loads(run.result, {}),
            "indexed_results": indexed,
        },
        message="任务结果获取成功",
    )
