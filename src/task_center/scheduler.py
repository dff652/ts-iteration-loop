"""
Task Center 最小调度器骨架。

说明：
- 仅负责扫描 task_definitions 并按策略创建/触发 run。
- 默认不自动启动；通过 `python -m src.task_center.scheduler` 手动运行。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from configs.settings import settings
from src.api import task_center as task_center_api
from src.core.logging_config import get_logger
from src.db.database import SessionLocal, TaskCenterDefinition, TaskCenterRun
from src.models.schemas import TaskStatus
from src.utils.time_utils import utc_now_naive


logger = get_logger(__name__)


@dataclass
class TickStats:
    scanned: int = 0
    created: int = 0
    dispatched: int = 0
    skipped: int = 0
    errors: int = 0
    swept: int = 0


def _parse_int(text: str, *, default: int) -> int:
    try:
        return int(str(text).strip())
    except Exception:
        return default


def _cron_field_match(expr: str, value: int, *, min_v: int, max_v: int) -> bool:
    expr = str(expr or "").strip()
    if expr == "*":
        return True
    for raw in expr.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "*":
            return True
        if token.startswith("*/"):
            step = _parse_int(token[2:], default=0)
            if step > 0 and value % step == 0:
                return True
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            a = _parse_int(left, default=min_v)
            b = _parse_int(right, default=max_v)
            if a <= value <= b:
                return True
            continue
        if _parse_int(token, default=min_v - 1) == value:
            return True
    return False


def _cron_matches(now: datetime, cron_expr: str | None) -> bool:
    text = str(cron_expr or "").strip()
    parts = text.split()
    if len(parts) != 5:
        return False
    minute, hour, dom, month, dow = parts
    dow_value = (now.weekday() + 1) % 7  # cron: 0=Sunday
    return (
        _cron_field_match(minute, now.minute, min_v=0, max_v=59)
        and _cron_field_match(hour, now.hour, min_v=0, max_v=23)
        and _cron_field_match(dom, now.day, min_v=1, max_v=31)
        and _cron_field_match(month, now.month, min_v=1, max_v=12)
        and _cron_field_match(dow, dow_value, min_v=0, max_v=6)
    )


def _window_exists_run(db, definition_id: str, window_start: datetime, window_end: datetime) -> bool:
    row = (
        db.query(TaskCenterRun.id)
        .filter(
            TaskCenterRun.definition_id == definition_id,
            TaskCenterRun.created_at >= window_start,
            TaskCenterRun.created_at < window_end,
        )
        .first()
    )
    return row is not None


def _auto_should_trigger(db, definition_id: str) -> bool:
    active = (
        db.query(TaskCenterRun.id)
        .filter(
            TaskCenterRun.definition_id == definition_id,
            TaskCenterRun.status.in_(
                [
                    TaskStatus.BLOCKED,
                    TaskStatus.RUNNABLE,
                    TaskStatus.PENDING,
                    TaskStatus.RUNNING,
                ]
            ),
        )
        .first()
    )
    return active is None


def run_sweeper_tick(db: Session, stats: TickStats) -> None:
    # 扫描所有非终态的任务，迫使进行外部同步并尝试应用超时和重试
    active_runs = (
        db.query(TaskCenterRun)
        .filter(
            TaskCenterRun.status.in_(
                [
                    TaskStatus.BLOCKED,
                    TaskStatus.RUNNABLE,
                    TaskStatus.PENDING,
                    TaskStatus.RUNNING,
                ]
            )
        )
        .order_by(TaskCenterRun.created_at.asc())
        .all()
    )

    from src.task_center.run_operations import _sync_run_from_legacy_tasks

    for run in active_runs:
        try:
            _sync_run_from_legacy_tasks(db, run)
            stats.swept += 1
        except Exception as exc:
            logger.error("sweeper error on run_id=%s: %s", run.id, exc, exc_info=True)


def run_scheduler_tick(now: datetime | None = None, execute_mode: str | None = None) -> TickStats:
    current = now or utc_now_naive()
    mode = str(execute_mode or getattr(settings, "TASK_CENTER_EXECUTION_MODE", "dispatch")).strip().lower()
    stats = TickStats()
    window_start = current.replace(second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=1)

    db = SessionLocal()
    try:
        run_sweeper_tick(db, stats)

        definitions = (
            db.query(TaskCenterDefinition)
            .filter(TaskCenterDefinition.enabled == True)  # noqa: E712
            .order_by(TaskCenterDefinition.created_at.asc())
            .all()
        )
        for definition in definitions:
            stats.scanned += 1
            trigger_mode = str(definition.trigger_mode or "").strip().lower()
            should_trigger = False

            if trigger_mode == "schedule":
                should_trigger = _cron_matches(current, definition.schedule_cron)
            elif trigger_mode == "auto":
                should_trigger = _auto_should_trigger(db, definition.id)
            else:
                stats.skipped += 1
                continue

            if not should_trigger:
                stats.skipped += 1
                continue

            if _window_exists_run(db, definition.id, window_start, window_end):
                stats.skipped += 1
                continue

            cfg = task_center_api._json_loads(definition.config, {})  # noqa: SLF001
            if not isinstance(cfg, dict):
                cfg = {}

            # 并发检查
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
                    stats.skipped += 1
                    logger.debug("skip definition_id=%s: max_concurrent_runs reached (%s/%s)", definition.id, running_count, max_concurrent)
                    continue

            try:
                run = task_center_api._create_run_record(  # noqa: SLF001
                    db,
                    definition_id=definition.id,
                    task_type=str(cfg.get("task_type") or definition.task_type or "acquire_inference"),
                    trigger_mode=trigger_mode,
                    input_payload=cfg.get("input_payload") if isinstance(cfg.get("input_payload"), dict) else {},
                    steps=cfg.get("steps") if isinstance(cfg.get("steps"), list) else None,
                    step_specs=cfg.get("step_specs") if isinstance(cfg.get("step_specs"), list) else None,
                    max_retries=cfg.get("max_retries"),
                    retry_policy=cfg.get("retry_policy"),
                    retry_delay_sec=cfg.get("retry_delay_sec"),
                    retry_backoff_factor=cfg.get("retry_backoff_factor"),
                    retry_max_delay_sec=cfg.get("retry_max_delay_sec"),
                    retry_on_errors=cfg.get("retry_on_errors") if isinstance(cfg.get("retry_on_errors"), list) else None,
                    timeout_sec=cfg.get("timeout_sec"),
                    created_at=current,
                )
                stats.created += 1
            except Exception as exc:
                stats.errors += 1
                logger.error("scheduler create run failed definition_id=%s: %s", definition.id, exc, exc_info=True)
                continue

            if mode == "simulate":
                task_center_api._execute_run_impl(db, run, simulate=True)  # noqa: SLF001
                stats.dispatched += 1
            elif mode == "dispatch":
                task_center_api._execute_run_impl(db, run, simulate=False)  # noqa: SLF001
                stats.dispatched += 1
            else:
                logger.warning("unknown TASK_CENTER_EXECUTION_MODE=%s, skip execute", mode)
    finally:
        db.close()

    return stats


def run_scheduler_loop() -> None:
    interval = int(getattr(settings, "TASK_CENTER_SCHEDULER_INTERVAL_SEC", 30) or 30)
    interval = max(2, interval)
    logger.info("Task Center scheduler started | interval=%ss | mode=%s", interval, getattr(settings, "TASK_CENTER_EXECUTION_MODE", "dispatch"))
    while True:
        try:
            stats = run_scheduler_tick()
            logger.info(
                "scheduler tick | swept=%s scanned=%s created=%s dispatched=%s skipped=%s errors=%s",
                stats.swept,
                stats.scanned,
                stats.created,
                stats.dispatched,
                stats.skipped,
                stats.errors,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error("scheduler tick failed: %s", exc, exc_info=True)
        time.sleep(interval)


if __name__ == "__main__":
    run_scheduler_loop()
