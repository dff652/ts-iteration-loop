"""
Task Center 工具函数与序列化。
"""
import json
from datetime import datetime

from fastapi import HTTPException

from src.db.database import TaskCenterStepRun
from src.models.schemas import TaskStatus


_TASK_CENTER_META_KEY = "__task_center"


def _json_dumps(payload: dict | list | None) -> str:
    if payload is None:
        payload = {}
    return json.dumps(payload, ensure_ascii=False)


def _json_loads(text: str | None, default):
    if not text:
        return default
    try:
        loaded = json.loads(text)
    except Exception:
        return default
    return loaded


def _dt_iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _int_value(value, *, default: int = 0, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        num = int(value)
    except Exception:
        num = default
    if min_value is not None and num < min_value:
        num = min_value
    if max_value is not None and num > max_value:
        num = max_value
    return num


def _float_value(value, *, default: float = 0.0, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        num = float(value)
    except Exception:
        num = default
    if min_value is not None and num < min_value:
        num = min_value
    if max_value is not None and num > max_value:
        num = max_value
    return num


def _normalize_steps(raw_steps: list[str] | None) -> list[str]:
    steps = []
    for item in raw_steps or []:
        text = str(item or "").strip().lower()
        if text and text not in steps:
            steps.append(text)
    return steps or ["acquire", "inference"]


def _normalize_task_status(value: str | None) -> str:
    if isinstance(value, TaskStatus):
        status = str(value.value or "").strip().lower()
    else:
        status = str(value or "").strip().lower()
    if status.startswith("taskstatus."):
        status = status.split(".", 1)[1]
    if status in {TaskStatus.BLOCKED}:
        return TaskStatus.BLOCKED
    if status in {TaskStatus.RUNNABLE}:
        return TaskStatus.RUNNABLE
    if status in {TaskStatus.PENDING, "queued"}:
        return TaskStatus.PENDING
    if status in {TaskStatus.RUNNING}:
        return TaskStatus.RUNNING
    if status in {TaskStatus.COMPLETED, "success"}:
        return TaskStatus.COMPLETED
    if status in {TaskStatus.FAILED, "error"}:
        return TaskStatus.FAILED
    if status in {TaskStatus.CANCELLED}:
        return TaskStatus.CANCELLED
    if status in {TaskStatus.TIMEOUT}:
        return TaskStatus.TIMEOUT
    return TaskStatus.PENDING


def _append_step_log(step: TaskCenterStepRun, text: str) -> None:
    line = str(text or "").strip()
    if not line:
        return
    if step.logs:
        step.logs = f"{step.logs}\n{line}".strip()
    else:
        step.logs = line


def _run_payload(run) -> dict:
    raw = _json_loads(run.input_payload, {})
    return raw if isinstance(raw, dict) else {}


def _build_run_message(status_text: str) -> str:
    if status_text == TaskStatus.BLOCKED:
        return "任务依赖未满足"
    if status_text == TaskStatus.RUNNABLE:
        return "任务可执行"
    if status_text == TaskStatus.FAILED:
        return "任务执行失败"
    if status_text == TaskStatus.COMPLETED:
        return "任务执行完成"
    if status_text == TaskStatus.CANCELLED:
        return "任务已取消"
    if status_text == TaskStatus.TIMEOUT:
        return "任务执行超时"
    if status_text == TaskStatus.RUNNING:
        return "任务执行中"
    return "任务排队中"


def _parse_depends_on(step: TaskCenterStepRun) -> list[str]:
    raw = _json_loads(step.depends_on, [])
    if not isinstance(raw, list):
        return []
    deps: list[str] = []
    for item in raw:
        text = str(item or "").strip().lower()
        if text and text not in deps:
            deps.append(text)
    return deps


def _build_step_specs(step_specs_raw: list[dict] | None, steps_raw: list[str] | None) -> list[dict]:
    specs: list[dict] = []
    raw_specs = step_specs_raw if isinstance(step_specs_raw, list) else []
    if raw_specs:
        for item in raw_specs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("step_name") or "").strip().lower()
            if not name:
                continue
            depends_on_raw = item.get("depends_on")
            depends_on: list[str] = []
            if isinstance(depends_on_raw, list):
                for dep in depends_on_raw:
                    dep_name = str(dep or "").strip().lower()
                    if dep_name and dep_name not in depends_on and dep_name != name:
                        depends_on.append(dep_name)
            specs.append({"name": name, "depends_on": depends_on})
    else:
        names = _normalize_steps(steps_raw)
        for i, name in enumerate(names):
            depends_on = [names[i - 1]] if i > 0 else []
            specs.append({"name": name, "depends_on": depends_on})

    names = [str(spec.get("name") or "") for spec in specs]
    unique_names = set(names)
    if len(unique_names) != len(names):
        raise HTTPException(status_code=422, detail="step 名称必须唯一")
    for spec in specs:
        for dep in spec.get("depends_on") or []:
            if dep not in unique_names:
                raise HTTPException(status_code=422, detail=f"step 依赖不存在: {dep}")
    if not specs:
        raise HTTPException(status_code=422, detail="至少需要一个 step")
    return specs
