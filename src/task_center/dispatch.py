"""
Task Center 执行器分发：acquire/inference/training。
"""
import uuid
from copy import deepcopy

from sqlalchemy.orm import Session

from src.core.tasks import celery_app
from src.db.database import Task, TaskCenterRun, TaskCenterStepRun
from src.models.schemas import TaskStatus
from src.utils.time_utils import utc_now_naive

from src.task_center.helpers import (
    _append_step_log,
    _json_dumps,
    _run_payload,
)


def _dispatch_inference_legacy_task(
    db: Session,
    run: TaskCenterRun,
    step: TaskCenterStepRun,
    payload: dict,
) -> None:
    model = str(payload.get("model") or "").strip()
    input_files = payload.get("input_files")
    if not model or not isinstance(input_files, list) or not input_files:
        raise ValueError("inference step 缺少必填字段: model/input_files")

    algorithm = str(payload.get("algorithm") or "chatts")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}

    legacy_task_id = str(uuid.uuid4())
    cfg = {
        "model": model,
        "algorithm": algorithm,
        "input_files": input_files,
        "params": params,
        "executor": "task_center",
        "task_center_run_id": run.id,
        "task_center_step_id": step.id,
    }
    legacy_task = Task(
        id=legacy_task_id,
        type="inference",
        status=TaskStatus.PENDING,
        config=_json_dumps(cfg),
    )
    db.add(legacy_task)
    db.flush()

    async_result = celery_app.send_task(
        "inference.batch",
        kwargs={
            "task_id": legacy_task_id,
            "model": model,
            "algorithm": algorithm,
            "input_files": input_files,
            "params": params,
        },
    )
    celery_task_id = str(getattr(async_result, "id", "") or "")
    cfg["celery_task_id"] = celery_task_id
    legacy_task.config = _json_dumps(cfg)

    step.status = TaskStatus.RUNNING
    step.started_at = step.started_at or utc_now_naive()
    step.message = "inference step dispatched"
    step.result = _json_dumps(
        {
            "dispatch_mode": "legacy_task",
            "executor": "inference.batch",
            "legacy_task_id": legacy_task_id,
            "celery_task_id": celery_task_id,
        }
    )
    _append_step_log(
        step,
        f"[{utc_now_naive().isoformat()}] dispatched inference.batch legacy_task_id={legacy_task_id} celery_task_id={celery_task_id}",
    )


def _dispatch_training_legacy_task(
    db: Session,
    run: TaskCenterRun,
    step: TaskCenterStepRun,
    payload: dict,
) -> None:
    config_name = str(payload.get("config_name") or "").strip()
    if not config_name:
        raise ValueError("training step 缺少必填字段: config_name")

    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}

    kwargs = {
        "task_id": str(uuid.uuid4()),
        "config_name": config_name,
        "version_tag": payload.get("version_tag"),
        "model_family": str(payload.get("model_family") or "chatts"),
        "auto_eval": bool(payload.get("auto_eval", False)),
        "eval_truth_dir": payload.get("eval_truth_dir"),
        "eval_data_dir": payload.get("eval_data_dir"),
        "eval_dataset_name": payload.get("eval_dataset_name"),
        "eval_output_dir": payload.get("eval_output_dir"),
        "eval_device": payload.get("eval_device"),
        "eval_method": payload.get("eval_method"),
        "params": params,
    }
    legacy_task_id = kwargs["task_id"]
    cfg = deepcopy(kwargs)
    cfg.update(
        {
            "executor": "task_center",
            "task_center_run_id": run.id,
            "task_center_step_id": step.id,
        }
    )
    legacy_task = Task(
        id=legacy_task_id,
        type="training",
        status=TaskStatus.PENDING,
        config=_json_dumps(cfg),
    )
    db.add(legacy_task)
    db.flush()

    async_result = celery_app.send_task("training.run", kwargs=kwargs)
    celery_task_id = str(getattr(async_result, "id", "") or "")
    cfg["celery_task_id"] = celery_task_id
    legacy_task.config = _json_dumps(cfg)

    step.status = TaskStatus.RUNNING
    step.started_at = step.started_at or utc_now_naive()
    step.message = "training step dispatched"
    step.result = _json_dumps(
        {
            "dispatch_mode": "legacy_task",
            "executor": "training.run",
            "legacy_task_id": legacy_task_id,
            "celery_task_id": celery_task_id,
        }
    )
    _append_step_log(
        step,
        f"[{utc_now_naive().isoformat()}] dispatched training.run legacy_task_id={legacy_task_id} celery_task_id={celery_task_id}",
    )


def _dispatch_acquire_legacy_task(
    db: Session,
    run: TaskCenterRun,
    step: TaskCenterStepRun,
    payload: dict,
) -> None:
    source = str(payload.get("source") or "").strip()
    if not source:
        raise ValueError("acquire step 缺少必填字段: source")

    kwargs = {
        "task_id": str(uuid.uuid4()),
        "source": source,
        "target_points": int(payload.get("target_points") or 5000),
        "start_time": payload.get("start_time"),
        "end_time": payload.get("end_time"),
        "host": str(payload.get("host") or "192.168.199.185"),
        "port": str(payload.get("port") or "6667"),
        "user": str(payload.get("user") or ""),
        "password": str(payload.get("password") or ""),
        "point_name": str(payload.get("point_name") or "*"),
    }
    legacy_task_id = kwargs["task_id"]
    cfg = deepcopy(kwargs)
    cfg.update(
        {
            "executor": "task_center",
            "task_center_run_id": run.id,
            "task_center_step_id": step.id,
            "password": "***" if cfg.get("password") else "",
        }
    )
    legacy_task = Task(
        id=legacy_task_id,
        type="acquire",
        status=TaskStatus.PENDING,
        config=_json_dumps(cfg),
    )
    db.add(legacy_task)
    db.flush()

    async_result = celery_app.send_task("data.acquire", kwargs=kwargs)
    celery_task_id = str(getattr(async_result, "id", "") or "")
    cfg["celery_task_id"] = celery_task_id
    legacy_task.config = _json_dumps(cfg)

    step.status = TaskStatus.RUNNING
    step.started_at = step.started_at or utc_now_naive()
    step.message = "acquire step dispatched"
    step.result = _json_dumps(
        {
            "dispatch_mode": "legacy_task",
            "executor": "data.acquire",
            "legacy_task_id": legacy_task_id,
            "celery_task_id": celery_task_id,
        }
    )
    _append_step_log(
        step,
        f"[{utc_now_naive().isoformat()}] dispatched data.acquire legacy_task_id={legacy_task_id} celery_task_id={celery_task_id}",
    )


def _dispatch_step(db: Session, run: TaskCenterRun, step: TaskCenterStepRun) -> None:
    payload = _run_payload(run)
    name = str(step.step_name or "").strip().lower()
    if name == "acquire":
        _dispatch_acquire_legacy_task(db, run, step, payload)
        return
    if name == "inference":
        _dispatch_inference_legacy_task(db, run, step, payload)
        return
    if name == "training":
        _dispatch_training_legacy_task(db, run, step, payload)
        return
    raise ValueError(f"当前版本暂不支持 step: {name}")
