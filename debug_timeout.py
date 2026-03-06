import sys
import asyncio
import time
from datetime import timedelta
from src.task_center.scheduler import run_sweeper_tick, TickStats
from src.models.schemas import TaskCenterDefinitionRequest, TaskCenterRunCreateRequest
from src.db import database as db_mod
from src.api import task_center as task_center_api
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_mod.Base.metadata.create_all(bind=engine)

with SessionLocal() as db:
    req = TaskCenterDefinitionRequest(
        name="sweeper-test",
        trigger_mode="manual",
        config={
            "timeout_sec": 1,
            "max_retries": 1,
            "retry_delay_sec": 0
        }
    )
    resp = asyncio.run(task_center_api.create_definition(req, db=db))
    
    run_req = TaskCenterRunCreateRequest(
        task_type="inference",
        trigger_mode="manual",
        steps=["inference"],
        input_payload={
            "__task_center": {
                "timeout_sec": 1,
                "max_retries": 1,
                "retry_delay_sec": 0
            }
        },
        definition_id=resp.data["id"],
        auto_execute=False
    )
    run_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
    run_id = run_resp.data["run_id"]
    
    run = db.query(db_mod.TaskCenterRun).filter_by(id=run_id).first()
    import src.task_center.state_machine as sm_mod
    print("CONTROL INITIAL:", sm_mod._extract_run_control(sm_mod._run_payload(run)))

    run.status = "running"
    step = db.query(db_mod.TaskCenterStepRun).filter_by(run_id=run_id).first()
    if step:
        step.status = "running"
    
    now = task_center_api.utc_now_naive()
    run.started_at = now
    if step:
        step.started_at = now
    db.commit()
    
    time.sleep(1.5)
    
    orig = sm_mod._maybe_auto_retry_run
    def verbose_retry(db, r, step_rows):
        print(">> check run status:", r.status)
        control = sm_mod._extract_run_control(sm_mod._run_payload(r))
        print(">> control:", control)
        return orig(db, r, step_rows)

    sm_mod._maybe_auto_retry_run = verbose_retry
    import src.task_center.run_operations as ro_mod
    orig_sync = ro_mod._sync_run_from_legacy_tasks
    def mock_sync(db, r):
        orig_sync(db, r)
        step_rows = db.query(db_mod.TaskCenterStepRun).filter_by(run_id=r.id).all()
        verbose_retry(db, r, step_rows)
    ro_mod._sync_run_from_legacy_tasks = mock_sync

    import src.task_center.scheduler as sched_mod
    sched_mod._sync_run_from_legacy_tasks = mock_sync
    
    stats = TickStats()
    run_sweeper_tick(db, stats)

