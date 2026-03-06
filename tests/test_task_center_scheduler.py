import json
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import task_center as task_center_api
from src.db import database as db_mod
from src.task_center import scheduler as scheduler_mod


class _FakeAsyncResult:
    def __init__(self, task_id: str):
        self.id = task_id


def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    monkeypatch.setattr(scheduler_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)
    return test_session_local


def test_scheduler_tick_creates_and_dispatches_schedule_run(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {"calls": 0, "names": []}

    def _fake_send_task(name, kwargs=None, **_):
        captured["calls"] += 1
        captured["names"].append(name)
        task_id = str((kwargs or {}).get("task_id") or "x")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    with test_session_local() as db:
        definition = db_mod.TaskCenterDefinition(
            id="def-schedule-1",
            name="scheduled-acquire",
            task_type="acquire",
            trigger_mode="schedule",
            schedule_cron="* * * * *",
            enabled=True,
            config=json.dumps(
                {
                    "task_type": "acquire",
                    "steps": ["acquire"],
                    "input_payload": {"source": "root.sg.dev"},
                }
            ),
        )
        db.add(definition)
        db.commit()

    now = datetime(2026, 3, 3, 12, 30, 5)
    stats = scheduler_mod.run_scheduler_tick(now=now, execute_mode="dispatch")
    assert stats.scanned == 1
    assert stats.created == 1
    assert stats.dispatched == 1
    assert captured["calls"] == 1
    assert captured["names"] == ["data.acquire"]

    # 同一分钟再次 tick，不应重复创建 run
    stats_again = scheduler_mod.run_scheduler_tick(now=datetime(2026, 3, 3, 12, 30, 40), execute_mode="dispatch")
    assert stats_again.created == 0

    with test_session_local() as db:
        runs = db.query(db_mod.TaskCenterRun).all()
        assert len(runs) == 1
        steps = db.query(db_mod.TaskCenterStepRun).filter(db_mod.TaskCenterStepRun.run_id == runs[0].id).all()
        assert len(steps) == 1
        assert steps[0].step_name == "acquire"

def test_scheduler_sweeper_triggers_timeout_and_retry(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    from src.task_center.scheduler import run_sweeper_tick, TickStats
    from src.api.task_center import execute_run
    from src.models.schemas import TaskCenterDefinitionRequest, TaskCenterRunCreateRequest
    import asyncio
    from datetime import timedelta
    
    with test_session_local() as db:
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
        run.status = "running"
        step = db.query(db_mod.TaskCenterStepRun).filter_by(run_id=run_id).first()
        if step:
            step.status = "running"
        
        # 手工更新时间在当下以确保计算基准正确
        now = task_center_api.utc_now_naive()
        run.started_at = now
        if step:
             step.started_at = now
        db.commit()

        from unittest.mock import MagicMock
        mock_send = MagicMock()
        monkeypatch.setattr(task_center_api.celery_app, "send_task", mock_send)

        import time
        time.sleep(1.5)  # 真实等待超过 1s 的超时设定

        stats = TickStats()
        run_sweeper_tick(db, stats)
        assert stats.swept >= 1
    
        # 原有的 run 应该变成了 timeout，并且又新建了一个 pending run（作为 retry）
        db.refresh(run)
        assert run.status == "timeout"
        
        retry_run = db.query(db_mod.TaskCenterRun).filter(
            db_mod.TaskCenterRun.id != run_id,
            db_mod.TaskCenterRun.definition_id == resp.data["id"]
        ).first()
        
        assert retry_run is not None
        assert retry_run.status in ("pending", "running", "failed")

