import asyncio
import json
from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import task_center as task_center_api
from src.db import database as db_mod
from src.task_center import state_machine as _sm_mod
from src.task_center import run_operations as _ro_mod
from src.models.schemas import (
    TaskCenterDeadLetterReplayRequest,
    TaskCenterDefinitionRequest,
    TaskCenterEventTriggerRequest,
    TaskCenterRunCreateRequest,
    TaskCenterRunExecuteRequest,
)
from fastapi import HTTPException


def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)
    return test_session_local


def test_task_center_definition_create_and_list(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    req = TaskCenterDefinitionRequest(
        name="nightly acquire + infer",
        task_type="acquire_inference",
        trigger_mode="schedule",
        schedule_cron="0 2 * * *",
        config={"source": "root.a.b"},
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_definition(req, db=db))
        list_resp = asyncio.run(task_center_api.list_definitions(db=db))

    assert create_resp.success is True
    data = create_resp.data or {}
    assert data.get("name") == "nightly acquire + infer"
    assert data.get("trigger_mode") == "schedule"
    assert data.get("schedule_cron") == "0 2 * * *"
    assert isinstance(data.get("id"), str) and data.get("id")

    definitions = (list_resp.data or {}).get("definitions") or []
    assert len(definitions) == 1
    assert definitions[0].get("name") == "nightly acquire + infer"


def test_task_center_run_auto_execute_and_results(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    run_req = TaskCenterRunCreateRequest(
        task_type="acquire_inference",
        trigger_mode="manual",
        input_payload={
            "point_id": "P_1001",
            "model_version": "chatts-v1",
            "result_path": "/tmp/p1001.csv",
        },
        steps=["acquire", "inference"],
        auto_execute=True,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        run_id = str((create_resp.data or {}).get("run_id") or "")
        status_resp = asyncio.run(task_center_api.get_run_status(run_id, db=db))
        result_resp = asyncio.run(task_center_api.get_run_results(run_id, db=db))

    assert create_resp.success is True
    assert (create_resp.data or {}).get("status") == "completed"

    status_data = status_resp.data or {}
    assert status_data.get("status") == "completed"
    steps = status_data.get("steps") if isinstance(status_data.get("steps"), list) else []
    assert len(steps) == 2
    assert all((str(step.get("status")) == "completed") for step in steps)

    indexed = (result_resp.data or {}).get("indexed_results") or []
    assert len(indexed) == 1
    assert indexed[0].get("point_id") == "P_1001"
    assert indexed[0].get("model_version") == "chatts-v1"
    assert indexed[0].get("result_path") == "/tmp/p1001.csv"


def test_task_center_run_execute_and_incremental_log(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    run_req = TaskCenterRunCreateRequest(
        task_type="acquire_inference",
        trigger_mode="manual",
        steps=["acquire", "inference"],
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        run_id = str((create_resp.data or {}).get("run_id") or "")
        execute_resp = asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=True),
                db=db,
            )
        )
        log_resp_1 = asyncio.run(task_center_api.get_run_log(run_id, offset=0, max_bytes=40, db=db))
        offset_1 = int((log_resp_1.data or {}).get("offset", 0))
        log_resp_2 = asyncio.run(task_center_api.get_run_log(run_id, offset=offset_1, max_bytes=10000, db=db))

    assert execute_resp.success is True
    assert (execute_resp.data or {}).get("status") == "completed"

    assert log_resp_1.success is True
    assert (log_resp_1.data or {}).get("exists") is True
    tail = str((log_resp_2.data or {}).get("log") or "")
    assert "step=acquire started" in tail or "step=inference started" in tail
    assert "completed" in tail


def test_task_center_inference_submit_flow_locates_run(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {"calls": 0, "names": []}

    def _fake_send_task(name, kwargs=None, **_):
        captured["calls"] += 1
        captured["names"].append(name)
        task_id = str((kwargs or {}).get("task_id") or "x")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    with test_session_local() as db:
        create_resp = asyncio.run(
            task_center_api.create_run(
                TaskCenterRunCreateRequest(
                    task_type="inference",
                    trigger_mode="manual",
                    input_payload={
                        "model": "/tmp/model",
                        "algorithm": "chatts",
                        "input_files": ["/tmp/a.csv"],
                    },
                    steps=["inference"],
                    auto_execute=False,
                ),
                db=db,
            )
        )
        run_id = str((create_resp.data or {}).get("run_id") or "")

        exec_resp = asyncio.run(
            task_center_api.execute_run(run_id, request=TaskCenterRunExecuteRequest(simulate=False), db=db)
        )
        list_resp = asyncio.run(task_center_api.list_runs(run_id=run_id, db=db))
        rows = (list_resp.data or {}).get("runs") or []

    assert run_id
    assert exec_resp.success is True
    assert captured["calls"] == 1
    assert captured["names"] == ["inference.batch"]
    assert list_resp.success is True
    assert len(rows) == 1
    assert str(rows[0].get("run_id") or "") == run_id


def test_task_center_training_submit_flow_locates_run(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {"calls": 0, "names": []}

    def _fake_send_task(name, kwargs=None, **_):
        captured["calls"] += 1
        captured["names"].append(name)
        task_id = str((kwargs or {}).get("task_id") or "x")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    with test_session_local() as db:
        create_resp = asyncio.run(
            task_center_api.create_run(
                TaskCenterRunCreateRequest(
                    task_type="training",
                    trigger_mode="manual",
                    input_payload={
                        "config_name": "dev",
                        "model_family": "chatts",
                        "params": {"override_dataset": "demo_dataset"},
                    },
                    steps=["training"],
                    auto_execute=False,
                ),
                db=db,
            )
        )
        run_id = str((create_resp.data or {}).get("run_id") or "")

        exec_resp = asyncio.run(
            task_center_api.execute_run(run_id, request=TaskCenterRunExecuteRequest(simulate=False), db=db)
        )
        list_resp = asyncio.run(task_center_api.list_runs(run_id=run_id, db=db))
        rows = (list_resp.data or {}).get("runs") or []

    assert run_id
    assert exec_resp.success is True
    assert captured["calls"] == 1
    assert captured["names"] == ["training.run"]
    assert list_resp.success is True
    assert len(rows) == 1
    assert str(rows[0].get("run_id") or "") == run_id


def test_task_center_list_runs_filter_by_run_id(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    req_a = TaskCenterRunCreateRequest(task_type="inference", trigger_mode="manual", steps=["inference"])
    req_b = TaskCenterRunCreateRequest(task_type="training", trigger_mode="manual", steps=["training"])

    with test_session_local() as db:
        resp_a = asyncio.run(task_center_api.create_run(req_a, db=db))
        resp_b = asyncio.run(task_center_api.create_run(req_b, db=db))
        run_a = str((resp_a.data or {}).get("run_id") or "")
        run_b = str((resp_b.data or {}).get("run_id") or "")

        list_resp = asyncio.run(task_center_api.list_runs(run_id=run_a, db=db))
        rows = (list_resp.data or {}).get("runs") or []

    assert run_a and run_b and run_a != run_b
    assert list_resp.success is True
    assert len(rows) == 1
    assert str(rows[0].get("run_id") or "") == run_a


def test_task_center_event_definition_rejected_when_events_disabled(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENTS_ENABLED", False)

    req = TaskCenterDefinitionRequest(
        name="event-disabled",
        task_type="inference",
        trigger_mode="event",
        config={"event_key": "data.ready"},
    )

    with test_session_local() as db:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(task_center_api.create_definition(req, db=db))

    assert int(exc_info.value.status_code) == 409


class _FakeAsyncResult:
    def __init__(self, task_id: str):
        self.id = task_id


def test_task_center_trigger_event_creates_and_dispatches_run(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENTS_ENABLED", True)
    captured = {"calls": 0, "names": []}

    def _fake_send_task(name, kwargs=None, **_):
        captured["calls"] += 1
        captured["names"].append(name)
        task_id = str((kwargs or {}).get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-event-1",
                name="event-inference",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps(
                    {
                        "event_key": "data.ready",
                        "task_type": "inference",
                        "steps": ["inference"],
                        "input_payload": {
                            "model": "/tmp/model",
                            "algorithm": "chatts",
                            "input_files": ["/tmp/a.csv"],
                        },
                    }
                ),
            )
        )
        db.commit()

        resp = asyncio.run(
            task_center_api.trigger_event(
                TaskCenterEventTriggerRequest(event_key="data.ready", execute_mode="dispatch"),
                db=db,
            )
        )
        run_ids = (resp.data or {}).get("run_ids") or []
        assert len(run_ids) == 1
        status_resp = asyncio.run(task_center_api.get_run_status(str(run_ids[0]), db=db))

    assert resp.success is True
    assert (resp.data or {}).get("created_count") == 1
    assert (resp.data or {}).get("dispatched_count") == 1
    assert captured["calls"] == 1
    assert captured["names"] == ["inference.batch"]
    assert (status_resp.data or {}).get("trigger_mode") == "event"
    assert (status_resp.data or {}).get("event_key") == "data.ready"


def test_task_center_trigger_event_dedupe_key(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENTS_ENABLED", True)

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-event-2",
                name="event-inference-dedupe",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps(
                    {
                        "event_key": "data.ready",
                        "task_type": "inference",
                        "steps": ["inference"],
                        "input_payload": {
                            "model": "/tmp/model",
                            "algorithm": "chatts",
                            "input_files": ["/tmp/a.csv"],
                        },
                    }
                ),
            )
        )
        db.commit()

        first = asyncio.run(
            task_center_api.trigger_event(
                TaskCenterEventTriggerRequest(
                    event_key="data.ready",
                    dedupe_key="evt-001",
                    execute_mode="none",
                ),
                db=db,
            )
        )
        second = asyncio.run(
            task_center_api.trigger_event(
                TaskCenterEventTriggerRequest(
                    event_key="data.ready",
                    dedupe_key="evt-001",
                    execute_mode="none",
                ),
                db=db,
            )
        )

        runs = db.query(db_mod.TaskCenterRun).all()

    assert first.success is True
    assert (first.data or {}).get("created_count") == 1
    assert second.success is True
    assert (second.data or {}).get("created_count") == 0
    deduped = (second.data or {}).get("deduped_definition_ids") or []
    assert deduped == ["def-event-2"]
    assert len(runs) == 1


def test_task_center_trigger_event_skips_non_matching_event_key(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENTS_ENABLED", True)

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-event-3",
                name="event-mismatch",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps({"event_key": "another.event", "steps": ["inference"]}),
            )
        )
        db.commit()

        resp = asyncio.run(
            task_center_api.trigger_event(
                TaskCenterEventTriggerRequest(event_key="data.ready", execute_mode="none"),
                db=db,
            )
        )

    assert resp.success is True
    assert (resp.data or {}).get("created_count") == 0
    skipped = (resp.data or {}).get("skipped_definition_ids") or []
    assert skipped == ["def-event-3"]


def test_task_center_trigger_event_rate_limit(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENTS_ENABLED", True)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENT_RATE_LIMIT_PER_MIN", 1)
    task_center_api._EVENT_RATE_LIMIT_STATE.clear()  # noqa: SLF001

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-event-limit-1",
                name="event-rate-limit",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps({"event_key": "data.ready", "steps": ["inference"]}),
            )
        )
        db.commit()

        first = asyncio.run(
            task_center_api.trigger_event(
                TaskCenterEventTriggerRequest(event_key="data.ready", execute_mode="none"),
                db=db,
            )
        )
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                task_center_api.trigger_event(
                    TaskCenterEventTriggerRequest(event_key="data.ready", execute_mode="none"),
                    db=db,
                )
            )

    assert first.success is True
    assert exc_info.value.status_code == 429


def test_task_center_dead_letter_list_and_replay(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENTS_ENABLED", True)
    dead_file = tmp_path / "dead_letters.jsonl"
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_DEAD_LETTER_PATH", str(dead_file))
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENT_RATE_LIMIT_PER_MIN", 1000)
    task_center_api._EVENT_RATE_LIMIT_STATE.clear()  # noqa: SLF001

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-event-replay-1",
                name="event-replay",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps(
                    {
                        "event_key": "data.ready",
                        "task_type": "inference",
                        "steps": ["inference"],
                        "input_payload": {
                            "model": "/tmp/model",
                            "algorithm": "chatts",
                            "input_files": ["/tmp/a.csv"],
                        },
                    }
                ),
            )
        )
        db.commit()

        dead_id = task_center_api._record_dead_letter(  # noqa: SLF001
            "test",
            {
                "event_key": "data.ready",
                "definition_id": "def-event-replay-1",
                "execute_mode": "none",
                "payload": {},
                "dedupe_key": "",
            },
            "manual test dead letter",
        )

        list_resp = asyncio.run(task_center_api.list_dead_letters(limit=10, offset=0))
        replay_resp = asyncio.run(
            task_center_api.replay_dead_letter(
                TaskCenterDeadLetterReplayRequest(event_id=dead_id, execute_mode="none"),
                db=db,
            )
        )
        metrics_resp = asyncio.run(task_center_api.get_event_metrics())

        runs = db.query(db_mod.TaskCenterRun).all()

    assert list_resp.success is True
    items = (list_resp.data or {}).get("items") or []
    assert any(str(item.get("id") or "") == dead_id for item in items)
    assert replay_resp.success is True
    result = (replay_resp.data or {}).get("result") or {}
    assert result.get("created_count") == 1
    assert metrics_resp.success is True
    assert int((metrics_resp.data or {}).get("dead_letter_total") or 0) >= 1
    assert len(runs) == 1


def test_task_center_execute_dispatches_inference_to_celery(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {}

    def _fake_send_task(name, kwargs=None, **_):
        captured["name"] = name
        captured["kwargs"] = kwargs or {}
        return _FakeAsyncResult("celery-inference-1")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    run_req = TaskCenterRunCreateRequest(
        task_type="inference",
        trigger_mode="manual",
        steps=["inference"],
        input_payload={
            "model": "/tmp/model",
            "algorithm": "chatts",
            "input_files": ["/tmp/a.csv", "/tmp/b.csv"],
            "params": {"x": 1},
        },
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        run_id = str((create_resp.data or {}).get("run_id") or "")
        execute_resp = asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )
        status_resp = asyncio.run(task_center_api.get_run_status(run_id, db=db))

        step_row = (
            db.query(db_mod.TaskCenterStepRun)
            .filter(db_mod.TaskCenterStepRun.run_id == run_id)
            .first()
        )
        assert step_row is not None
        step_result = json.loads(step_row.result or "{}")
        legacy_task_id = str(step_result.get("legacy_task_id") or "")
        legacy_task = db.query(db_mod.Task).filter(db_mod.Task.id == legacy_task_id).first()

    assert execute_resp.success is True
    assert captured["name"] == "inference.batch"
    assert captured["kwargs"]["model"] == "/tmp/model"
    assert captured["kwargs"]["algorithm"] == "chatts"
    assert captured["kwargs"]["input_files"] == ["/tmp/a.csv", "/tmp/b.csv"]
    assert isinstance(captured["kwargs"]["task_id"], str) and captured["kwargs"]["task_id"]

    assert legacy_task is not None
    legacy_cfg = json.loads(legacy_task.config or "{}")
    assert legacy_cfg.get("executor") == "task_center"
    assert legacy_cfg.get("celery_task_id") == "celery-inference-1"
    assert (status_resp.data or {}).get("status") == "running"


def test_task_center_status_syncs_from_legacy_task(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    def _fake_send_task(_name, kwargs=None, **_):
        task_id = str((kwargs or {}).get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    run_req = TaskCenterRunCreateRequest(
        task_type="training",
        trigger_mode="manual",
        steps=["training"],
        input_payload={
            "config_name": "train_cfg_a",
            "model_family": "chatts",
        },
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        run_id = str((create_resp.data or {}).get("run_id") or "")
        asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )

        step_row = (
            db.query(db_mod.TaskCenterStepRun)
            .filter(db_mod.TaskCenterStepRun.run_id == run_id)
            .first()
        )
        assert step_row is not None
        step_result = json.loads(step_row.result or "{}")
        legacy_task_id = str(step_result.get("legacy_task_id") or "")
        legacy_task = db.query(db_mod.Task).filter(db_mod.Task.id == legacy_task_id).first()
        assert legacy_task is not None
        legacy_task.status = "completed"
        legacy_task.result = json.dumps({"success": True, "output_dir": "/tmp/out"})
        db.commit()

        status_resp = asyncio.run(task_center_api.get_run_status(run_id, db=db))

    assert status_resp.success is True
    data = status_resp.data or {}
    assert data.get("status") == "completed"
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    assert len(steps) == 1
    assert steps[0].get("status") == "completed"


def test_task_center_cancel_run_revokes_legacy_task(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {"revoke_task_id": None, "terminate": None}

    def _fake_send_task(_name, kwargs=None, **_):
        task_id = str((kwargs or {}).get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    def _fake_revoke(task_id, terminate=False, **_):
        captured["revoke_task_id"] = task_id
        captured["terminate"] = bool(terminate)

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)
    monkeypatch.setattr(task_center_api.celery_app.control, "revoke", _fake_revoke)

    run_req = TaskCenterRunCreateRequest(
        task_type="inference",
        trigger_mode="manual",
        steps=["inference"],
        input_payload={
            "model": "/tmp/model",
            "algorithm": "chatts",
            "input_files": ["/tmp/a.csv"],
        },
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        run_id = str((create_resp.data or {}).get("run_id") or "")
        asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )
        cancel_resp = asyncio.run(task_center_api.cancel_run(run_id, db=db))
        status_resp = asyncio.run(task_center_api.get_run_status(run_id, db=db))

    assert cancel_resp.success is True
    assert (cancel_resp.data or {}).get("status") == "cancelled"
    assert (cancel_resp.data or {}).get("cancelled_steps") == 1
    assert isinstance(captured["revoke_task_id"], str) and captured["revoke_task_id"].startswith("celery-")
    assert captured["terminate"] is True
    assert (status_resp.data or {}).get("status") == "cancelled"


def test_task_center_retry_creates_new_pending_run(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    with test_session_local() as db:
        source_run = db_mod.TaskCenterRun(
            id="run-source-1",
            task_type="training",
            trigger_mode="manual",
            status="failed",
            input_payload=json.dumps({"config_name": "cfg-a"}),
        )
        db.add(source_run)
        db.add(
            db_mod.TaskCenterStepRun(
                run_id="run-source-1",
                step_name="training",
                status="failed",
            )
        )
        db.commit()

        retry_resp = asyncio.run(task_center_api.retry_run("run-source-1", db=db))

        new_run_id = str((retry_resp.data or {}).get("new_run_id") or "")
        new_run = db.query(db_mod.TaskCenterRun).filter(db_mod.TaskCenterRun.id == new_run_id).first()
        new_step = (
            db.query(db_mod.TaskCenterStepRun)
            .filter(db_mod.TaskCenterStepRun.run_id == new_run_id)
            .first()
        )

    assert retry_resp.success is True
    assert (retry_resp.data or {}).get("source_run_id") == "run-source-1"
    assert new_run is not None
    assert new_run.status == "runnable"
    assert new_run.input_payload == json.dumps({"config_name": "cfg-a"})
    assert new_step is not None
    assert new_step.step_name == "training"
    assert new_step.status == "runnable"


def test_task_center_retry_rejects_running_run(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    with test_session_local() as db:
        run = db_mod.TaskCenterRun(
            id="run-running-1",
            task_type="inference",
            trigger_mode="manual",
            status="running",
            input_payload=json.dumps({"model": "/tmp/m", "input_files": ["/tmp/a.csv"]}),
        )
        db.add(run)
        db.add(
            db_mod.TaskCenterStepRun(
                run_id="run-running-1",
                step_name="inference",
                status="running",
            )
        )
        db.commit()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(task_center_api.retry_run("run-running-1", db=db))

    assert exc_info.value.status_code == 409


def test_task_center_execute_dispatches_acquire_to_celery(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {}

    def _fake_send_task(name, kwargs=None, **_):
        captured["name"] = name
        captured["kwargs"] = kwargs or {}
        return _FakeAsyncResult("celery-acquire-1")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    run_req = TaskCenterRunCreateRequest(
        task_type="acquire",
        trigger_mode="manual",
        steps=["acquire"],
        input_payload={
            "source": "root.sg.dev",
            "target_points": 2000,
            "host": "127.0.0.1",
            "port": "6667",
            "user": "root",
            "password": "root",
            "point_name": "FI_1.PV",
            "start_time": "2025-01-01 00:00:00",
            "end_time": "2025-01-01 01:00:00",
        },
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        run_id = str((create_resp.data or {}).get("run_id") or "")
        execute_resp = asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )
        status_resp = asyncio.run(task_center_api.get_run_status(run_id, db=db))

        step_row = (
            db.query(db_mod.TaskCenterStepRun)
            .filter(db_mod.TaskCenterStepRun.run_id == run_id)
            .first()
        )
        assert step_row is not None
        step_result = json.loads(step_row.result or "{}")
        legacy_task_id = str(step_result.get("legacy_task_id") or "")
        legacy_task = db.query(db_mod.Task).filter(db_mod.Task.id == legacy_task_id).first()

    assert execute_resp.success is True
    assert captured["name"] == "data.acquire"
    assert captured["kwargs"]["source"] == "root.sg.dev"
    assert captured["kwargs"]["target_points"] == 2000
    assert captured["kwargs"]["user"] == "root"
    assert captured["kwargs"]["password"] == "root"
    assert captured["kwargs"]["point_name"] == "FI_1.PV"
    assert legacy_task is not None
    legacy_cfg = json.loads(legacy_task.config or "{}")
    assert legacy_cfg.get("executor") == "task_center"
    assert legacy_cfg.get("password") == "***"
    assert legacy_cfg.get("celery_task_id") == "celery-acquire-1"
    assert (status_resp.data or {}).get("status") == "running"


def test_task_center_status_syncs_from_legacy_acquire_task(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    def _fake_send_task(_name, kwargs=None, **_):
        task_id = str((kwargs or {}).get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    run_req = TaskCenterRunCreateRequest(
        task_type="acquire",
        trigger_mode="manual",
        steps=["acquire"],
        input_payload={"source": "root.sg.dev"},
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        run_id = str((create_resp.data or {}).get("run_id") or "")
        asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )

        step_row = (
            db.query(db_mod.TaskCenterStepRun)
            .filter(db_mod.TaskCenterStepRun.run_id == run_id)
            .first()
        )
        assert step_row is not None
        step_result = json.loads(step_row.result or "{}")
        legacy_task_id = str(step_result.get("legacy_task_id") or "")
        legacy_task = db.query(db_mod.Task).filter(db_mod.Task.id == legacy_task_id).first()
        assert legacy_task is not None
        legacy_task.status = "completed"
        legacy_task.result = json.dumps({"success": True, "message": "done"})
        db.commit()

        status_resp = asyncio.run(task_center_api.get_run_status(run_id, db=db))

    assert status_resp.success is True
    assert (status_resp.data or {}).get("status") == "completed"


def test_task_center_step_specs_dependency_progression(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    calls = []

    def _fake_send_task(name, kwargs=None, **_):
        kwargs = kwargs or {}
        calls.append({"name": name, "kwargs": dict(kwargs)})
        task_id = str(kwargs.get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    run_req = TaskCenterRunCreateRequest(
        task_type="pipeline",
        trigger_mode="manual",
        step_specs=[
            {"name": "acquire"},
            {"name": "inference", "depends_on": ["acquire"]},
            {"name": "training", "depends_on": ["inference"]},
        ],
        input_payload={
            "source": "root.sg.dev",
            "model": "/tmp/model",
            "algorithm": "chatts",
            "input_files": ["/tmp/a.csv"],
            "config_name": "train_cfg_a",
        },
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        run_id = str((create_resp.data or {}).get("run_id") or "")
        steps_created = (create_resp.data or {}).get("steps") or []
        assert len(steps_created) == 3
        assert steps_created[0].get("status") == "runnable"
        assert steps_created[1].get("status") == "blocked"
        assert steps_created[2].get("status") == "blocked"
        assert steps_created[1].get("depends_on") == ["acquire"]
        assert steps_created[2].get("depends_on") == ["inference"]

        asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )
        assert len(calls) == 1
        assert calls[0]["name"] == "data.acquire"

        acquire_step = (
            db.query(db_mod.TaskCenterStepRun)
            .filter(
                db_mod.TaskCenterStepRun.run_id == run_id,
                db_mod.TaskCenterStepRun.step_name == "acquire",
            )
            .first()
        )
        assert acquire_step is not None
        acquire_result = json.loads(acquire_step.result or "{}")
        acquire_legacy_id = str(acquire_result.get("legacy_task_id") or "")
        acquire_legacy = db.query(db_mod.Task).filter(db_mod.Task.id == acquire_legacy_id).first()
        assert acquire_legacy is not None
        acquire_legacy.status = "completed"
        acquire_legacy.result = json.dumps({"success": True})
        db.commit()

        asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )
        assert len(calls) == 2
        assert calls[1]["name"] == "inference.batch"

        inference_step = (
            db.query(db_mod.TaskCenterStepRun)
            .filter(
                db_mod.TaskCenterStepRun.run_id == run_id,
                db_mod.TaskCenterStepRun.step_name == "inference",
            )
            .first()
        )
        assert inference_step is not None
        inference_result = json.loads(inference_step.result or "{}")
        infer_legacy_id = str(inference_result.get("legacy_task_id") or "")
        infer_legacy = db.query(db_mod.Task).filter(db_mod.Task.id == infer_legacy_id).first()
        assert infer_legacy is not None
        infer_legacy.status = "completed"
        infer_legacy.result = json.dumps({"success": True})
        db.commit()

        asyncio.run(
            task_center_api.execute_run(
                run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )
        assert len(calls) == 3
        assert calls[2]["name"] == "training.run"


def test_task_center_list_runs_with_filters(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    with test_session_local() as db:
        run1 = db_mod.TaskCenterRun(
            id="run-list-1",
            definition_id="def-a",
            task_type="inference",
            trigger_mode="manual",
            status="completed",
        )
        run2 = db_mod.TaskCenterRun(
            id="run-list-2",
            definition_id="def-b",
            task_type="training",
            trigger_mode="schedule",
            status="running",
        )
        db.add(run1)
        db.add(run2)
        db.add(db_mod.TaskCenterStepRun(run_id="run-list-1", step_name="inference", status="completed"))
        db.add(db_mod.TaskCenterStepRun(run_id="run-list-2", step_name="training", status="running"))
        db.commit()

        all_resp = asyncio.run(task_center_api.list_runs(limit=50, offset=0, db=db))
        filtered_resp = asyncio.run(
            task_center_api.list_runs(
                status="completed",
                task_type="inference",
                trigger_mode="manual",
                limit=50,
                offset=0,
                db=db,
            )
        )

    assert all_resp.success is True
    all_data = all_resp.data or {}
    assert all_data.get("total") == 2
    runs = all_data.get("runs") if isinstance(all_data.get("runs"), list) else []
    assert len(runs) == 2

    assert filtered_resp.success is True
    filtered_data = filtered_resp.data or {}
    filtered_runs = filtered_data.get("runs") if isinstance(filtered_data.get("runs"), list) else []
    assert filtered_data.get("total") == 1
    assert len(filtered_runs) == 1
    assert filtered_runs[0].get("run_id") == "run-list-1"
    assert filtered_runs[0].get("status") == "completed"


def test_task_center_auto_retry_on_failed_run(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {"calls": 0}

    def _fake_send_task(_name, kwargs=None, **_):
        captured["calls"] += 1
        task_id = str((kwargs or {}).get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    run_req = TaskCenterRunCreateRequest(
        task_type="inference",
        trigger_mode="manual",
        steps=["inference"],
        input_payload={
            "model": "/tmp/model",
            "algorithm": "chatts",
            "input_files": ["/tmp/a.csv"],
        },
        max_retries=1,
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        source_run_id = str((create_resp.data or {}).get("run_id") or "")
        asyncio.run(
            task_center_api.execute_run(
                source_run_id,
                request=TaskCenterRunExecuteRequest(simulate=False),
                db=db,
            )
        )

        source_step = (
            db.query(db_mod.TaskCenterStepRun)
            .filter(db_mod.TaskCenterStepRun.run_id == source_run_id)
            .first()
        )
        assert source_step is not None
        source_step_result = json.loads(source_step.result or "{}")
        legacy_task_id = str(source_step_result.get("legacy_task_id") or "")
        legacy_task = db.query(db_mod.Task).filter(db_mod.Task.id == legacy_task_id).first()
        assert legacy_task is not None
        legacy_task.status = "failed"
        legacy_task.error = "boom"
        db.commit()

        status_resp = asyncio.run(task_center_api.get_run_status(source_run_id, db=db))
        assert status_resp.success is True
        assert (status_resp.data or {}).get("status") == "failed"

        runs = db.query(db_mod.TaskCenterRun).order_by(db_mod.TaskCenterRun.created_at.asc()).all()
        assert len(runs) == 2
        retry_payload = json.loads(runs[1].input_payload or "{}")
        retry_meta = retry_payload.get("__task_center") or {}
        assert retry_meta.get("retry_of_run_id") == source_run_id
        assert retry_meta.get("retry_attempt") == 1

        # 重复同步不应再次创建重试 run
        asyncio.run(task_center_api.get_run_status(source_run_id, db=db))
        assert db.query(db_mod.TaskCenterRun).count() == 2

    assert captured["calls"] == 2


def test_task_center_run_timeout_transition(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)

    with test_session_local() as db:
        run = db_mod.TaskCenterRun(
            id="run-timeout-1",
            task_type="inference",
            trigger_mode="manual",
            status="running",
            input_payload=json.dumps(
                {
                    "model": "/tmp/model",
                    "input_files": ["/tmp/a.csv"],
                    "__task_center": {"timeout_sec": 1},
                }
            ),
            started_at=task_center_api.utc_now_naive() - timedelta(seconds=5),
        )
        db.add(run)
        db.add(
            db_mod.TaskCenterStepRun(
                run_id="run-timeout-1",
                step_name="inference",
                status="running",
                started_at=task_center_api.utc_now_naive() - timedelta(seconds=5),
            )
        )
        db.commit()

        status_resp = asyncio.run(task_center_api.get_run_status("run-timeout-1", db=db))

    assert status_resp.success is True
    status_data = status_resp.data or {}
    assert status_data.get("status") == "timeout"
    steps = status_data.get("steps") if isinstance(status_data.get("steps"), list) else []
    assert len(steps) == 1
    assert steps[0].get("status") == "timeout"


def test_task_center_auto_retry_respects_delay(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {"calls": 0}

    def _fake_send_task(_name, kwargs=None, **_):
        captured["calls"] += 1
        task_id = str((kwargs or {}).get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    run_req = TaskCenterRunCreateRequest(
        task_type="inference",
        trigger_mode="manual",
        steps=["inference"],
        input_payload={"model": "/tmp/model", "algorithm": "chatts", "input_files": ["/tmp/a.csv"]},
        max_retries=1,
        retry_delay_sec=30,
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        source_run_id = str((create_resp.data or {}).get("run_id") or "")
        asyncio.run(task_center_api.execute_run(source_run_id, request=TaskCenterRunExecuteRequest(simulate=False), db=db))

        source_step = db.query(db_mod.TaskCenterStepRun).filter(db_mod.TaskCenterStepRun.run_id == source_run_id).first()
        assert source_step is not None
        legacy_task_id = str((json.loads(source_step.result or "{}")).get("legacy_task_id") or "")
        legacy_task = db.query(db_mod.Task).filter(db_mod.Task.id == legacy_task_id).first()
        assert legacy_task is not None

        base = task_center_api.utc_now_naive()
        legacy_task.status = "failed"
        legacy_task.error = "transient network"
        legacy_task.completed_at = base
        db.commit()

        _t10 = lambda: base + timedelta(seconds=10)
        monkeypatch.setattr(task_center_api, "utc_now_naive", _t10)
        monkeypatch.setattr(_sm_mod, "utc_now_naive", _t10)
        monkeypatch.setattr(_ro_mod, "utc_now_naive", _t10)
        asyncio.run(task_center_api.get_run_status(source_run_id, db=db))
        assert db.query(db_mod.TaskCenterRun).count() == 1

        _t35 = lambda: base + timedelta(seconds=35)
        monkeypatch.setattr(task_center_api, "utc_now_naive", _t35)
        monkeypatch.setattr(_sm_mod, "utc_now_naive", _t35)
        monkeypatch.setattr(_ro_mod, "utc_now_naive", _t35)
        asyncio.run(task_center_api.get_run_status(source_run_id, db=db))
        assert db.query(db_mod.TaskCenterRun).count() == 2

    assert captured["calls"] == 2


def test_task_center_auto_retry_respects_error_filter(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {"calls": 0}

    def _fake_send_task(_name, kwargs=None, **_):
        captured["calls"] += 1
        task_id = str((kwargs or {}).get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    run_req = TaskCenterRunCreateRequest(
        task_type="inference",
        trigger_mode="manual",
        steps=["inference"],
        input_payload={"model": "/tmp/model", "algorithm": "chatts", "input_files": ["/tmp/a.csv"]},
        max_retries=1,
        retry_on_errors=["transient", "timeout"],
        auto_execute=False,
    )

    with test_session_local() as db:
        create_resp = asyncio.run(task_center_api.create_run(run_req, db=db))
        source_run_id = str((create_resp.data or {}).get("run_id") or "")
        asyncio.run(task_center_api.execute_run(source_run_id, request=TaskCenterRunExecuteRequest(simulate=False), db=db))

        source_step = db.query(db_mod.TaskCenterStepRun).filter(db_mod.TaskCenterStepRun.run_id == source_run_id).first()
        assert source_step is not None
        legacy_task_id = str((json.loads(source_step.result or "{}")).get("legacy_task_id") or "")
        legacy_task = db.query(db_mod.Task).filter(db_mod.Task.id == legacy_task_id).first()
        assert legacy_task is not None
        legacy_task.status = "failed"
        legacy_task.error = "validation failed"
        db.commit()

        asyncio.run(task_center_api.get_run_status(source_run_id, db=db))
        assert db.query(db_mod.TaskCenterRun).count() == 1

        legacy_task.error = "transient network failed"
        db.commit()
        asyncio.run(task_center_api.get_run_status(source_run_id, db=db))
        assert db.query(db_mod.TaskCenterRun).count() == 2

    assert captured["calls"] == 2


def test_task_center_auto_retry_exponential_backoff(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    captured = {"calls": 0}

    def _fake_send_task(_name, kwargs=None, **_):
        captured["calls"] += 1
        task_id = str((kwargs or {}).get("task_id") or "")
        return _FakeAsyncResult(f"celery-{task_id}")

    monkeypatch.setattr(task_center_api.celery_app, "send_task", _fake_send_task)

    base = task_center_api.utc_now_naive()
    with test_session_local() as db:
        run = db_mod.TaskCenterRun(
            id="run-exp-1",
            task_type="inference",
            trigger_mode="manual",
            status="failed",
            input_payload=json.dumps(
                {
                    "model": "/tmp/model",
                    "algorithm": "chatts",
                    "input_files": ["/tmp/a.csv"],
                    "__task_center": {
                        "max_retries": 2,
                        "retry_attempt": 1,
                        "retry_policy": "exponential",
                        "retry_delay_sec": 10,
                        "retry_backoff_factor": 2.0,
                    },
                }
            ),
            completed_at=base,
        )
        db.add(run)
        db.add(db_mod.TaskCenterStepRun(run_id="run-exp-1", step_name="inference", status="failed"))
        db.commit()

        _t19 = lambda: base + timedelta(seconds=19)
        monkeypatch.setattr(task_center_api, "utc_now_naive", _t19)
        monkeypatch.setattr(_sm_mod, "utc_now_naive", _t19)
        monkeypatch.setattr(_ro_mod, "utc_now_naive", _t19)
        asyncio.run(task_center_api.get_run_status("run-exp-1", db=db))
        assert db.query(db_mod.TaskCenterRun).count() == 1

        _t21 = lambda: base + timedelta(seconds=21)
        monkeypatch.setattr(task_center_api, "utc_now_naive", _t21)
        monkeypatch.setattr(_sm_mod, "utc_now_naive", _t21)
        monkeypatch.setattr(_ro_mod, "utc_now_naive", _t21)
        asyncio.run(task_center_api.get_run_status("run-exp-1", db=db))
        assert db.query(db_mod.TaskCenterRun).count() == 2

    assert captured["calls"] == 1

def test_task_center_concurrency_limit_by_scheduler(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    from src.task_center.scheduler import run_scheduler_tick

    with test_session_local() as db:
        # 创建一个要求并发=1 的 definition (启用它以便 scheduler 能扫描到)
        req = TaskCenterDefinitionRequest(
            name="limit-test",
            trigger_mode="auto",
            config={"max_concurrent_runs": 1}
        )
        resp = asyncio.run(task_center_api.create_definition(req, db=db))
        def_id = resp.data["id"]

        # 原本没有任务，tick 一次应该会创建一个
        # 为 scheduler 的 SessionLocal 注入 test DB engine 绑定
        monkeypatch.setattr("src.task_center.scheduler.SessionLocal", test_session_local)
        
        stats1 = run_scheduler_tick(execute_mode="simulate")
        assert stats1.created == 1
        
        # 将刚刚创建的任务状态改为 running，模拟正在执行
        run = db.query(db_mod.TaskCenterRun).filter_by(definition_id=def_id).first()
        run.status = "running"
        db.commit()

        # 第二次 tick，因为并发为1，此刻已有一个 running，理应被 limit 挡住，不创建新 run
        stats2 = run_scheduler_tick(execute_mode="simulate")
        assert stats2.created == 0
        assert stats2.skipped >= 1


def test_task_center_concurrency_limit_by_event(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENTS_ENABLED", True)

    with test_session_local() as db:
        # 并发 = 1
        req = TaskCenterDefinitionRequest(
            name="event-limit",
            trigger_mode="event",
            config={"max_concurrent_runs": 1, "event_key": "limit_evt"}
        )
        asyncio.run(task_center_api.create_definition(req, db=db))

        # 触发第一次，应该成功
        trigger_req = TaskCenterEventTriggerRequest(event_key="limit_evt", execute_mode="simulate")
        resp1 = asyncio.run(task_center_api.trigger_event(trigger_req, db=db))
        assert len(resp1.data["run_ids"]) == 1
        assert len(resp1.data.get("rate_limited_definition_ids", [])) == 0

        # 手动把刚刚那个设为 running
        run = db.query(db_mod.TaskCenterRun).first()
        run.status = "running"
        db.commit()

        # 触发第二次，应该被拦下
        resp2 = asyncio.run(task_center_api.trigger_event(trigger_req, db=db))
        assert len(resp2.data["run_ids"]) == 0
        assert len(resp2.data.get("rate_limited_definition_ids", [])) == 1
        assert "限流拦截" in resp2.message

