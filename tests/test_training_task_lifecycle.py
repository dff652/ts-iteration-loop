import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core import tasks as core_tasks
from src.db import database as db_mod


def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)
    return test_session_local


def test_run_training_task_waits_for_completion_and_persists_result(tmp_path, monkeypatch):
    test_session_local = _setup_db(tmp_path, monkeypatch)
    monkeypatch.setattr(core_tasks.run_training_task, "update_state", lambda *args, **kwargs: None)

    captured = {}

    class _FakeAdapter:
        def __init__(self, model_family="chatts"):
            self.model_family = model_family

        def run_training(self, **kwargs):
            captured.update(kwargs)
            return {"success": True, "output_dir": "/tmp/model", "return_code": 0}

    monkeypatch.setattr("src.adapters.chatts_training.ChatTSTrainingAdapter", _FakeAdapter)

    with test_session_local() as db:
        db.add(
            db_mod.Task(
                id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                type="training",
                status="pending",
                config=json.dumps({"config_name": "demo"}),
            )
        )
        db.commit()

    result = core_tasks.run_training_task.run(
        task_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        config_name="demo",
        version_tag="v1",
        model_family="chatts",
        auto_eval=False,
        params={"override_learning_rate": "1e-5"},
    )

    assert result["success"] is True
    assert captured.get("wait_for_completion") is True
    assert callable(captured.get("should_cancel"))

    with test_session_local() as db:
        task = db.query(db_mod.Task).filter(db_mod.Task.id == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa").first()
        assert task is not None
        assert task.status == "completed"
        saved = json.loads(task.result or "{}")
        assert saved.get("success") is True


def test_run_training_task_marks_cancelled_when_adapter_reports_cancelled(tmp_path, monkeypatch):
    test_session_local = _setup_db(tmp_path, monkeypatch)
    monkeypatch.setattr(core_tasks.run_training_task, "update_state", lambda *args, **kwargs: None)

    class _FakeAdapter:
        def __init__(self, model_family="chatts"):
            self.model_family = model_family

        def run_training(self, **kwargs):
            return {"success": False, "cancelled": True, "return_code": -15}

    monkeypatch.setattr("src.adapters.chatts_training.ChatTSTrainingAdapter", _FakeAdapter)

    with test_session_local() as db:
        db.add(
            db_mod.Task(
                id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                type="training",
                status="pending",
                config=json.dumps({"config_name": "demo"}),
            )
        )
        db.commit()

    result = core_tasks.run_training_task.run(
        task_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        config_name="demo",
        version_tag="v2",
        model_family="chatts",
        auto_eval=False,
        params={},
    )

    assert result.get("cancelled") is True
    with test_session_local() as db:
        task = db.query(db_mod.Task).filter(db_mod.Task.id == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb").first()
        assert task is not None
        assert task.status == "cancelled"
