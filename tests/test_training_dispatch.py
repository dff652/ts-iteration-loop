import asyncio
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import training as training_api
from src.db import database as db_mod
from src.models.schemas import TrainingTaskRequest


class _FakeAsyncResult:
    def __init__(self, task_id: str):
        self.id = task_id


def test_start_training_dispatches_to_celery(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    captured = {}

    def _fake_send_task(name, kwargs=None, **_):
        captured["name"] = name
        captured["kwargs"] = kwargs or {}
        return _FakeAsyncResult("celery-training-1")

    monkeypatch.setattr(training_api.celery_app, "send_task", _fake_send_task)

    req = TrainingTaskRequest(
        config_name="chatts_lora",
        version_tag="v-test",
        model_family="qwen",
        auto_eval=True,
        eval_truth_dir="/tmp/truth",
        eval_data_dir="/tmp/data",
        eval_dataset_name="golden",
        eval_output_dir="/tmp/output",
        eval_device="cuda:0",
        eval_method="chatts",
        params={"override_learning_rate": "3e-5"},
    )

    with test_session_local() as db:
        resp = asyncio.run(training_api.start_training(req, db=db))
        task = db.query(db_mod.Task).filter(db_mod.Task.id == resp.task_id).first()

    assert resp.status == "pending"
    assert captured["name"] == "training.run"
    assert captured["kwargs"]["task_id"] == resp.task_id
    assert captured["kwargs"]["model_family"] == "qwen"
    assert captured["kwargs"]["auto_eval"] is True
    assert captured["kwargs"]["params"]["override_learning_rate"] == "3e-5"
    assert task is not None
    cfg = json.loads(task.config or "{}")
    assert cfg.get("executor") == "celery"
    assert cfg.get("celery_task_id") == "celery-training-1"


def test_stop_training_revokes_celery_and_marks_cancelled(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    revoked = {}

    def _fake_revoke(task_id, terminate=False, **_):
        revoked["task_id"] = task_id
        revoked["terminate"] = terminate

    monkeypatch.setattr(training_api.celery_app.control, "revoke", _fake_revoke)

    with test_session_local() as db:
        task = db_mod.Task(
            id="11111111-1111-1111-1111-111111111111",
            type="training",
            status="running",
            config=json.dumps({"model_family": "chatts", "celery_task_id": "celery-x"}),
        )
        db.add(task)
        db.commit()

        resp = asyncio.run(training_api.stop_training(task.id, db=db))
        db.refresh(task)

    assert resp.status == "cancelled"
    assert revoked["task_id"] == "celery-x"
    assert revoked["terminate"] is True
    assert task.status == "cancelled"


def test_get_training_task_log_reads_incremental_content(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    save_root = tmp_path / "saves" / "chatts"
    save_root.mkdir(parents=True, exist_ok=True)

    class _FakeAdapter:
        def __init__(self, saves_path: Path):
            self.saves_path = saves_path

    monkeypatch.setattr(training_api, "get_adapter", lambda _family="chatts": _FakeAdapter(save_root))

    task_id = "22222222-2222-2222-2222-222222222222"
    output_dir = save_root / "chatts_lora_v-test"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="training",
            status="running",
            config=json.dumps({
                "config_name": "chatts_lora",
                "version_tag": "v-test",
                "model_family": "chatts",
            }),
        )
        db.add(task)
        db.commit()

        resp1 = asyncio.run(training_api.get_training_task_log(task_id, offset=0, max_bytes=8, db=db))
        offset_1 = int((resp1.data or {}).get("offset", 0))
        resp2 = asyncio.run(training_api.get_training_task_log(task_id, offset=offset_1, max_bytes=100, db=db))

    assert resp1.success is True
    assert (resp1.data or {}).get("status") == "running"
    assert (resp1.data or {}).get("exists") is True
    assert (resp1.data or {}).get("log") == "line1\nli"
    assert offset_1 == 8
    assert (resp2.data or {}).get("log") == "ne2\nline3\n"


def test_get_training_status_includes_paths_and_message(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    save_root = tmp_path / "saves" / "chatts"
    save_root.mkdir(parents=True, exist_ok=True)

    class _FakeAdapter:
        def __init__(self, saves_path: Path):
            self.saves_path = saves_path

        def get_training_progress(self, _task_id: str):
            return {"status": "running", "progress": 0}

    monkeypatch.setattr(training_api, "get_adapter", lambda _family="chatts": _FakeAdapter(save_root))

    task_id = "33333333-3333-3333-3333-333333333333"
    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="training",
            status="running",
            config=json.dumps({
                "config_name": "chatts_lora",
                "version_tag": "v-test",
                "model_family": "chatts",
            }),
        )
        db.add(task)
        db.commit()

        resp = asyncio.run(training_api.get_training_status(task_id, db=db))

    data = resp.data or {}
    assert resp.success is True
    assert data.get("status") == "running"
    assert data.get("message") == "训练进行中"
    assert data.get("output_dir") == str(save_root / "chatts_lora_v-test")
    assert data.get("log_path") == str((save_root / "chatts_lora_v-test") / "train.log")
