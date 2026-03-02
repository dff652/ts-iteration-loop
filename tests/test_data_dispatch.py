import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import data as data_api
from src.db import database as db_mod
from src.models.schemas import AcquireTaskRequest


class _FakeAsyncResult:
    def __init__(self, task_id: str):
        self.id = task_id


def test_start_acquire_task_dispatches_to_celery(tmp_path, monkeypatch):
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
        return _FakeAsyncResult("celery-acquire-1")

    monkeypatch.setattr(data_api.celery_app, "send_task", _fake_send_task)

    req = AcquireTaskRequest(
        source="root.a.b",
        host="127.0.0.1",
        port="6667",
        user="root",
        password="root",
        point_name="FI_1.PV",
        target_points=5000,
        start_time="2025-01-01 00:00:00",
        end_time="2025-01-01 01:00:00",
    )

    with test_session_local() as db:
        resp = asyncio.run(data_api.start_acquire_task(req, db=db))
        task = db.query(db_mod.Task).filter(db_mod.Task.id == resp.task_id).first()

    assert resp.status == "pending"
    assert captured["name"] == "data.acquire"
    assert captured["kwargs"]["task_id"] == resp.task_id
    assert captured["kwargs"]["source"] == "root.a.b"
    assert captured["kwargs"]["password"] == "root"
    assert captured["kwargs"]["point_name"] == "FI_1.PV"
    assert task is not None
    cfg = json.loads(task.config or "{}")
    assert cfg.get("executor") == "celery"
    assert cfg.get("celery_task_id") == "celery-acquire-1"
    assert cfg.get("password") == "***"


def test_get_acquire_status_contract(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    data_api.adapter.data_path = Path(tmp_path / "downsampled")

    task_id = "44444444-4444-4444-4444-444444444444"
    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="acquire",
            status="running",
            config=json.dumps({"source": "root.a.b"}),
        )
        db.add(task)
        db.commit()
        resp = asyncio.run(data_api.get_task_status(task_id, db=db))

    data = resp.data or {}
    assert resp.success is True
    assert data.get("task_id") == task_id
    assert data.get("status") == "running"
    assert data.get("type") == "acquire"
    assert data.get("message") == "采集中"
    assert data.get("output_dir") == str(data_api.adapter.data_path)
    assert isinstance(data.get("progress"), dict)


def test_get_acquire_log_contract_incremental(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    task_id = "55555555-5555-5555-5555-555555555555"
    result_payload = {"stdout": "line-a\nline-b\n", "stderr": "warn-x\n"}
    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="acquire",
            status="completed",
            result=json.dumps(result_payload),
        )
        db.add(task)
        db.commit()
        resp1 = asyncio.run(data_api.get_task_log(task_id, offset=0, max_bytes=7, db=db))
        offset_1 = int((resp1.data or {}).get("offset", 0))
        resp2 = asyncio.run(data_api.get_task_log(task_id, offset=offset_1, max_bytes=100, db=db))

    assert resp1.success is True
    assert (resp1.data or {}).get("status") == "completed"
    assert (resp1.data or {}).get("exists") is True
    assert (resp1.data or {}).get("log") == "line-a\n"
    assert offset_1 == 7
    assert (resp2.data or {}).get("log") == "line-b\n\nwarn-x"


def test_preview_rejects_path_traversal(tmp_path):
    data_api.adapter.data_path = Path(tmp_path / "downsampled")
    data_api.adapter.data_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(data_api.preview_data("/etc/passwd", limit=10))
    assert exc_info.value.status_code == 400
