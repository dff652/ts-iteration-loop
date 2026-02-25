import asyncio
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import inference as inference_api
from src.db import database as db_mod
from src.models.schemas import InferenceTaskRequest
from configs.settings import settings


class _FakeAsyncResult:
    def __init__(self, task_id: str):
        self.id = task_id


def test_start_batch_inference_dispatches_to_celery(tmp_path, monkeypatch):
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
        return _FakeAsyncResult("celery-test-1")

    monkeypatch.setattr(inference_api.celery_app, "send_task", _fake_send_task)

    req = InferenceTaskRequest(
        model="/tmp/model",
        algorithm="chatts",
        input_files=["/tmp/a.csv", "/tmp/b.csv"],
    )

    with test_session_local() as db:
        resp = asyncio.run(inference_api.start_batch_inference(req, db=db))
        task = db.query(db_mod.Task).filter(db_mod.Task.id == resp.task_id).first()

    assert resp.status == "pending"
    assert captured["name"] == "inference.batch"
    assert captured["kwargs"]["task_id"] == resp.task_id
    assert captured["kwargs"]["algorithm"] == "chatts"
    assert captured["kwargs"]["input_files"] == ["/tmp/a.csv", "/tmp/b.csv"]
    assert task is not None
    cfg = json.loads(task.config or "{}")
    assert cfg.get("executor") == "celery"
    assert cfg.get("celery_task_id") == "celery-test-1"


def test_get_inference_status_contract(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    task_id = "66666666-6666-6666-6666-666666666666"
    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="inference",
            status="running",
            config=json.dumps({"algorithm": "chatts"}),
        )
        db.add(task)
        db.commit()
        resp = asyncio.run(inference_api.get_inference_status(task_id, db=db))

    data = resp.data or {}
    assert resp.success is True
    assert data.get("task_id") == task_id
    assert data.get("status") == "running"
    assert data.get("type") == "inference"
    assert data.get("message") == "推理执行中"
    assert data.get("output_dir") == f"{settings.DATA_INFERENCE_DIR}/chatts"
    assert isinstance(data.get("progress"), dict)


def test_get_inference_log_contract_incremental(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    task_id = "77777777-7777-7777-7777-777777777777"
    result_payload = {
        "total": 2,
        "successful": 1,
        "errors": [{"file": "a.csv", "error": "boom"}],
        "results": [{"file": "a.csv", "success": False}, {"file": "b.csv", "success": True}],
    }
    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="inference",
            status="failed",
            error="first error",
            result=json.dumps(result_payload),
            config=json.dumps({"algorithm": "chatts"}),
        )
        db.add(task)
        db.commit()
        resp1 = asyncio.run(inference_api.get_inference_log(task_id, offset=0, max_bytes=18, db=db))
        offset_1 = int((resp1.data or {}).get("offset", 0))
        resp2 = asyncio.run(inference_api.get_inference_log(task_id, offset=offset_1, max_bytes=400, db=db))

    assert resp1.success is True
    assert (resp1.data or {}).get("status") == "failed"
    assert (resp1.data or {}).get("exists") is True
    assert (resp1.data or {}).get("log") == "summary: total=2, "
    tail = str((resp2.data or {}).get("log") or "")
    assert "successful=1" in tail
    assert "error: a.csv | boom" in tail
    assert "task_error: first error" in tail


def test_export_to_annotation_returns_rows_by_default(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    task_id = "88888888-8888-8888-8888-888888888888"
    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="inference",
            status="completed",
            result=json.dumps({"results": [{"file": "a.csv", "result": {"detected_anomalies": []}}]}),
        )
        db.add(task)
        db.commit()

        calls = {"rows_payload": None, "convert_called": False}

        def _fake_rows(payload):
            calls["rows_payload"] = payload
            return [{"filename": "a.csv", "annotations": [], "source": "inference"}]

        def _fake_convert(_payload):
            calls["convert_called"] = True
            return "/tmp/should-not-be-used.json"

        monkeypatch.setattr(inference_api.adapter, "to_annotation_rows", _fake_rows)
        monkeypatch.setattr(inference_api.adapter, "convert_to_annotation_format", _fake_convert)

        resp = asyncio.run(inference_api.export_to_annotation(task_id, db=db))

    data = resp.data or {}
    assert resp.success is True
    assert data.get("row_count") == 1
    assert isinstance(data.get("rows"), list)
    assert data.get("annotation_file") is None
    assert calls["rows_payload"] == task.result
    assert calls["convert_called"] is False


def test_export_to_annotation_persist_file_mode(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    task_id = "99999999-9999-9999-9999-999999999999"
    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="inference",
            status="completed",
            result=json.dumps({"results": []}),
        )
        db.add(task)
        db.commit()

        calls = {"rows_payload": None, "convert_payload": None}

        def _fake_rows(payload):
            calls["rows_payload"] = payload
            return []

        def _fake_convert(payload):
            calls["convert_payload"] = payload
            return "/tmp/inference_annotations_compat.json"

        monkeypatch.setattr(inference_api.adapter, "to_annotation_rows", _fake_rows)
        monkeypatch.setattr(inference_api.adapter, "convert_to_annotation_format", _fake_convert)

        resp = asyncio.run(inference_api.export_to_annotation(task_id, persist_file=True, db=db))

    data = resp.data or {}
    assert resp.success is True
    assert data.get("row_count") == 0
    assert data.get("rows") == []
    assert data.get("annotation_file") == "/tmp/inference_annotations_compat.json"
    assert calls["rows_payload"] == task.result
    assert calls["convert_payload"] == task.result


def test_export_to_annotation_requires_completed_task(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    task_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    with test_session_local() as db:
        task = db_mod.Task(
            id=task_id,
            type="inference",
            status="running",
            result=json.dumps({"results": []}),
        )
        db.add(task)
        db.commit()

        calls = {"rows_called": False, "convert_called": False}

        def _fake_rows(_payload):
            calls["rows_called"] = True
            return []

        def _fake_convert(_payload):
            calls["convert_called"] = True
            return "/tmp/not-used.json"

        monkeypatch.setattr(inference_api.adapter, "to_annotation_rows", _fake_rows)
        monkeypatch.setattr(inference_api.adapter, "convert_to_annotation_format", _fake_convert)

        resp = asyncio.run(inference_api.export_to_annotation(task_id, db=db))

    assert resp.success is False
    assert "任务尚未完成" in (resp.message or "")
    assert (resp.data or {}).get("status") == "running"
    assert calls["rows_called"] is False
    assert calls["convert_called"] is False
