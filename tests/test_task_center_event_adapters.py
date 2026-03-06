import asyncio
import hmac
import json
import time
from hashlib import sha256

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import task_center as task_center_api
from src.db import database as db_mod
from src.task_center import event_adapters as adapters_mod


def _setup_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    monkeypatch.setattr(adapters_mod, "SessionLocal", test_session_local)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_EVENTS_ENABLED", True)
    db_mod.Base.metadata.create_all(bind=engine)
    return test_session_local


class _FakeRequest:
    def __init__(self, body: bytes, headers: dict[str, str]):
        self._body = body
        self.headers = headers

    async def body(self) -> bytes:
        return self._body


def test_webhook_signature_helper():
    secret = "s3cr3t"
    body = b'{"event_key":"k"}'
    sig = hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    assert task_center_api._verify_webhook_signature(secret, body, f"sha256={sig}") is True  # noqa: SLF001
    assert task_center_api._verify_webhook_signature(secret, body, sig) is True  # noqa: SLF001
    assert task_center_api._verify_webhook_signature(secret, body, "bad") is False  # noqa: SLF001


def test_webhook_trigger_with_signature_and_dedupe(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(task_center_api.settings, "TASK_CENTER_WEBHOOK_SECRETS", "sensor=abc123")

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-webhook-1",
                name="webhook-event",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps(
                    {
                        "event_key": "sensor.arrived",
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

        payload = {
            "event_key": "sensor.arrived",
            "definition_id": "def-webhook-1",
            "dedupe_key": "evt-1",
            "execute_mode": "none",
            "payload": {"value": 1},
        }
        body = json.dumps(payload).encode("utf-8")
        sig = hmac.new(b"abc123", body, sha256).hexdigest()
        req = _FakeRequest(body, {"X-TaskCenter-Signature": f"sha256={sig}"})

        first = asyncio.run(task_center_api.trigger_event_webhook("sensor", req, db=db))
        second = asyncio.run(task_center_api.trigger_event_webhook("sensor", req, db=db))

        runs = db.query(db_mod.TaskCenterRun).all()

    assert first.success is True
    assert (first.data or {}).get("created_count") == 1
    assert second.success is True
    assert (second.data or {}).get("created_count") == 0
    assert len(runs) == 1


def test_file_watch_adapter_scan_create_then_dedupe(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    (watch_dir / "evt_1.json").write_text('{"k":1}', encoding="utf-8")

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-file-watch-1",
                name="file-watch-event",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps(
                    {
                        "adapter": "file_watch",
                        "watch_dir": str(watch_dir),
                        "file_glob": "*.json",
                        "event_key": "file.arrived",
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

    stats_1 = adapters_mod.run_file_event_scan(execute_mode="none")
    stats_2 = adapters_mod.run_file_event_scan(execute_mode="none")

    with test_session_local() as db:
        runs = db.query(db_mod.TaskCenterRun).all()

    assert stats_1.created_runs == 1
    assert stats_2.created_runs == 0
    assert stats_2.deduped >= 1
    assert len(runs) == 1


def test_file_watch_adapter_post_action_move(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    watch_dir = tmp_path / "watch_move"
    archive_dir = tmp_path / "archive_move"
    watch_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    src_file = watch_dir / "evt_1.json"
    src_file.write_text('{"k":1}', encoding="utf-8")

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-file-watch-move-1",
                name="file-watch-move",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps(
                    {
                        "adapter": "file_watch",
                        "watch_dir": str(watch_dir),
                        "file_glob": "*.json",
                        "event_key": "file.arrived",
                        "post_action": "move",
                        "archive_dir": str(archive_dir),
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

    stats = adapters_mod.run_file_event_scan(execute_mode="none")

    moved_files = list(archive_dir.glob("evt_1*.json"))
    assert stats.created_runs == 1
    assert stats.moved_files == 1
    assert src_file.exists() is False
    assert len(moved_files) == 1


def test_file_watch_adapter_post_action_move_by_date(tmp_path, monkeypatch):
    test_session_local = _setup_test_db(tmp_path, monkeypatch)
    watch_dir = tmp_path / "watch_move_date"
    archive_dir = tmp_path / "archive_move_date"
    watch_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    src_file = watch_dir / "evt_2.json"
    src_file.write_text('{"k":2}', encoding="utf-8")
    day = time.strftime("%Y-%m-%d", time.localtime(src_file.stat().st_mtime))

    with test_session_local() as db:
        db.add(
            db_mod.TaskCenterDefinition(
                id="def-file-watch-move-date-1",
                name="file-watch-move-date",
                task_type="inference",
                trigger_mode="event",
                enabled=True,
                config=json.dumps(
                    {
                        "adapter": "file_watch",
                        "watch_dir": str(watch_dir),
                        "file_glob": "*.json",
                        "event_key": "file.arrived",
                        "post_action": "move",
                        "archive_dir": str(archive_dir),
                        "archive_by_date": True,
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

    stats = adapters_mod.run_file_event_scan(execute_mode="none")

    moved_files = list((archive_dir / day).glob("evt_2*.json"))
    assert stats.created_runs == 1
    assert stats.moved_files == 1
    assert src_file.exists() is False
    assert len(moved_files) == 1
