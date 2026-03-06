import asyncio
import io
import json

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import data as data_api
from src.db import database as db_mod
from src.models.schemas import AcquireTaskRequest


def test_upload_dataset_csv_create_duplicate_and_overwrite(tmp_path, monkeypatch):
    data_dir = tmp_path / "downsampled"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(data_api.adapter, "data_path", data_dir)

    file1 = UploadFile(filename="a.csv", file=io.BytesIO(b"t,v\n1,10\n"))
    resp1 = asyncio.run(data_api.upload_dataset_csv(file=file1, dataset_name="my dataset", overwrite=False))
    assert resp1.success is True
    saved = data_dir / "my_dataset.csv"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == "t,v\n1,10\n"

    file2 = UploadFile(filename="b.csv", file=io.BytesIO(b"t,v\n2,20\n"))
    with pytest.raises(HTTPException) as e_info:
        asyncio.run(data_api.upload_dataset_csv(file=file2, dataset_name="my dataset", overwrite=False))
    assert e_info.value.status_code == 400
    assert "数据集已存在" in str(e_info.value.detail)

    file3 = UploadFile(filename="c.csv", file=io.BytesIO(b"t,v\n3,30\n"))
    resp3 = asyncio.run(data_api.upload_dataset_csv(file=file3, dataset_name="my dataset", overwrite=True))
    assert resp3.success is True
    assert saved.read_text(encoding="utf-8") == "t,v\n3,30\n"


def test_create_dataset_from_iotdb_creates_task(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(data_api, "_dispatch_acquire_task", lambda task_id, request: "celery-test-id")

    request = AcquireTaskRequest(
        source="root.demo.path",
        host="127.0.0.1",
        port="6667",
        user="demo",
        password="secret",
        point_name="*",
        target_points=5000,
        start_time="2026-03-01 00:00:00",
        end_time="2026-03-01 23:59:59",
    )

    with test_session_local() as db:
        resp = asyncio.run(data_api.create_dataset_from_iotdb(request=request, db=db))
    assert resp.task_id
    assert resp.status == "pending"

    with test_session_local() as db:
        task = db.query(db_mod.Task).filter(db_mod.Task.id == resp.task_id).first()
        assert task is not None
        assert task.type == "acquire"
        assert task.status == "pending"
        cfg = json.loads(task.config or "{}")
        assert cfg.get("password") == "***"
        assert cfg.get("celery_task_id") == "celery-test-id"
