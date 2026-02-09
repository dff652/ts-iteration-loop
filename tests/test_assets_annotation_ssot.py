import asyncio
from datetime import datetime
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import assets as assets_api
from src.db import database as db_mod


def test_assets_sources_annotations_use_db_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(assets_api.settings, "DEFAULT_USER", "tester")

    with test_session_local() as db:
        db.add(
            db_mod.AnnotationRecord(
                id="ann-1",
                user_id="tester",
                source_id="P_1001",
                filename="P_1001.csv",
                source_kind="human",
                is_human_edited=True,
                annotation_count=1,
                segment_count=2,
                overall_attribute_json="{}",
                annotations_json='[{"id":"a1","segments":[{"start":1,"end":2},{"start":3,"end":5}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.AnnotationRecord(
                id="ann-2",
                user_id="tester",
                source_id="P_1002",
                filename="P_1002.csv",
                source_kind="auto",
                is_human_edited=False,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"auto_1","segments":[{"start":10,"end":12}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.AnnotationRecord(
                id="ann-3",
                user_id="tester",
                source_id="P_1003",
                filename="P_1003.csv",
                source_kind="auto",
                is_human_edited=False,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"auto_2","segments":[{"start":20,"end":25}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-1",
                task_id="task-1",
                method="chatts",
                model="m",
                point_name="P_1001",
                result_path="/tmp/P_1001.csv",
                metrics_path="",
                segments_path="",
                score_avg=0.91,
                score_max=0.95,
                segment_count=2,
                meta="{}",
                created_at=datetime.utcnow(),
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-2",
                task_id="task-2",
                method="chatts",
                model="m",
                point_name="P_1002",
                result_path="/tmp/P_1002.csv",
                metrics_path="",
                segments_path="",
                score_avg=0.42,
                score_max=0.55,
                segment_count=1,
                meta="{}",
                created_at=datetime.utcnow(),
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-3",
                task_id="task-3",
                method="chatts",
                model="m",
                point_name="P_1003",
                result_path="/tmp/P_1003.csv",
                metrics_path="",
                segments_path="",
                score_avg=0.73,
                score_max=0.81,
                segment_count=1,
                meta="{}",
                created_at=datetime.utcnow(),
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-1",
                source_type="annotation",
                source_id="P_1001",
                method="chatts",
                model=None,
                point_name="P_1001",
                score=0.91,
                strategy="topk",
                status="approved",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-2",
                source_type="annotation",
                source_id="P_1002",
                method="chatts",
                model=None,
                point_name="P_1002",
                score=0.42,
                strategy="topk",
                status="pending",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-3",
                source_type="annotation",
                source_id="P_1003",
                method="chatts",
                model=None,
                point_name="P_1003",
                score=0.73,
                strategy="topk",
                status="approved",
            )
        )
        db.commit()

    with test_session_local() as db:
        resp = asyncio.run(
            assets_api.list_source_items(
                source_type="annotations",
                source_kind="human",
                method="chatts",
                min_score=0.5,
                max_score=None,
                keyword=None,
                limit=20,
                db=db,
            )
        )

    assert resp.success is True
    choices = (resp.data or {}).get("choices") or []
    assert len(choices) == 1
    assert choices[0]["value"] == "P_1001"
    assert "Score: 0.91" in choices[0]["label"]
    assert "[HUMAN]" in choices[0]["label"]

    with test_session_local() as db:
        auto_resp = asyncio.run(
            assets_api.list_source_items(
                source_type="annotations",
                source_kind="auto",
                method=None,
                min_score=None,
                max_score=None,
                keyword=None,
                limit=20,
                db=db,
            )
        )

    assert auto_resp.success is True
    auto_choices = (auto_resp.data or {}).get("choices") or []
    assert len(auto_choices) == 1
    assert auto_choices[0]["value"] == "P_1003"
    assert "[AUTO]" in auto_choices[0]["label"]

    with test_session_local() as db:
        inf_resp = asyncio.run(
            assets_api.list_source_items(
                source_type="inference",
                source_kind=None,
                method="chatts",
                sort_by="score_asc",
                min_score=None,
                max_score=None,
                keyword=None,
                limit=20,
                db=db,
            )
        )

    assert inf_resp.success is True
    inf_choices = (inf_resp.data or {}).get("choices") or []
    assert len(inf_choices) == 3
    assert inf_choices[0]["value"] == "P_1002"
    assert "[INFERENCE]" in inf_choices[0]["label"]


def test_save_asset_rejects_unapproved_points(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    with test_session_local() as db:
        db.add(
            db_mod.ReviewQueue(
                id="rq-ok",
                source_type="annotation",
                source_id="P_approved",
                method="chatts",
                model=None,
                point_name="P_approved",
                score=0.88,
                strategy="topk",
                status="approved",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-no",
                source_type="annotation",
                source_id="P_pending",
                method="chatts",
                model=None,
                point_name="P_pending",
                score=0.52,
                strategy="topk",
                status="pending",
            )
        )
        db.commit()

    req = assets_api.AssetSaveRequest(
        name="train_case",
        dataset_type="train",
        items=["P_approved", "P_pending"],
        overwrite=False,
        freeze=False,
    )

    with test_session_local() as db:
        with pytest.raises(Exception) as exc:
            asyncio.run(assets_api.save_asset(req, db=db))
        assert "仅允许保存审核通过点位" in str(exc.value)

    ok_req = assets_api.AssetSaveRequest(
        name="train_case_ok",
        dataset_type="train",
        items=["P_approved"],
        overwrite=False,
        freeze=False,
    )
    with test_session_local() as db:
        resp = asyncio.run(assets_api.save_asset(ok_req, db=db))
    assert resp.success is True


def test_assets_sources_training_supports_approved_filter(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    chatts_dir = tmp_path / "chatts"
    qwen_dir = tmp_path / "qwen"
    chatts_dir.mkdir(parents=True, exist_ok=True)
    qwen_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(assets_api.settings, "DATA_TRAINING_CHATTS_DIR", str(chatts_dir))
    monkeypatch.setattr(assets_api.settings, "DATA_TRAINING_QWEN_DIR", str(qwen_dir))

    chatts_payload = [
        {"id": "P_1001.PV", "input": "x", "output": "y"},
        {"id": "P_1002.PV", "input": "x", "output": "y"},
    ]
    (chatts_dir / "chatts_converted_2_20260209.json").write_text(
        json.dumps(chatts_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    qwen_payload = [
        {"image": "/tmp/xxx/数据集P_2001.PV.jpg", "conversations": []},
    ]
    (qwen_dir / "qwen_converted_1_20260209.json").write_text(
        json.dumps(qwen_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with test_session_local() as db:
        db.add(
            db_mod.ReviewQueue(
                id="rq-allow-1",
                source_type="annotation",
                source_id="P_1001.PV",
                method="chatts",
                model=None,
                point_name="P_1001.PV",
                score=0.99,
                strategy="topk",
                status="approved",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-allow-2",
                source_type="annotation",
                source_id="P_2001.PV",
                method="qwen",
                model=None,
                point_name="P_2001.PV",
                score=0.88,
                strategy="topk",
                status="approved",
            )
        )
        db.commit()

    with test_session_local() as db:
        resp_approved = asyncio.run(
            assets_api.list_source_items(
                source_type="training",
                source_kind=None,
                model_family="all",
                method=None,
                sort_by="name_asc",
                approved_only=True,
                min_score=None,
                max_score=None,
                keyword=None,
                limit=50,
                db=db,
            )
        )

    approved_choices = (resp_approved.data or {}).get("choices") or []
    approved_values = [c.get("value") for c in approved_choices]
    assert "P_1001.PV" in approved_values
    assert "P_2001.PV" in approved_values
    assert "P_1002.PV" not in approved_values

    with test_session_local() as db:
        resp_all = asyncio.run(
            assets_api.list_source_items(
                source_type="training",
                source_kind=None,
                model_family="chatts",
                method=None,
                sort_by="name_asc",
                approved_only=False,
                min_score=None,
                max_score=None,
                keyword=None,
                limit=50,
                db=db,
            )
        )

    all_choices = (resp_all.data or {}).get("choices") or []
    all_values = [c.get("value") for c in all_choices]
    assert "P_1001.PV" in all_values
    assert "P_1002.PV" in all_values
