import asyncio
from datetime import datetime, timezone
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
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
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
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
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
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
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
    monkeypatch.setattr(assets_api.settings, "DEFAULT_USER", "tester")

    with test_session_local() as db:
        db.add(
            db_mod.AnnotationRecord(
                id="ann-train-1",
                user_id="tester",
                source_id="P_1001.PV",
                filename="P_1001.PV.csv",
                source_kind="human",
                method="chatts",
                is_human_edited=True,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"a1","segments":[{"start":1,"end":2}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.AnnotationRecord(
                id="ann-train-2",
                user_id="tester",
                source_id="P_1002.PV",
                filename="P_1002.PV.csv",
                source_kind="auto",
                method="chatts",
                is_human_edited=False,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"a2","segments":[{"start":3,"end":5}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.AnnotationRecord(
                id="ann-train-3",
                user_id="tester",
                source_id="P_2001.PV",
                filename="P_2001.PV.csv",
                source_kind="auto",
                method="qwen",
                is_human_edited=False,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"a3","segments":[{"start":7,"end":9}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.AnnotationRecord(
                id="ann-train-other-user",
                user_id="other-user",
                source_id="P_9999.PV",
                filename="P_9999.PV.csv",
                source_kind="human",
                method="chatts",
                is_human_edited=True,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"a4","segments":[{"start":1,"end":2}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-train-1",
                task_id="task-train-1",
                method="chatts",
                model="m",
                point_name="P_1001.PV",
                result_path="/tmp/P_1001.PV.csv",
                metrics_path="",
                segments_path="",
                score_avg=0.99,
                score_max=0.99,
                segment_count=1,
                meta="{}",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-train-2",
                task_id="task-train-2",
                method="chatts",
                model="m",
                point_name="P_1002.PV",
                result_path="/tmp/P_1002.PV.csv",
                metrics_path="",
                segments_path="",
                score_avg=0.55,
                score_max=0.61,
                segment_count=1,
                meta="{}",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-train-3",
                task_id="task-train-3",
                method="qwen",
                model="m",
                point_name="P_2001.PV",
                result_path="/tmp/P_2001.PV.csv",
                metrics_path="",
                segments_path="",
                score_avg=0.88,
                score_max=0.90,
                segment_count=1,
                meta="{}",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
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
        db.add(
            db_mod.ReviewQueue(
                id="rq-deny-1",
                source_type="annotation",
                source_id="P_1002.PV",
                method="chatts",
                model=None,
                point_name="P_1002.PV",
                score=0.55,
                strategy="topk",
                status="pending",
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
    assert "P_9999.PV" not in approved_values

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
    assert "P_2001.PV" not in all_values
    assert all(c.get("model_family") == "chatts" for c in all_choices)

    with test_session_local() as db:
        resp_qwen_by_method = asyncio.run(
            assets_api.list_source_items(
                source_type="training",
                source_kind=None,
                model_family="all",
                method="qwen",
                sort_by="name_asc",
                approved_only=False,
                min_score=None,
                max_score=None,
                keyword=None,
                limit=50,
                db=db,
            )
        )
    qwen_choices = (resp_qwen_by_method.data or {}).get("choices") or []
    assert [c.get("value") for c in qwen_choices] == ["P_2001.PV"]


def test_export_training_persists_export_history(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(assets_api.settings, "DEFAULT_USER", "tester")
    monkeypatch.setattr(assets_api.settings, "DATA_TRAINING_CHATTS_DIR", str(tmp_path / "chatts"))
    monkeypatch.setattr(assets_api.settings, "DATA_IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setattr(assets_api.settings, "DATA_DOWNSAMPLED_DIR", str(tmp_path / "csv"))

    class _FakeAnnotationExportAdapter:
        def convert_annotations(self, input_dir, output_path, image_dir, model_family, csv_src_dir):
            return {"success": True, "output_path": output_path}

    class _FakeTrainingAdapter:
        def __init__(self, model_family):
            self.model_family = model_family

        def get_dataset_list(self):
            return {"success": True}

    monkeypatch.setattr(assets_api, "AnnotationExportAdapter", _FakeAnnotationExportAdapter)
    monkeypatch.setattr(assets_api, "ChatTSTrainingAdapter", _FakeTrainingAdapter)

    with test_session_local() as db:
        db.add(
            db_mod.DatasetAsset(
                id="asset-train-1",
                name="asset_train",
                dataset_type="train",
                status="draft",
                point_count=2,
                meta='{"note":"seed"}',
            )
        )
        db.add(db_mod.DatasetItem(dataset_id="asset-train-1", point_id="P_1001.PV", point_name="P_1001.PV"))
        db.add(db_mod.DatasetItem(dataset_id="asset-train-1", point_id="P_1002.PV", point_name="P_1002.PV"))
        db.add(
            db_mod.AnnotationRecord(
                id="ann-export-1",
                user_id="tester",
                source_id="P_1001.PV",
                filename="P_1001.PV.csv",
                source_kind="human",
                method="chatts",
                is_human_edited=True,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"a1","segments":[{"start":1,"end":2}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.AnnotationRecord(
                id="ann-export-2",
                user_id="tester",
                source_id="P_1002.PV",
                filename="P_1002.PV.csv",
                source_kind="human",
                method="chatts",
                is_human_edited=True,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"a2","segments":[{"start":3,"end":4}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-export-1",
                source_type="annotation",
                source_id="P_1001.PV",
                point_id="P_1001.PV",
                method="chatts",
                model=None,
                point_name="P_1001.PV",
                score=0.90,
                strategy="topk",
                status="approved",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-export-2",
                source_type="annotation",
                source_id="P_1002.PV",
                point_id="P_1002.PV",
                method="chatts",
                model=None,
                point_name="P_1002.PV",
                score=0.70,
                strategy="topk",
                status="pending",
            )
        )
        db.commit()

    req = assets_api.ExportTrainingRequest(
        dataset_id="asset-train-1",
        model_family="chatts",
        output_name="bundle_v1",
        approved_only=True,
    )

    with test_session_local() as db:
        resp = asyncio.run(assets_api.export_training_dataset(req, db=db))
        refreshed = db.query(db_mod.DatasetAsset).filter(db_mod.DatasetAsset.id == "asset-train-1").first()
        meta = json.loads(refreshed.meta or "{}")

    assert resp.success is True
    assert (resp.data or {}).get("selected_count") == 1
    assert (resp.data or {}).get("export_id")
    assert (resp.data or {}).get("output_path", "").endswith("bundle_v1.jsonl")

    assert meta.get("note") == "seed"
    assert isinstance(meta.get("exports"), list)
    assert len(meta["exports"]) == 1
    last_export = meta.get("last_export") or {}
    assert last_export.get("selected_count") == 1
    assert last_export.get("approved_only") is True
    assert last_export.get("model_family") == "chatts"
    assert last_export.get("point_count") == 2
    assert last_export.get("output_path", "").endswith("bundle_v1.jsonl")


def test_assets_owner_org_scope_and_audit_fields(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(assets_api.settings, "DEFAULT_USER", "tester")
    monkeypatch.setattr(assets_api.settings, "DEFAULT_ORG", "org-default")

    with test_session_local() as db:
        db.add(
            db_mod.ReviewQueue(
                id="rq-scope-1",
                source_type="annotation",
                source_id="P_scope_1",
                point_id="P_scope_1",
                method="chatts",
                model=None,
                point_name="P_scope_1",
                score=0.88,
                strategy="topk",
                status="approved",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-scope-2",
                source_type="annotation",
                source_id="P_scope_2",
                point_id="P_scope_2",
                method="chatts",
                model=None,
                point_name="P_scope_2",
                score=0.86,
                strategy="topk",
                status="approved",
            )
        )
        db.commit()

    req_a = assets_api.AssetSaveRequest(
        name="asset_scope_a",
        dataset_type="train",
        items=["P_scope_1"],
        owner_id="owner_a",
        org_id="org_a",
    )
    req_b = assets_api.AssetSaveRequest(
        name="asset_scope_b",
        dataset_type="train",
        items=["P_scope_2"],
        owner_id="owner_b",
        org_id="org_b",
    )

    with test_session_local() as db:
        resp_a = asyncio.run(assets_api.save_asset(req_a, db=db))
        resp_b = asyncio.run(assets_api.save_asset(req_b, db=db))
        row_a = db.query(db_mod.DatasetAsset).filter(db_mod.DatasetAsset.id == resp_a.data["id"]).first()

    assert resp_a.success is True
    assert resp_b.success is True
    assert (resp_a.data or {}).get("owner_id") == "owner_a"
    assert (resp_a.data or {}).get("org_id") == "org_a"
    assert row_a.owner_id == "owner_a"
    assert row_a.org_id == "org_a"
    assert row_a.created_by == "tester"
    assert row_a.updated_by == "tester"

    with test_session_local() as db:
        filtered = asyncio.run(
            assets_api.list_assets(
                dataset_type="train",
                owner_id="owner_a",
                org_id="org_a",
                db=db,
            )
        )
    assets = ((filtered.data or {}).get("assets") or [])
    assert len(assets) == 1
    assert assets[0]["name"] == "asset_scope_a"
    assert assets[0]["owner_id"] == "owner_a"
    assert assets[0]["org_id"] == "org_a"

    with test_session_local() as db:
        detail_ok = asyncio.run(
            assets_api.get_asset(
                dataset_id=resp_a.data["id"],
                owner_id="owner_a",
                org_id="org_a",
                db=db,
            )
        )
        assert detail_ok.success is True

    with test_session_local() as db:
        with pytest.raises(Exception) as exc:
            asyncio.run(
                assets_api.get_asset(
                    dataset_id=resp_a.data["id"],
                    owner_id="owner_b",
                    org_id="org_b",
                    db=db,
                )
            )
        assert "数据集不存在" in str(exc.value)
