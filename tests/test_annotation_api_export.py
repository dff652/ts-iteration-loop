import asyncio
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import annotation as annotation_api
from src.db import database as db_mod


def test_export_training_data_uses_export_adapter_and_approved_filter(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(annotation_api.settings, "DEFAULT_USER", "tester")

    captured = {"selected_files": 0}

    class _FakeAnnotationExportAdapter:
        def convert_annotations(self, input_dir, output_path, **kwargs):
            captured["selected_files"] = len(list(Path(input_dir).glob("*.json")))
            return {"success": True, "output_path": output_path, "stdout": "", "stderr": ""}

    monkeypatch.setattr(annotation_api, "AnnotationExportAdapter", _FakeAnnotationExportAdapter)

    with test_session_local() as db:
        db.add(
            db_mod.AnnotationRecord(
                id="ann-1",
                user_id="tester",
                source_id="P_approved",
                filename="P_approved.csv",
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
                id="ann-2",
                user_id="tester",
                source_id="P_pending",
                filename="P_pending.csv",
                source_kind="human",
                method="chatts",
                is_human_edited=True,
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json='[{"id":"a2","segments":[{"start":2,"end":3}]}]',
                meta="{}",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-1",
                source_type="annotation",
                source_id="P_approved",
                point_id="P_approved",
                method="chatts",
                model=None,
                point_name="P_approved",
                score=0.91,
                strategy="topk",
                status="approved",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-2",
                source_type="annotation",
                source_id="P_pending",
                point_id="P_pending",
                method="chatts",
                model=None,
                point_name="P_pending",
                score=0.42,
                strategy="topk",
                status="pending",
            )
        )
        db.commit()

    resp = asyncio.run(
        annotation_api.export_training_data(
            output_path=str(tmp_path / "training.jsonl"),
            approved_only=True,
        )
    )

    assert resp.success is True
    assert captured["selected_files"] == 1
    assert (resp.data or {}).get("output_path", "").endswith("training.jsonl")

