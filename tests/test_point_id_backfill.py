from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import database as db_mod
from src.utils.point_id_backfill import backfill_point_ids


def test_backfill_point_ids(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    with test_session_local() as db:
        db.add(
            db_mod.AnnotationRecord(
                id="ann-1",
                user_id="tester",
                point_id=None,
                source_id="P_1001",
                filename="P_1001.csv",
                source_kind="human",
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json="[]",
                meta="{}",
            )
        )
        db.add(
            db_mod.AnnotationSegment(
                annotation_id="ann-1",
                user_id="tester",
                point_id=None,
                source_id="P_1001",
                ann_index=0,
                seg_index=0,
                start=1,
                end=3,
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-1",
                task_id="task-1",
                point_id=None,
                method="chatts",
                point_name="P_1001.csv",
                result_path="/tmp/P_1001.csv",
                score_avg=0.8,
                score_max=0.9,
                segment_count=1,
                meta="{}",
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-1",
                source_type="annotation",
                source_id="P_1001.csv",
                point_id=None,
                point_name=None,
                score=0.8,
                strategy="topk",
                status="pending",
            )
        )
        db.add(
            db_mod.DatasetItem(
                dataset_id="d-1",
                point_id=None,
                point_name="P_1001.csv",
            )
        )
        db.commit()

    with test_session_local() as db:
        stats = backfill_point_ids(db)
        assert stats["total"] >= 5

        ann = db.query(db_mod.AnnotationRecord).filter(db_mod.AnnotationRecord.id == "ann-1").first()
        seg = db.query(db_mod.AnnotationSegment).filter(db_mod.AnnotationSegment.annotation_id == "ann-1").first()
        inf = db.query(db_mod.InferenceResult).filter(db_mod.InferenceResult.id == "inf-1").first()
        rev = db.query(db_mod.ReviewQueue).filter(db_mod.ReviewQueue.id == "rq-1").first()
        item = db.query(db_mod.DatasetItem).filter(db_mod.DatasetItem.dataset_id == "d-1").first()

        assert ann.point_id == "P_1001"
        assert seg.point_id == "P_1001"
        assert inf.point_id == "P_1001"
        assert rev.point_id == "P_1001"
        assert item.point_id == "P_1001"
