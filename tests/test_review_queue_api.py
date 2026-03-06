import asyncio
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import review as review_api
from src.db import database as db_mod


def test_review_queue_list_and_stats(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(review_api.settings, "DEFAULT_USER", "tester")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with test_session_local() as db:
        db.add(
            db_mod.AnnotationRecord(
                id="ann-1",
                user_id="tester",
                source_id="P_1001",
                filename="P_1001.csv",
                source_kind="human",
                annotation_count=2,
                segment_count=3,
                overall_attribute_json="{}",
                annotations_json="[]",
                meta="{}",
                updated_at=now,
            )
        )
        db.add(
            db_mod.AnnotationRecord(
                id="ann-2",
                user_id="tester",
                source_id="P_1002",
                filename="P_1002.csv",
                source_kind="auto",
                annotation_count=1,
                segment_count=1,
                overall_attribute_json="{}",
                annotations_json="[]",
                meta="{}",
                updated_at=now,
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-1",
                task_id="task-1",
                point_name="P_1001",
                method="chatts",
                model="/tmp/model",
                result_path="/tmp/P_1001.csv",
                score_avg=0.88,
                score_max=0.90,
                segment_count=2,
                meta="{}",
                created_at=now,
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-1",
                source_type="annotation",
                source_id="P_1001",
                point_id="P_1001",
                point_name="P_1001",
                status="approved",
                reviewer="alice",
                updated_at=now,
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-2",
                source_type="annotation",
                source_id="P_1002",
                point_id="P_1002",
                point_name="P_1002",
                status="rejected",
                reviewer="bob",
                updated_at=now,
            )
        )
        db.commit()

    with test_session_local() as db:
        resp = asyncio.run(
            review_api.list_review_queue(
                status=None,
                method=None,
                annotation_kind="all",
                keyword=None,
                limit=20,
                offset=0,
                db=db,
            )
        )
    assert resp.success is True
    data = resp.data or {}
    assert int(data.get("total", 0)) == 2
    stats = data.get("stats") or {}
    assert int(stats.get("approved", 0)) == 1
    assert int(stats.get("needs_fix", 0)) == 1

    with test_session_local() as db:
        filtered = asyncio.run(
            review_api.list_review_queue(
                status="approved",
                method=None,
                annotation_kind="all",
                keyword=None,
                limit=20,
                offset=0,
                db=db,
            )
        )
    assert filtered.success is True
    items = (filtered.data or {}).get("items") or []
    assert len(items) == 1
    assert items[0]["point_id"] == "P_1001"


def test_review_queue_batch_update_upsert_and_modify(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with test_session_local() as db:
        db.add(
            db_mod.ReviewQueue(
                id="rq-existing",
                source_type="annotation",
                source_id="P_2001",
                point_id="P_2001",
                point_name="P_2001",
                status="pending",
                reviewer="init",
                updated_at=now,
            )
        )
        db.commit()

    payload = review_api.ReviewBatchUpdateRequest(
        point_ids=["P_2001", "P_2002.csv"],
        status="approved",
        reviewer="tester",
    )

    with test_session_local() as db:
        resp = asyncio.run(review_api.batch_update_review_status(payload, db=db))
    assert resp.success is True
    data = resp.data or {}
    assert int(data.get("updated", 0)) == 2
    assert data.get("status") == "approved"

    with test_session_local() as db:
        rows = (
            db.query(db_mod.ReviewQueue)
            .filter(db_mod.ReviewQueue.source_type == "annotation")
            .order_by(db_mod.ReviewQueue.point_id.asc())
            .all()
        )
    by_point = {str(row.point_id): row for row in rows}
    assert "P_2001" in by_point
    assert "P_2002" in by_point
    assert by_point["P_2001"].status == "approved"
    assert by_point["P_2002"].status == "approved"
    assert by_point["P_2001"].reviewer == "tester"
