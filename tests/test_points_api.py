import asyncio
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import points as points_api
from src.db import database as db_mod


def test_points_api_list_detail_timeline(tmp_path, monkeypatch):
    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(points_api.settings, "DEFAULT_USER", "tester")

    with test_session_local() as db:
        db.add(
            db_mod.AnnotationRecord(
                id="ann-1",
                user_id="tester",
                source_id="P_1001",
                filename="P_1001.csv",
                source_kind="human",
                annotation_count=1,
                segment_count=2,
                overall_attribute_json="{}",
                annotations_json="[]",
                meta="{}",
                updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        db.add(
            db_mod.InferenceResult(
                id="inf-1",
                task_id="task-1",
                method="chatts",
                point_name="P_1001",
                result_path="/tmp/P_1001.csv",
                score_avg=0.88,
                score_max=0.92,
                segment_count=2,
                meta="{}",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        db.add(
            db_mod.ReviewQueue(
                id="rq-1",
                source_type="annotation",
                source_id="P_1001",
                point_name="P_1001",
                score=0.88,
                strategy="topk",
                status="approved",
                updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        db.commit()

    with test_session_local() as db:
        list_resp = asyncio.run(points_api.list_points(keyword=None, limit=20, db=db))
    assert list_resp.success is True
    points = (list_resp.data or {}).get("points") or []
    assert len(points) == 1
    assert points[0]["point_id"] == "P_1001"
    assert points[0]["review_status"] == "approved"

    with test_session_local() as db:
        detail_resp = asyncio.run(points_api.get_point("P_1001", db=db))
    assert detail_resp.success is True
    detail = detail_resp.data or {}
    assert detail["point_id"] == "P_1001"
    assert detail["annotation_summary"]["count"] == 1
    assert detail["inference_summary"]["count"] == 1
    assert detail["review_summary"]["count"] == 1

    with test_session_local() as db:
        timeline_resp = asyncio.run(points_api.get_point_timeline("P_1001", limit=20, db=db))
    assert timeline_resp.success is True
    events = (timeline_resp.data or {}).get("events") or []
    assert len(events) == 3
    assert {e["type"] for e in events} == {"annotation", "inference", "review"}
