import importlib
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import database as db_mod


@pytest.fixture()
def annotator_env(tmp_path, monkeypatch):
    app_mod = importlib.import_module("services.annotator.backend.app")
    auth_mod = importlib.import_module("auth")

    monkeypatch.setenv("ANNOTATOR_AUTH_BYPASS", "true")
    monkeypatch.setenv("ANNOTATOR_AUTH_BYPASS_USER", "tester")

    user_data_dir = tmp_path / "user_data"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    ann_root = tmp_path / "annotations"
    (ann_root / "tester").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        auth_mod,
        "load_users",
        lambda: {
            "tester": {
                "name": "Tester",
                "password_hash": "sha256:dummy",
                "data_path": str(user_data_dir),
            }
        },
    )
    monkeypatch.setattr(auth_mod, "save_users", lambda _users: None)

    monkeypatch.setattr(app_mod, "ANNOTATIONS_DIR", str(ann_root))
    monkeypatch.setattr(app_mod, "_CSV_PATH_CACHE", {})

    registry_db = tmp_path / "file_registry.db"
    monkeypatch.setattr(app_mod, "REGISTRY_DB", str(registry_db))
    app_mod.init_db(str(registry_db))

    db_path = tmp_path / "iteration_loop.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.Base.metadata.create_all(bind=engine)

    client = app_mod.app.test_client()
    return {
        "client": client,
        "session_local": test_session_local,
        "user_data_dir": user_data_dir,
    }


def _insert_inference_rows(session_local, rows):
    with session_local() as db:
        for row in rows:
            db.add(
                db_mod.InferenceResult(
                    id=row["id"],
                    task_id=row.get("task_id"),
                    method=row.get("method", "chatts"),
                    model=row.get("model", "/tmp/model"),
                    point_name=row.get("point_name", row["id"]),
                    result_path=row["result_path"],
                    metrics_path=row.get("metrics_path", ""),
                    segments_path=row.get("segments_path", ""),
                    score_avg=row.get("score_avg", 0.0),
                    score_max=row.get("score_max", row.get("score_avg", 0.0)),
                    segment_count=row.get("segment_count", 0),
                    meta=row.get("meta", "{}"),
                    created_at=row.get("created_at", datetime.utcnow()),
                )
            )
        db.commit()


def test_review_queue_e2e_from_inference_sample(annotator_env):
    client = annotator_env["client"]
    session_local = annotator_env["session_local"]
    user_data_dir: Path = annotator_env["user_data_dir"]

    (user_data_dir / "high.csv").write_text("v\n1\n2\n", encoding="utf-8")
    (user_data_dir / "low.csv").write_text("v\n3\n4\n", encoding="utf-8")

    _insert_inference_rows(
        session_local,
        [
            {
                "id": "inf-high",
                "method": "chatts",
                "point_name": "high_point",
                "result_path": str(user_data_dir / "high.csv"),
                "score_avg": 0.91,
                "score_max": 0.95,
                "segment_count": 2,
            },
            {
                "id": "inf-low",
                "method": "chatts",
                "point_name": "low_point",
                "result_path": str(user_data_dir / "low.csv"),
                "score_avg": 0.21,
                "score_max": 0.30,
                "segment_count": 1,
            },
        ],
    )

    sample_resp = client.post(
        "/api/review/sample",
        json={
            "source_type": "inference",
            "strategy": "topk",
            "limit": 1,
            "score_by": "score_avg",
            "method": "chatts",
        },
    )
    sample_payload = sample_resp.get_json()
    assert sample_resp.status_code == 200
    assert sample_payload["success"] is True
    assert sample_payload["created"] == 1

    queue_resp = client.get("/api/review/queue?source_type=inference")
    queue_payload = queue_resp.get_json()
    assert queue_resp.status_code == 200
    assert queue_payload["success"] is True
    assert len(queue_payload["items"]) == 1
    item = queue_payload["items"][0]
    assert item["source_id"] == "inf-high"
    assert item["status"] == "pending"

    batch_resp = client.patch(
        "/api/review/queue/batch",
        json={"ids": [item["id"]], "status": "approved"},
    )
    batch_payload = batch_resp.get_json()
    assert batch_resp.status_code == 200
    assert batch_payload["success"] is True
    assert batch_payload["updated"] == 1

    stats_resp = client.get("/api/review/stats?source_type=inference")
    stats_payload = stats_resp.get_json()
    assert stats_resp.status_code == 200
    assert stats_payload["success"] is True
    assert stats_payload["stats"].get("approved", 0) == 1


def test_files_endpoint_filters_candidates_by_score(annotator_env):
    client = annotator_env["client"]
    session_local = annotator_env["session_local"]
    user_data_dir: Path = annotator_env["user_data_dir"]

    (user_data_dir / "keep.csv").write_text("v\n10\n11\n", encoding="utf-8")
    (user_data_dir / "drop.csv").write_text("v\n12\n13\n", encoding="utf-8")

    _insert_inference_rows(
        session_local,
        [
            {
                "id": "inf-keep",
                "method": "chatts",
                "point_name": "keep_point",
                "result_path": str(user_data_dir / "keep.csv"),
                "score_avg": 0.88,
                "score_max": 0.91,
            },
            {
                "id": "inf-drop",
                "method": "chatts",
                "point_name": "drop_point",
                "result_path": str(user_data_dir / "drop.csv"),
                "score_avg": 0.22,
                "score_max": 0.30,
            },
        ],
    )

    resp = client.get("/api/files?min_score=0.5&score_by=score_avg&strategy=topk&limit=10")
    payload = resp.get_json()
    assert resp.status_code == 200
    assert payload["success"] is True
    names = [f["name"] for f in payload["files"]]
    assert "keep.csv" in names
    assert "drop.csv" not in names

    keep = next(f for f in payload["files"] if f["name"] == "keep.csv")
    assert abs(float(keep["score_avg"]) - 0.88) < 1e-6


def test_annotations_db_first_save_get_and_file_marker(annotator_env):
    client = annotator_env["client"]
    session_local = annotator_env["session_local"]
    user_data_dir: Path = annotator_env["user_data_dir"]

    (user_data_dir / "anno.csv").write_text("v\n1\n2\n", encoding="utf-8")

    save_resp = client.post(
        "/api/annotations/anno.csv",
        json={
            "annotations": [
                {
                    "id": "infer_1",
                    "source": "inference",
                    "label": {"id": "chatts_detected", "text": "ChatTS"},
                    "segments": [{"start": 1, "end": 3, "count": 3}],
                }
            ],
            "overall_attributes": {"device": "A"},
        },
    )
    payload = save_resp.get_json()
    assert save_resp.status_code == 200
    assert payload["success"] is True
    assert payload["source_kind"] == "auto"

    get_resp = client.get("/api/annotations/anno.csv")
    get_payload = get_resp.get_json()
    assert get_resp.status_code == 200
    assert get_payload["success"] is True
    assert len(get_payload["annotations"]) == 1

    files_resp = client.get("/api/files")
    files_payload = files_resp.get_json()
    assert files_resp.status_code == 200
    assert files_payload["success"] is True
    anno_row = next(item for item in files_payload["files"] if item["name"] == "anno.csv")
    assert anno_row["has_annotations"] is True
    assert int(anno_row["annotation_count"]) == 1
    assert anno_row["annotation_source_kind"] == "auto"

    with session_local() as db:
        stored = db.query(db_mod.AnnotationRecord).filter(db_mod.AnnotationRecord.source_id == "anno").first()
        assert stored is not None
        assert stored.source_kind == "auto"
        assert int(stored.segment_count or 0) == 1
