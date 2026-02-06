import json
from pathlib import Path

from flask import Flask

from src.adapters.check_outlier import CheckOutlierAdapter
from services.annotator.backend.auth import login_required


def test_convert_to_annotation_format_from_legacy_payload():
    adapter = CheckOutlierAdapter()
    payload = {
        "success": True,
        "results": [
            {
                "file": "test_data_001.csv",
                "success": True,
                "result": {
                    "detected_anomalies": [
                        {"type": "point", "interval": [10, 20], "reason": "spike"},
                        {"type": "trend", "interval": [30, 40], "reason": "drift"},
                    ]
                },
            }
        ],
    }

    output = adapter.convert_to_annotation_format(payload)
    output_path = Path(output)
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["filename"] == "test_data_001.csv"
    assert data[0]["source"] == "inference"
    assert len(data[0]["annotations"]) == 2


def test_login_required_default_requires_auth(monkeypatch):
    monkeypatch.delenv("ANNOTATOR_AUTH_BYPASS", raising=False)

    app = Flask(__name__)

    @login_required
    def protected(current_user=None):
        return {"ok": True, "user": current_user}

    with app.test_request_context("/api/test", headers={}):
        resp = protected()
        assert isinstance(resp, tuple)
        assert resp[1] == 401


def test_login_required_bypass_with_env(monkeypatch):
    monkeypatch.setenv("ANNOTATOR_AUTH_BYPASS", "true")
    monkeypatch.setenv("ANNOTATOR_AUTH_BYPASS_USER", "dev_user")

    app = Flask(__name__)

    @login_required
    def protected(current_user=None):
        return {"ok": True, "user": current_user}

    with app.test_request_context("/api/test", headers={}):
        resp = protected()
        assert resp["ok"] is True
        assert resp["user"] == "dev_user"
