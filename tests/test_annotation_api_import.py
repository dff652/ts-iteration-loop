import asyncio

import pytest
from fastapi import HTTPException

from src.api import annotation as annotation_api


def test_import_inference_results_internal_uses_import_adapter(monkeypatch):
    captured = {"file": None}

    class _FakeImportAdapter:
        def __init__(self, annotator_api_url):
            self.annotator_api_url = annotator_api_url

        async def import_from_file(self, inference_file, token=None):
            captured["file"] = inference_file
            return {"success": True, "count": 2, "errors": []}

    monkeypatch.setattr(annotation_api, "AnnotationImportAdapter", _FakeImportAdapter)
    result = asyncio.run(annotation_api.import_inference_results_internal("/tmp/inference.json"))

    assert result["success"] is True
    assert result["count"] == 2
    assert captured["file"] == "/tmp/inference.json"


def test_import_inference_rows_internal_uses_import_adapter(monkeypatch):
    captured = {"rows": None, "token": None}

    class _FakeImportAdapter:
        def __init__(self, annotator_api_url):
            self.annotator_api_url = annotator_api_url

        async def import_rows(self, rows, token=None):
            captured["rows"] = rows
            captured["token"] = token
            return {"success": True, "count": 1, "errors": []}

    monkeypatch.setattr(annotation_api, "AnnotationImportAdapter", _FakeImportAdapter)
    rows = [{"filename": "p.csv", "annotations": [{"id": "a1"}]}]
    result = asyncio.run(annotation_api.import_inference_rows_internal(rows, token="abc"))
    assert result["success"] is True
    assert result["count"] == 1
    assert captured["rows"] == rows
    assert captured["token"] == "abc"


def test_import_inference_results_accepts_rows_body(monkeypatch):
    captured = {"rows": None}

    async def _fake_rows(rows, token=None):
        captured["rows"] = rows
        return {"success": True, "count": len(rows), "errors": []}

    monkeypatch.setattr(annotation_api, "import_inference_rows_internal", _fake_rows)
    req = annotation_api.ImportInferenceRequest(rows=[{"filename": "x.csv", "annotations": []}])
    resp = asyncio.run(annotation_api.import_inference_results(inference_file=None, request=req))
    assert resp.success is True
    assert (resp.data or {}).get("imported_count") == 1
    assert (resp.data or {}).get("mode") == "rows"
    assert (resp.data or {}).get("deprecated_file_mode") is False
    assert len(captured["rows"]) == 1


def test_import_inference_results_file_mode_marked_compat(monkeypatch):
    captured = {"file": None}

    async def _fake_file(inference_file, token=None):
        captured["file"] = inference_file
        return {"success": True, "count": 2, "errors": []}

    monkeypatch.setattr(annotation_api, "import_inference_results_internal", _fake_file)
    resp = asyncio.run(annotation_api.import_inference_results(inference_file="/tmp/inference.json", request=None))
    assert resp.success is True
    assert (resp.data or {}).get("imported_count") == 2
    assert (resp.data or {}).get("mode") == "file_compat"
    assert (resp.data or {}).get("deprecated_file_mode") is True
    assert "建议改用 rows" in (resp.message or "")
    assert captured["file"] == "/tmp/inference.json"


def test_import_inference_results_requires_file_or_rows():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(annotation_api.import_inference_results(inference_file=None, request=None))
    assert exc.value.status_code == 400
    assert "必须提供 inference_file 或 rows" in str(exc.value.detail)
