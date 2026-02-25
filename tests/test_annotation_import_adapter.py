import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from src.adapters.annotation_import import AnnotationImportAdapter


def test_import_from_file_returns_missing_file_error(tmp_path):
    adapter = AnnotationImportAdapter(annotator_api_url="http://annotator")
    result = asyncio.run(adapter.import_from_file(str(tmp_path / "missing.json")))
    assert result["success"] is False
    assert result["count"] == 0
    assert "文件不存在" in (result.get("error") or "")


def test_import_rows_handles_wrapped_results_and_partial_failures(tmp_path, monkeypatch):
    payload = {
        "results": [
            {
                "file": "raw.csv",
                "result": [
                    {"filename": "P_1001.csv", "annotations": [{"label": "x", "segments": [{"start": 1, "end": 2}]}]},
                    {"filename": "P_1002.csv", "annotations": [{"label": "y", "segments": [{"start": 3, "end": 4}]}]},
                ],
            }
        ]
    }
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    calls = []

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            calls.append({"url": url, "json": json, "headers": headers})
            if url.endswith("P_1002.csv"):
                return SimpleNamespace(status_code=500, text="boom")
            return SimpleNamespace(status_code=200, text="")

    monkeypatch.setattr("src.adapters.annotation_import.httpx.AsyncClient", _FakeClient)
    adapter = AnnotationImportAdapter(annotator_api_url="http://annotator")
    result = asyncio.run(adapter.import_from_file(str(input_file), token="tkn"))

    assert result["success"] is False
    assert result["count"] == 1
    assert len(result["errors"]) == 1
    assert result["errors"][0]["file"] == "P_1002.csv"
    assert len(calls) == 2
    assert all(c["headers"].get("Authorization") == "Bearer tkn" for c in calls)
    assert calls[0]["url"] == "http://annotator/api/annotations/P_1001.csv"


def test_extract_rows_accepts_direct_list():
    adapter = AnnotationImportAdapter(annotator_api_url="http://annotator")
    rows = adapter._extract_annotation_rows(
        [{"filename": "P_3001.csv", "annotations": [{"label": "z", "segments": []}]}]
    )
    assert rows == [{"filename": "P_3001.csv", "annotations": [{"label": "z", "segments": []}]}]
