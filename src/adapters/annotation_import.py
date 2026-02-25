"""
标注导入适配器
负责将推理产物导入标注服务。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from configs.settings import settings


class AnnotationImportAdapter:
    """封装推理结果导入标注服务的流程。"""

    def __init__(self, annotator_api_url: Optional[str] = None):
        self.annotator_api_url = (annotator_api_url or settings.ANNOTATOR_API_URL).rstrip("/")

    async def import_from_file(self, inference_file: str, token: Optional[str] = None) -> Dict:
        file_path = Path(inference_file)
        if not file_path.exists():
            return {"success": False, "error": f"文件不存在: {inference_file}", "count": 0, "errors": []}

        with file_path.open("r", encoding="utf-8") as f:
            raw_payload = json.load(f)

        rows = self._extract_annotation_rows(raw_payload)
        return await self.import_rows(rows, token=token)

    async def import_rows(self, rows: List[Dict], token: Optional[str] = None) -> Dict:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        imported_count = 0
        errors: List[Dict] = []

        async with httpx.AsyncClient() as client:
            for item in rows:
                filename = str(item.get("filename") or "").strip()
                annotations = item.get("annotations")
                if not filename or not isinstance(annotations, list):
                    continue

                try:
                    resp = await client.post(
                        f"{self.annotator_api_url}/api/annotations/{filename}",
                        json={"annotations": annotations},
                        headers=headers,
                    )
                except Exception as e:
                    errors.append({"file": filename, "error": str(e)})
                    continue

                if resp.status_code == 200:
                    imported_count += 1
                else:
                    errors.append(
                        {
                            "file": filename,
                            "status_code": resp.status_code,
                            "error": (resp.text or "")[:200],
                        }
                    )

        return {
            "success": len(errors) == 0,
            "count": imported_count,
            "errors": errors,
        }

    def _extract_annotation_rows(self, payload: Any) -> List[Dict]:
        if payload is None:
            return []

        if isinstance(payload, list):
            rows: List[Dict] = []
            for item in payload:
                rows.extend(self._normalize_item(item))
            return rows

        if isinstance(payload, dict):
            results = payload.get("results")
            if isinstance(results, list):
                rows: List[Dict] = []
                for item in results:
                    rows.extend(self._normalize_item(item))
                return rows
            return self._normalize_item(payload)

        return []

    def _normalize_item(self, item: Any) -> List[Dict]:
        if not isinstance(item, dict):
            return []

        filename = item.get("filename") or item.get("file")
        annotations = item.get("annotations")
        if filename and isinstance(annotations, list):
            return [{"filename": filename, "annotations": annotations}]

        wrapped = item.get("result")
        if wrapped is None:
            return []
        return self._normalize_wrapped(wrapped)

    def _normalize_wrapped(self, wrapped: Any) -> List[Dict]:
        if isinstance(wrapped, str):
            try:
                wrapped = json.loads(wrapped)
            except Exception:
                return []

        if isinstance(wrapped, dict):
            filename = wrapped.get("filename") or wrapped.get("file")
            annotations = wrapped.get("annotations")
            if filename and isinstance(annotations, list):
                return [{"filename": filename, "annotations": annotations}]
            return []

        if isinstance(wrapped, list):
            rows: List[Dict] = []
            for entry in wrapped:
                if not isinstance(entry, dict):
                    continue
                filename = entry.get("filename") or entry.get("file")
                annotations = entry.get("annotations")
                if filename and isinstance(annotations, list):
                    rows.append({"filename": filename, "annotations": annotations})
            return rows

        return []

