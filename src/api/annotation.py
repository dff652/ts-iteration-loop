"""
标注服务 API
集成 timeseries-annotator-v2 项目
"""
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Any, List, Optional
import json
import httpx
from pydantic import BaseModel

from src.adapters.annotation_export import AnnotationExportAdapter
from src.adapters.annotation_import import AnnotationImportAdapter
from src.core.logging_config import get_logger
from src.db.database import get_db
from src.models.schemas import AnnotationFile, ApiResponse
from configs.settings import settings

router = APIRouter()
logger = get_logger(__name__)


# 标注工具 API 基础 URL
ANNOTATOR_API = settings.ANNOTATOR_API_URL


class ImportInferenceRequest(BaseModel):
    inference_file: Optional[str] = None
    rows: Optional[List[dict[str, Any]]] = None
    token: Optional[str] = None


@router.get("/files", response_model=ApiResponse)
async def list_annotatable_files():
    """获取可标注文件列表（代理到标注工具）"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ANNOTATOR_API}/api/files")
            if resp.status_code == 200:
                return ApiResponse(
                    success=True,
                    data=resp.json(),
                    message="获取文件列表成功"
                )
            else:
                raise HTTPException(status_code=resp.status_code, detail="标注服务请求失败")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"标注服务不可用: {e}")


@router.get("/{filename}", response_model=ApiResponse)
async def get_annotations(filename: str, token: str = None):
    """获取文件标注（代理到标注工具）"""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{ANNOTATOR_API}/api/annotations/{filename}",
                headers=headers
            )
            if resp.status_code == 200:
                return ApiResponse(
                    success=True,
                    data=resp.json(),
                    message="获取标注成功"
                )
            else:
                raise HTTPException(status_code=resp.status_code, detail="获取标注失败")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"标注服务不可用: {e}")


@router.post("/{filename}", response_model=ApiResponse)
async def save_annotations(filename: str, annotations: AnnotationFile, token: str = None):
    """保存文件标注（代理到标注工具）"""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{ANNOTATOR_API}/api/annotations/{filename}",
                json=annotations.model_dump(),
                headers=headers
            )
            if resp.status_code == 200:
                return ApiResponse(
                    success=True,
                    message="保存标注成功"
                )
            else:
                raise HTTPException(status_code=resp.status_code, detail="保存标注失败")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"标注服务不可用: {e}")


@router.post("/import-inference", response_model=ApiResponse)
async def import_inference_results(
    inference_file: Optional[str] = None,
    request: Optional[ImportInferenceRequest] = Body(default=None),
):
    """
    导入推理结果作为预标注
    用于迭代循环中的反馈机制
    """
    try:
        req_file = inference_file or ((request.inference_file if request else None) or "")
        req_rows = request.rows if request else None
        req_token = request.token if request else None
        import_mode = "rows"
        deprecated_file_mode = False

        if isinstance(req_rows, list):
            result = await import_inference_rows_internal(req_rows, token=req_token)
        elif req_file:
            import_mode = "file_compat"
            deprecated_file_mode = True
            logger.warning("annotation/import-inference 使用文件路径兼容模式，建议改为 rows 直传")
            result = await import_inference_results_internal(req_file, token=req_token)
        else:
            raise HTTPException(status_code=400, detail="必须提供 inference_file 或 rows")

        message = f"成功导入 {result['count']} 个文件的预标注"
        if deprecated_file_mode:
            message += "（文件路径模式兼容，建议改用 rows）"

        return ApiResponse(
            success=True,
            data={
                "imported_count": result["count"],
                "mode": import_mode,
                "deprecated_file_mode": deprecated_file_mode,
            },
            message=message,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def import_inference_results_internal(inference_file: str, token: Optional[str] = None) -> dict:
    """内部导入逻辑，支持跨模块调用"""
    adapter = AnnotationImportAdapter(annotator_api_url=ANNOTATOR_API)
    return await adapter.import_from_file(inference_file, token=token)


async def import_inference_rows_internal(rows: List[dict[str, Any]], token: Optional[str] = None) -> dict:
    """内部导入逻辑：直接导入标注行（不依赖中间文件）。"""
    adapter = AnnotationImportAdapter(annotator_api_url=ANNOTATOR_API)
    return await adapter.import_rows(rows, token=token)


@router.get("/export/training-data", response_model=ApiResponse)
async def export_training_data(output_path: str = None, approved_only: bool = True):
    """
    导出标注结果为微调训练数据格式
    用于迭代循环中标注→微调的转换
    """
    try:
        import tempfile
        from pathlib import Path
        from src.db.database import SessionLocal, AnnotationRecord, ReviewQueue
        from src.utils.annotation_store import canonical_point_id, record_to_payload
        
        adapter = AnnotationExportAdapter()
        
        approved_set = None
        if approved_only:
            db = SessionLocal()
            try:
                rows = db.query(ReviewQueue.point_id, ReviewQueue.source_id).filter(
                    ReviewQueue.source_type == 'annotation',
                    ReviewQueue.status == 'approved'
                ).all()
                approved_set = {canonical_point_id(row[0], row[1]) for row in rows if canonical_point_id(row[0], row[1])}
            finally:
                db.close()

        # 构建临时标注目录（仅 approved）
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            selected = 0
            db = SessionLocal()
            try:
                rows = (
                    db.query(AnnotationRecord)
                    .filter(AnnotationRecord.user_id == settings.DEFAULT_USER)
                    .order_by(AnnotationRecord.updated_at.desc())
                    .all()
                )
                for row in rows:
                    normalized = canonical_point_id(getattr(row, "point_id", None), row.source_id, row.filename)
                    if approved_set is not None and normalized not in approved_set:
                        continue
                    data = record_to_payload(row)
                    out_name = f"{normalized}.json"
                    with open(tmp_path / out_name, 'w', encoding='utf-8') as wf:
                        json.dump(data, wf, ensure_ascii=False, indent=2)
                    selected += 1
            finally:
                db.close()

            if selected == 0:
                return ApiResponse(
                    success=False,
                    data={"output_path": output_path or ""},
                    message="未找到已审核通过的标注"
                )

            # 使用 Data-Processing 转换脚本
            output = output_path or "/tmp/training_data.jsonl"
            result = adapter.convert_annotations(
                input_dir=str(tmp_path),
                output_path=output,
                csv_src_dir=settings.DATA_DOWNSAMPLED_DIR,
                model_family="qwen"
            )
        
            return ApiResponse(
                success=result.get("success", False),
                data={"output_path": output},
                message="导出训练数据成功" if result.get("success") else "导出失败"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
