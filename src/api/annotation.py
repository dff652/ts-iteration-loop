"""
标注服务 API
集成 timeseries-annotator-v2 项目
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json
import httpx

from src.db.database import get_db
from src.models.schemas import AnnotationFile, ApiResponse
from configs.settings import settings

router = APIRouter()


# 标注工具 API 基础 URL
ANNOTATOR_API = settings.ANNOTATOR_API_URL


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
async def import_inference_results(inference_file: str):
    """
    导入推理结果作为预标注
    用于迭代循环中的反馈机制
    """
    try:
        result = await import_inference_results_internal(inference_file)
        return ApiResponse(
            success=True,
            data={"imported_count": result["count"]},
            message=f"成功导入 {result['count']} 个文件的预标注"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def import_inference_results_internal(inference_file: str) -> dict:
    """内部导入逻辑，支持跨模块调用"""
    import json
    from pathlib import Path
    
    file_path = Path(inference_file)
    if not file_path.exists():
        return {"success": False, "error": f"文件不存在: {inference_file}", "count": 0}
    
    with open(file_path, "r", encoding="utf-8") as f:
        inference_data = json.load(f)
    
    # 转换为标注工具可接受的格式并导入
    imported_count = 0
    for item in inference_data:
        filename = item.get("filename")
        annotations = item.get("annotations", [])
        
        if filename and annotations:
            # 调用标注工具 API 保存
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{ANNOTATOR_API}/api/annotations/{filename}",
                    json={"annotations": annotations}
                )
                if resp.status_code == 200:
                    imported_count += 1
    
    return {"success": True, "count": imported_count}


@router.get("/export/training-data", response_model=ApiResponse)
async def export_training_data(output_path: str = None, approved_only: bool = True):
    """
    导出标注结果为微调训练数据格式
    用于迭代循环中标注→微调的转换
    """
    try:
        import tempfile
        from pathlib import Path
        from src.adapters.data_processing import DataProcessingAdapter
        from src.db.database import SessionLocal, ReviewQueue
        
        adapter = DataProcessingAdapter()
        
        ann_dir = Path(settings.ANNOTATIONS_ROOT) / settings.DEFAULT_USER
        if not ann_dir.exists():
            raise HTTPException(status_code=404, detail="标注目录不存在")

        approved_set = None
        if approved_only:
            db = SessionLocal()
            try:
                rows = db.query(ReviewQueue.source_id).filter(
                    ReviewQueue.source_type == 'annotation',
                    ReviewQueue.status == 'approved'
                ).all()
                approved_set = {str(row[0]).replace('.csv', '').replace('.json', '') for row in rows}
            finally:
                db.close()

        # 构建临时标注目录（仅 approved）
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            selected = 0
            for json_file in ann_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue

                filename = data.get('filename') or json_file.stem
                normalized = str(filename).replace('.csv', '').replace('.json', '')
                if approved_set is not None and normalized not in approved_set:
                    continue

                out_name = json_file.name if json_file.name.endswith('.json') else f"{json_file.stem}.json"
                with open(tmp_path / out_name, 'w', encoding='utf-8') as wf:
                    json.dump(data, wf, ensure_ascii=False, indent=2)
                selected += 1

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
