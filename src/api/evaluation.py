"""
模型评估 API 路由
"""
import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from src.db.database import get_db, ModelEval
from src.models.schemas import ApiResponse, ModelEvalInfo

router = APIRouter()

@router.get("/", response_model=ApiResponse)
async def list_evaluations(model_path: Optional[str] = None, db=Depends(get_db)):
    """获取评估结果列表"""
    try:
        query = db.query(ModelEval)
        if model_path:
            query = query.filter(ModelEval.model_path == model_path)
            
        evals = query.order_by(ModelEval.created_at.desc()).all()
        
        results = []
        for e in evals:
            results.append(ModelEvalInfo(
                id=e.id,
                task_id=e.task_id,
                model_family=e.model_family,
                model_path=e.model_path,
                dataset_id=e.dataset_id,
                dataset_name=e.dataset_name,
                metrics=json.loads(e.metrics) if e.metrics else {},
                created_at=e.created_at
            ).model_dump())
            
        return ApiResponse(success=True, data={"evaluations": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
