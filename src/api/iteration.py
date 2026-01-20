"""
迭代版本管理 API
管理迭代循环的版本追踪
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import uuid
import json

from src.db.database import get_db, IterationVersion
from src.models.schemas import ApiResponse

router = APIRouter()


# ==================== Pydantic 模型 ====================

from pydantic import BaseModel


class IterationCreate(BaseModel):
    """创建迭代版本请求"""
    version: str
    description: Optional[str] = None
    dataset_path: Optional[str] = None
    model_path: Optional[str] = None


class IterationUpdate(BaseModel):
    """更新迭代版本请求"""
    description: Optional[str] = None
    annotation_count: Optional[int] = None
    model_path: Optional[str] = None
    metrics: Optional[dict] = None


class IterationResponse(BaseModel):
    """迭代版本响应"""
    id: str
    version: str
    description: Optional[str]
    dataset_path: Optional[str]
    annotation_count: int
    model_path: Optional[str]
    metrics: Optional[dict]
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== API 端点 ====================

@router.post("/", response_model=ApiResponse)
async def create_iteration(
    request: IterationCreate,
    db: Session = Depends(get_db)
):
    """创建新的迭代版本"""
    iteration_id = str(uuid.uuid4())
    
    iteration = IterationVersion(
        id=iteration_id,
        version=request.version,
        description=request.description,
        dataset_path=request.dataset_path,
        model_path=request.model_path,
        annotation_count=0,
        created_at=datetime.utcnow()
    )
    
    db.add(iteration)
    db.commit()
    db.refresh(iteration)
    
    return ApiResponse(
        success=True,
        data={"id": iteration_id, "version": request.version},
        message=f"迭代版本 {request.version} 创建成功"
    )


@router.get("/", response_model=ApiResponse)
async def list_iterations(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """获取迭代版本列表"""
    iterations = db.query(IterationVersion)\
        .order_by(IterationVersion.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    result = []
    for it in iterations:
        result.append({
            "id": it.id,
            "version": it.version,
            "description": it.description,
            "annotation_count": it.annotation_count or 0,
            "model_path": it.model_path,
            "created_at": it.created_at.isoformat() if it.created_at else None
        })
    
    return ApiResponse(
        success=True,
        data={"iterations": result, "total": len(result)},
        message=f"找到 {len(result)} 个迭代版本"
    )


@router.get("/{iteration_id}", response_model=ApiResponse)
async def get_iteration(
    iteration_id: str,
    db: Session = Depends(get_db)
):
    """获取迭代版本详情"""
    iteration = db.query(IterationVersion)\
        .filter(IterationVersion.id == iteration_id)\
        .first()
    
    if not iteration:
        raise HTTPException(status_code=404, detail="迭代版本不存在")
    
    metrics = None
    if iteration.metrics:
        try:
            metrics = json.loads(iteration.metrics)
        except:
            metrics = None
    
    return ApiResponse(
        success=True,
        data={
            "id": iteration.id,
            "version": iteration.version,
            "description": iteration.description,
            "dataset_path": iteration.dataset_path,
            "annotation_count": iteration.annotation_count or 0,
            "model_path": iteration.model_path,
            "metrics": metrics,
            "created_at": iteration.created_at.isoformat() if iteration.created_at else None
        },
        message="获取详情成功"
    )


@router.put("/{iteration_id}", response_model=ApiResponse)
async def update_iteration(
    iteration_id: str,
    request: IterationUpdate,
    db: Session = Depends(get_db)
):
    """更新迭代版本"""
    iteration = db.query(IterationVersion)\
        .filter(IterationVersion.id == iteration_id)\
        .first()
    
    if not iteration:
        raise HTTPException(status_code=404, detail="迭代版本不存在")
    
    if request.description is not None:
        iteration.description = request.description
    if request.annotation_count is not None:
        iteration.annotation_count = request.annotation_count
    if request.model_path is not None:
        iteration.model_path = request.model_path
    if request.metrics is not None:
        iteration.metrics = json.dumps(request.metrics)
    
    db.commit()
    
    return ApiResponse(
        success=True,
        data={"id": iteration_id},
        message="更新成功"
    )


@router.delete("/{iteration_id}", response_model=ApiResponse)
async def delete_iteration(
    iteration_id: str,
    db: Session = Depends(get_db)
):
    """删除迭代版本"""
    iteration = db.query(IterationVersion)\
        .filter(IterationVersion.id == iteration_id)\
        .first()
    
    if not iteration:
        raise HTTPException(status_code=404, detail="迭代版本不存在")
    
    version = iteration.version
    db.delete(iteration)
    db.commit()
    
    return ApiResponse(
        success=True,
        message=f"迭代版本 {version} 已删除"
    )
