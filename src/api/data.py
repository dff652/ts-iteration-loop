"""
数据服务 API
封装 Data-Processing 项目功能
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime

from src.db.database import get_db, Task
from src.models.schemas import (
    DatasetInfo, AcquireTaskRequest, 
    TaskResponse, TaskStatus, ApiResponse
)
from src.adapters.data_processing import DataProcessingAdapter

router = APIRouter()
adapter = DataProcessingAdapter()


@router.get("/datasets", response_model=ApiResponse)
async def list_datasets():
    """获取数据集列表"""
    try:
        datasets = adapter.list_datasets()
        return ApiResponse(
            success=True,
            data={"datasets": datasets},
            message=f"找到 {len(datasets)} 个数据集"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acquire", response_model=TaskResponse)
async def start_acquire_task(
    request: AcquireTaskRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """启动数据采集任务"""
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    task = Task(
        id=task_id,
        type="acquire",
        status=TaskStatus.PENDING,
        config=request.model_dump_json()
    )
    db.add(task)
    db.commit()
    
    # 后台执行采集
    background_tasks.add_task(
        adapter.run_acquire_task,
        task_id=task_id,
        source=request.source,
        host=request.host,
        port=request.port,
        user=request.user,
        password=request.password,
        point_name=request.point_name,
        target_points=request.target_points,
        start_time=request.start_time,
        end_time=request.end_time
    )
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="数据采集任务已提交"
    )


@router.get("/status/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """获取任务状态"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return TaskResponse(
        task_id=task.id,
        status=TaskStatus(task.status),
        message=task.error or "运行中" if task.status == "running" else "完成"
    )


@router.get("/preview/{filename}")
async def preview_data(filename: str, limit: int = 100):
    """预览数据文件"""
    try:
        data = adapter.preview_csv(filename, limit=limit)
        return ApiResponse(
            success=True,
            data={"filename": filename, "preview": data},
            message=f"显示前 {limit} 条数据"
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
