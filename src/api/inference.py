"""
推理服务 API
封装 check_outlier 项目功能
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import uuid

from src.db.database import get_db, Task
from src.models.schemas import (
    InferenceTaskRequest, InferenceResult,
    TaskResponse, TaskStatus, ApiResponse
)
from src.adapters.check_outlier import CheckOutlierAdapter

router = APIRouter()
adapter = CheckOutlierAdapter()


@router.get("/algorithms", response_model=ApiResponse)
async def list_algorithms():
    """获取可用算法列表"""
    algorithms = [
        {"id": "chatts", "name": "ChatTS", "description": "ChatTS 大模型检测"},
        {"id": "adtk_hbos", "name": "ADTK-HBOS", "description": "传统统计方法"},
        {"id": "ensemble", "name": "Ensemble", "description": "集成方法"}
    ]
    return ApiResponse(
        success=True,
        data={"algorithms": algorithms},
        message=f"支持 {len(algorithms)} 种算法"
    )


@router.post("/batch", response_model=TaskResponse)
async def start_batch_inference(
    request: InferenceTaskRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """启动批量推理任务"""
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    task = Task(
        id=task_id,
        type="inference",
        status=TaskStatus.PENDING,
        config=request.model_dump_json()
    )
    db.add(task)
    db.commit()
    
    # 后台执行推理 (使用 wrapper 更新数据库状态)
    background_tasks.add_task(
        run_inference_task,
        task_id=task_id,
        model=request.model,
        algorithm=request.algorithm,
        input_files=request.input_files
    )
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"推理任务已提交，处理 {len(request.input_files)} 个文件"
    )

def run_inference_task(task_id: str, model: str, algorithm: str, input_files: List[str]):
    """后台任务包装器: 执行推理并更新数据库状态"""
    db = next(get_db())
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return
            
        # Use schema-defined status to avoid enum mismatch.
        task.status = TaskStatus.RUNNING
        db.commit()
        
        # 执行推理
        result = adapter.run_batch_inference(
            task_id=task_id,
            model=model,
            algorithm=algorithm,
            input_files=input_files
        )
        
        # 更新结果
        import json
        task.result = json.dumps(result["results"], ensure_ascii=False)
        task.status = TaskStatus.COMPLETED if result["success"] else TaskStatus.FAILED
        if not result["success"]:
            task.error = str(result.get("errors", "Unknown error"))
            
        db.commit()
    except Exception as e:
        print(f"Task execution failed: {e}")
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                db.commit()
        except:
            pass
    finally:
        db.close()


@router.get("/status/{task_id}", response_model=TaskResponse)
async def get_inference_status(task_id: str, db: Session = Depends(get_db)):
    """获取推理任务状态"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return TaskResponse(
        task_id=task.id,
        status=TaskStatus(task.status),
        message=task.error or ""
    )


@router.get("/results/{task_id}", response_model=ApiResponse)
async def get_inference_results(task_id: str, db: Session = Depends(get_db)):
    """获取推理结果"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.COMPLETED:
        return ApiResponse(
            success=False,
            message=f"任务尚未完成，当前状态: {task.status}"
        )
    
    import json
    results = json.loads(task.result) if task.result else {}
    
    return ApiResponse(
        success=True,
        data=results,
        message="推理结果获取成功"
    )


@router.post("/export-to-annotation/{task_id}", response_model=ApiResponse)
async def export_to_annotation(task_id: str, db: Session = Depends(get_db)):
    """将推理结果导出为预标注格式"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    try:
        output_path = adapter.convert_to_annotation_format(task.result)
        return ApiResponse(
            success=True,
            data={"annotation_file": output_path},
            message="已导出为标注格式，可在标注工具中加载"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
