"""
Celery 任务定义
用于异步执行长时间运行的任务（训练、推理等）
"""
from celery import Celery
from datetime import datetime
import json

from configs.settings import settings

# 创建 Celery 应用
celery_app = Celery(
    "ts_iteration_loop",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Celery 配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    result_expires=86400,  # 结果保留 24 小时
)


@celery_app.task(bind=True, name="training.run")
def run_training_task(self, task_id: str, config_name: str, version_tag: str, params: dict):
    """
    执行训练任务
    
    Args:
        task_id: 任务 ID
        config_name: 训练配置名称
        version_tag: 版本标签
        params: 训练参数（learning_rate, epochs 等）
    """
    from src.adapters.chatts_training import ChatTSTrainingAdapter
    from src.db.database import SessionLocal, Task
    
    adapter = ChatTSTrainingAdapter()
    db = SessionLocal()
    
    try:
        # 更新任务状态为运行中
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "running"
            task.started_at = datetime.utcnow()
            db.commit()
        
        # 更新进度
        self.update_state(state="RUNNING", meta={
            "progress": 0,
            "message": "正在启动训练..."
        })
        
        # 执行训练
        result = adapter.run_training(
            task_id=task_id,
            config_name=config_name,
            version_tag=version_tag
        )
        
        # 更新任务状态
        if task:
            task.status = "completed" if result.get("success") else "failed"
            task.completed_at = datetime.utcnow()
            task.result = json.dumps(result)
            if not result.get("success"):
                task.error = result.get("error", "Unknown error")
            db.commit()
        
        return result
        
    except Exception as e:
        # 更新任务状态为失败
        if task:
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            task.error = str(e)
            db.commit()
        raise
    
    finally:
        db.close()


@celery_app.task(bind=True, name="inference.batch")
def run_inference_task(self, task_id: str, model: str, algorithm: str, input_files: list):
    """
    执行批量推理任务
    
    Args:
        task_id: 任务 ID
        model: 模型路径
        algorithm: 算法名称
        input_files: 输入文件列表
    """
    from src.adapters.check_outlier import CheckOutlierAdapter
    from src.db.database import SessionLocal, Task
    
    adapter = CheckOutlierAdapter()
    db = SessionLocal()
    
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "running"
            task.started_at = datetime.utcnow()
            db.commit()
        
        # 更新进度
        total_files = len(input_files)
        results = []
        
        for i, file in enumerate(input_files):
            self.update_state(state="RUNNING", meta={
                "progress": int((i / total_files) * 100),
                "message": f"处理文件 {i+1}/{total_files}: {file}"
            })
            
            # 执行单个文件推理
            result = adapter._run_single_inference(file, algorithm, {"model_path": model})
            results.append(result)
        
        final_result = {
            "success": True,
            "results": results,
            "total": total_files,
            "successful": sum(1 for r in results if r.get("success"))
        }
        
        if task:
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.result = json.dumps(final_result)
            db.commit()
        
        return final_result
        
    except Exception as e:
        if task:
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            task.error = str(e)
            db.commit()
        raise
    
    finally:
        db.close()


@celery_app.task(name="data.acquire")
def run_acquire_task(task_id: str, source: str, target_points: int, start_time: str, end_time: str):
    """
    执行数据采集任务
    """
    from src.adapters.data_processing import DataProcessingAdapter
    from src.db.database import SessionLocal, Task
    
    adapter = DataProcessingAdapter()
    db = SessionLocal()
    
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "running"
            task.started_at = datetime.utcnow()
            db.commit()
        
        result = adapter.run_acquire_task(
            task_id=task_id,
            source=source,
            target_points=target_points,
            start_time=start_time,
            end_time=end_time
        )
        
        if task:
            task.status = "completed" if result.get("success") else "failed"
            task.completed_at = datetime.utcnow()
            task.result = json.dumps(result)
            db.commit()
        
        return result
        
    except Exception as e:
        if task:
            task.status = "failed"
            task.error = str(e)
            db.commit()
        raise
    
    finally:
        db.close()
