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
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
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
        
        # --- 自动反馈逻辑 (Phase 3) ---
        try:
            # 1. 将推理结果保存为临时文件或直接传递
            # 2. 调用转换逻辑
            temp_anno_file = adapter.convert_to_annotation_format(final_result)
            
            # 3. 导入标注系统 (内部调用导入 API 的核心逻辑)
            from src.api.annotation import import_inference_results_internal
            import asyncio
            
            # 使用 asyncio.run 更加稳妥，它会自动处理事件循环的创建和关闭
            try:
                import_result = asyncio.run(import_inference_results_internal(temp_anno_file))
            except RuntimeError:
                # 如果当前线程已有运行中的循环 (虽然在 Celery Worker 中较少见)
                loop = asyncio.get_event_loop()
                import_result = loop.run_until_complete(import_inference_results_internal(temp_anno_file))
            
            # 更新任务结果信息记录导入状态
            if task:
                final_result["feedback"] = import_result
                task.result = json.dumps(final_result)
                db.commit()
        except Exception as feedback_error:
            # 反馈环节失败不应导致任务本身失败，仅记录日志
            print(f"自动反馈失败: {feedback_error}")
        # -----------------------------
        
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
