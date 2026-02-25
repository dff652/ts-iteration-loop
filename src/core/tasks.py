"""
Celery 任务定义
用于异步执行长时间运行的任务（训练、推理等）
"""
from celery import Celery
import json

from configs.settings import settings
from src.core.logging_config import get_logger
from src.utils.time_utils import utc_now_naive

logger = get_logger(__name__)

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
def run_training_task(
    self,
    task_id: str,
    config_name: str,
    version_tag: str,
    model_family: str = "chatts",
    auto_eval: bool = False,
    eval_truth_dir: str | None = None,
    eval_data_dir: str | None = None,
    eval_dataset_name: str | None = None,
    eval_output_dir: str | None = None,
    eval_device: str | None = None,
    eval_method: str | None = None,
    params: dict | None = None,
):
    """
    执行训练任务
    
    Args:
        task_id: 任务 ID
        config_name: 训练配置名称
        version_tag: 版本标签
        model_family: 模型族（chatts/qwen）
    """
    from src.adapters.chatts_training import ChatTSTrainingAdapter
    from src.db.database import SessionLocal, Task
    
    adapter = ChatTSTrainingAdapter(model_family=model_family or "chatts")
    db = SessionLocal()
    
    try:
        # 更新任务状态为运行中
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "running"
            task.started_at = utc_now_naive()
            db.commit()
        
        # 更新进度
        self.update_state(state="RUNNING", meta={
            "progress": 0,
            "message": "正在启动训练..."
        })
        
        # 执行训练
        allowed_override_keys = {
            "override_model_path",
            "override_dataset",
            "override_learning_rate",
            "override_epochs",
            "override_batch_size",
            "override_lora_rank",
            "override_lora_alpha",
            "override_grad_accum_steps",
            "override_cutoff_len",
            "override_precision",
            "override_image_max_pixels",
            "override_image_min_pixels",
            "override_nproc_per_node",
            "override_cuda_visible_devices",
            "override_extra_args",
            "override_logging_steps",
            "override_save_steps",
            "override_warmup_steps",
            "override_warmup_ratio",
            "override_lr_scheduler_type",
            "override_lora_dropout",
            "override_lora_target",
            "override_freeze_vision_tower",
            "override_freeze_multi_modal_projector",
            "override_freeze_trainable_layers",
            "override_freeze_trainable_modules",
        }
        runtime_overrides = {}
        if isinstance(params, dict):
            runtime_overrides = {
                str(k): v
                for k, v in params.items()
                if isinstance(k, str) and k in allowed_override_keys
            }

        result = adapter.run_training(
            task_id=task_id,
            config_name=config_name,
            version_tag=version_tag,
            auto_eval=auto_eval,
            eval_truth_dir=eval_truth_dir,
            eval_data_dir=eval_data_dir,
            eval_dataset_name=eval_dataset_name,
            eval_output_dir=eval_output_dir,
            eval_device=eval_device,
            eval_method=eval_method,
            **runtime_overrides,
        )
        
        # 更新任务状态
        if task:
            task.status = "completed" if result.get("success") else "failed"
            task.completed_at = utc_now_naive()
            task.result = json.dumps(result)
            if not result.get("success"):
                task.error = result.get("error", "Unknown error")
            db.commit()
        
        return result
        
    except Exception as e:
        # 更新任务状态为失败
        if task:
            task.status = "failed"
            task.completed_at = utc_now_naive()
            task.error = str(e)
            db.commit()
        raise
    
    finally:
        db.close()


@celery_app.task(bind=True, name="inference.batch")
def run_inference_task(
    self,
    task_id: str,
    model: str,
    algorithm: str,
    input_files: list,
    params: dict | None = None,
):
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
        if task and task.status == "cancelled":
            return {"success": False, "cancelled": True, "results": [], "total": len(input_files), "successful": 0}
        if task:
            task.status = "running"
            task.started_at = utc_now_naive()
            db.commit()
        
        # 更新进度
        total_files = len(input_files)
        results = []
        errors = []
        runtime_args = adapter._build_algorithm_args(algorithm, model)
        if isinstance(params, dict):
            runtime_args.update(params)
        
        for i, file in enumerate(input_files):
            task = db.query(Task).filter(Task.id == task_id).first()
            if task and task.status == "cancelled":
                break
            self.update_state(state="RUNNING", meta={
                "progress": int((i / total_files) * 100),
                "message": f"处理文件 {i+1}/{total_files}: {file}"
            })
            
            # 执行单个文件推理
            result = adapter._run_single_inference(
                file,
                algorithm,
                runtime_args,
                task_id=task_id,
            )
            results.append(result)
            if not result.get("success", False):
                errors.append({"file": file, "error": result.get("error", "Unknown error")})
        
        final_result = {
            "success": len(errors) == 0 and not (task and task.status == "cancelled"),
            "results": results,
            "total": total_files,
            "successful": sum(1 for r in results if r.get("success")),
            "errors": errors,
        }
        
        if task:
            if task.status == "cancelled":
                task.completed_at = utc_now_naive()
                task.result = json.dumps(final_result)
                db.commit()
                return final_result
            task.status = "completed" if final_result["success"] else "failed"
            task.completed_at = utc_now_naive()
            task.result = json.dumps(final_result)
            if not final_result["success"] and errors:
                task.error = "; ".join([str(e.get("error")) for e in errors[:3]])
            db.commit()
        
        # --- 自动反馈逻辑 (Phase 3) ---
        try:
            # 1. 直接在内存中转换为标注行（不依赖临时 JSON 文件）
            rows = adapter.to_annotation_rows(final_result)

            # 2. 导入标注系统
            from src.adapters.annotation_import import AnnotationImportAdapter
            import asyncio

            import_adapter = AnnotationImportAdapter(annotator_api_url=settings.ANNOTATOR_API_URL)

            if not rows:
                import_result = {"success": False, "count": 0, "errors": [{"error": "no annotation rows"}]}
            else:
                try:
                    import_result = asyncio.run(import_adapter.import_rows(rows))
                except RuntimeError:
                    # 如果当前线程已有运行中的循环 (虽然在 Celery Worker 中较少见)
                    loop = asyncio.get_event_loop()
                    import_result = loop.run_until_complete(import_adapter.import_rows(rows))
            
            # 更新任务结果信息记录导入状态
            if task:
                final_result["feedback"] = import_result
                task.result = json.dumps(final_result)
                db.commit()
        except Exception as feedback_error:
            # 反馈环节失败不应导致任务本身失败，仅记录日志
            logger.warning("自动反馈失败: %s", feedback_error, exc_info=True)
        # -----------------------------
        
        return final_result
        
    except Exception as e:
        if task:
            task.status = "failed"
            task.completed_at = utc_now_naive()
            task.error = str(e)
            db.commit()
        raise
    
    finally:
        db.close()


@celery_app.task(name="data.acquire")
def run_acquire_task(
    task_id: str,
    source: str,
    target_points: int,
    start_time: str | None,
    end_time: str | None,
    host: str = "192.168.199.185",
    port: str = "6667",
    user: str = "root",
    password: str = "root",
    point_name: str = "*",
):
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
            task.started_at = utc_now_naive()
            db.commit()
        
        result = adapter.run_acquire_task(
            task_id=task_id,
            source=source,
            host=host,
            port=port,
            user=user,
            password=password,
            point_name=point_name,
            target_points=target_points,
            start_time=start_time,
            end_time=end_time
        )
        
        if task:
            task.status = "completed" if result.get("success") else "failed"
            task.completed_at = utc_now_naive()
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
