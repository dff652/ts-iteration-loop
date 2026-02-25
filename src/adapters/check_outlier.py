"""
check_outlier 项目适配器
封装推理检测功能
"""
import os
import sys
import json
import uuid
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any

from configs.settings import settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class CheckOutlierAdapter:
    """check_outlier 项目适配器"""
    
    def __init__(self):
        self.project_path = Path(settings.CHECK_OUTLIER_PATH)
        self.project_path = Path(settings.CHECK_OUTLIER_PATH)
        self.run_script = self.project_path / "run.py"
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.cancelled_tasks: set = set()

    def stop_inference_task(self, task_id: str):
        """停止指定的推理任务"""
        self.cancelled_tasks.add(task_id)
        if task_id in self.active_processes:
            try:
                self.active_processes[task_id].terminate()
                self.active_processes[task_id].kill()  # Force kill to be safe
                del self.active_processes[task_id]
                return True
            except Exception as e:
                logger.error("停止推理任务失败 task_id=%s: %s", task_id, e)
                return False
        # If not in active_processes, we still marked it as cancelled, so return True
        return True

    def convert_to_annotation_format(self, inference_result: Any) -> str:
        """
        将推理结果转换为标注工具可导入的 JSON 文件。

        支持输入:
        - dict: {"results": [...]} 或已是 {"filename","annotations"} 结构
        - list: 多条结果
        - str: JSON 字符串或 JSON 文件路径
        """
        rows = self.to_annotation_rows(inference_result)

        output_dir = Path("/tmp/ts_iteration_loop")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"inference_annotations_{uuid.uuid4().hex}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return str(output_path)

    def to_annotation_rows(self, inference_result: Any) -> List[Dict]:
        """
        将推理结果转换为标注行（内存结构，不落盘）。
        """
        payload = inference_result
        if isinstance(inference_result, str):
            candidate_path = Path(inference_result.strip())
            if candidate_path.exists():
                with open(candidate_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                payload = json.loads(inference_result)
        return self._extract_annotation_rows(payload)

    def _extract_annotation_rows(self, payload: Any) -> List[Dict]:
        if payload is None:
            return []

        if isinstance(payload, list):
            rows: List[Dict] = []
            for item in payload:
                rows.extend(self._normalize_result_item(item))
            return rows

        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                rows: List[Dict] = []
                for item in payload["results"]:
                    rows.extend(self._normalize_result_item(item))
                return rows
            return self._normalize_result_item(payload)

        return []

    def _normalize_result_item(self, item: Any) -> List[Dict]:
        if not isinstance(item, dict):
            return []

        # Already in annotator import format.
        filename = item.get("filename") or item.get("file")
        annotations = item.get("annotations")
        if filename and isinstance(annotations, list):
            return [{
                "filename": filename,
                "annotations": annotations,
                "source": item.get("source", "inference")
            }]

        # Wrapped result from adapter output: {"file": "...", "result": ...}
        wrapped_result = item.get("result")
        if wrapped_result is None:
            return []

        return self._parse_wrapped_result(wrapped_result, fallback_filename=item.get("file"))

    def _parse_wrapped_result(self, wrapped_result: Any, fallback_filename: Optional[str]) -> List[Dict]:
        if isinstance(wrapped_result, str):
            try:
                wrapped_result = json.loads(wrapped_result)
            except Exception:
                return []

        if isinstance(wrapped_result, list):
            rows: List[Dict] = []
            for entry in wrapped_result:
                if not isinstance(entry, dict):
                    continue
                if entry.get("filename") and isinstance(entry.get("annotations"), list):
                    rows.append({
                        "filename": entry["filename"],
                        "annotations": entry["annotations"],
                        "source": entry.get("source", "inference")
                    })
            return rows

        if isinstance(wrapped_result, dict):
            # Single record already in target format.
            if wrapped_result.get("filename") and isinstance(wrapped_result.get("annotations"), list):
                return [{
                    "filename": wrapped_result["filename"],
                    "annotations": wrapped_result["annotations"],
                    "source": wrapped_result.get("source", "inference")
                }]

            # Legacy shape from mocked tests.
            anomalies = wrapped_result.get("detected_anomalies") or wrapped_result.get("anomalies") or []
            if not fallback_filename or not isinstance(anomalies, list):
                return []

            converted_annotations = []
            for idx, anomaly in enumerate(anomalies):
                if not isinstance(anomaly, dict):
                    continue
                segment = self._extract_segment(anomaly)
                if segment is None:
                    continue
                converted_annotations.append({
                    "label": str(anomaly.get("type") or anomaly.get("label") or f"inference_{idx+1}"),
                    "color": "#d946ef",
                    "segments": [segment],
                    "analysis": anomaly.get("reason") or anomaly.get("analysis") or ""
                })

            if converted_annotations:
                return [{
                    "filename": fallback_filename,
                    "annotations": converted_annotations,
                    "source": "inference"
                }]

        return []

    def _extract_segment(self, anomaly: Dict) -> Optional[Dict]:
        interval = anomaly.get("interval") or anomaly.get("segment")
        if isinstance(interval, (list, tuple)) and len(interval) >= 2:
            try:
                start = int(interval[0])
                end = int(interval[1])
                if end < start:
                    start, end = end, start
                return {"start": start, "end": end}
            except Exception:
                return None

        if "start" in anomaly and "end" in anomaly:
            try:
                start = int(anomaly["start"])
                end = int(anomaly["end"])
                if end < start:
                    start, end = end, start
                return {"start": start, "end": end}
            except Exception:
                return None

        return None
    
    def run_batch_inference(
        self,
        task_id: str,
        model: str,
        algorithm: str,
        input_files: List[str],
        **kwargs
    ) -> Dict:
        """
        执行批量推理任务
        调用 run.py 脚本
        """
        if not self.run_script.exists():
            return {"success": False, "error": f"脚本不存在: {self.run_script}"}
        
        # 根据算法选择基础参数
        algorithm_args = self._build_algorithm_args(algorithm, model)
        
        # 合并额外的 UI 参数
        algorithm_args.update(kwargs)
        
        # 构建配置文件或命令行参数
        results = []
        errors = []
        
        for input_file in input_files:
            try:
                result = self._run_single_inference(
                    input_file, 
                    algorithm, 
                    algorithm_args,
                    task_id=task_id
                )
                results.append(result)
            except Exception as e:
                errors.append({"file": input_file, "error": str(e)})
        
        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors,
            "total": len(input_files),
            "successful": len(results)
        }
    
    def _build_algorithm_args(self, algorithm: str, model: str) -> Dict:
        """构建算法参数"""
        if algorithm == "chatts":
            return {
                "method": "chatts",
                "chatts_model_path": model, # 注意：这是传给 run.py 的 --chatts_model_path
                "chatts_enabled": True
            }
        elif algorithm == "qwen":
            return {
                "method": "qwen",
                "chatts_model_path": model, # Pass model path to reused argument in run.py
                "chatts_enabled": False # Not using legacy ChatTS logic but new Qwen block
            }
        elif algorithm == "adtk_hbos":
            return {
                "method": "adtk_hbos",
                "chatts_enabled": False
            }
        else:
            return {"method": algorithm}
    
    def _run_single_inference(
        self, 
        input_file: str, 
        algorithm: str,
        args: Dict,
        task_id: Optional[str] = None
    ) -> Dict:
        """执行单个文件推理"""
        # 使用 Python 解释器（统一模式使用 PYTHON_UNIFIED）
        python_exe = settings.PYTHON_UNIFIED if settings.USE_LOCAL_MODULES else settings.PYTHON_ILABEL
        cmd = [
            python_exe, str(self.run_script),
            "--input", input_file,
            "--method", algorithm,
            "--task_name", ""  # Suppress default 'global' subfolder
        ]
        if task_id:
            cmd.extend(["--task-id", str(task_id)])
        
        if "n_downsample" not in args:
            cmd.extend(["--n_downsample", str(settings.DEFAULT_DOWNSAMPLE_POINTS)])

        # Match streaming path behavior so API/Celery and UI local mode keep parity.
        if algorithm == "chatts":
            if args.get("lora_adapter_path"):
                cmd.extend(["--chatts_lora_adapter_path", str(args["lora_adapter_path"])])
            if args.get("base_model_path"):
                cmd.extend(["--chatts_model_path", str(args["base_model_path"])])

        if algorithm == "qwen":
            if args.get("base_model_path"):
                cmd.extend(["--qwen_model_path", str(args["base_model_path"])])
        
        # 处理参数映射
        # 遍历 args 添加参数
        for k, v in args.items():
            if v is None or v == "":
                continue
            
            # 跳过已处理或无效参数
            if k in ["model", "base_model_path", "lora_adapter_path", "chatts_enabled"]:
                continue

            # 防止重复添加 method
            if k == "method":
                continue

            # 与流式路径一致：保持下划线参数名
            arg_name = f"--{k}"
            cmd.extend([arg_name, str(v)])
        
        # 强制指定输出路径，确保与系统统一配置一致
        if "--data_path" not in cmd:
            cmd.extend(["--data_path", settings.DATA_INFERENCE_DIR])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            # 尝试解析输出为 JSON
            output = result.stdout.strip()
            parsed = None
            
            # 尝试提取标记之间的 JSON
            import re
            match = re.search(r"__JSON_START__\s*(.*?)\s*__JSON_END__", output, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    parsed = json.loads(json_str)
                    # parsed 应该是一个 list, 我们取第一个或者合并
                    # run.py 输出的是 [{"filename":..., "annotations":...}, ...]
                    # 这里的 adapter 是 _run_single_inference, 所以通常只有一个结果，但这取决于 run.py 是怎么被调用的
                    # 如果 input_file 只有一个，run.py loop 只跑一次，list len=1
                    # 我们返回整个 list 或者 result
                except:
                    pass
            
            if parsed is None:
                # Fallback: try parsing whole output (legacy behavior)
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    parsed = {"raw_output": output}
            
            return {
                "file": input_file,
                "success": result.returncode == 0,
                "result": parsed
            }
        except subprocess.TimeoutExpired:
            return {"file": input_file, "success": False, "error": "超时"}
        except Exception as e:
            return {"file": input_file, "success": False, "error": str(e)}
    
    def run_batch_inference_streaming(
        self,
        task_id: str,
        model: str,
        algorithm: str,
        input_files: List[str],
        **kwargs
    ):
        """
        执行批量推理任务（流式输出版）
        使用生成器 yield 实时输出日志
        """
        if not self.run_script.exists():
            yield f"❌ 脚本不存在: {self.run_script}"
            return
        
        # 根据算法选择基础参数
        algorithm_args = self._build_algorithm_args(algorithm, model)
        algorithm_args.update(kwargs)
        
        total_files = len(input_files)
        yield f"🚀 Starting batch inference for {total_files} files...\n"
        
        success_count = 0
        failed_count = 0
        errors = []
        
        for idx, input_file in enumerate(input_files, 1):
            if task_id in self.cancelled_tasks:
                yield "🛑 Task cancelled by user.\n"
                break
                
            yield f"\n📄 **Processing file ({idx}/{total_files}):** `{Path(input_file).name}`\n"
            
            try:
                # 执行单个文件推理（流式）
                file_success = False
                for log in self._run_single_inference_streaming(
                    task_id,
                    input_file, 
                    algorithm, 
                    algorithm_args
                ):
                    # 检查是否是完成标记（自定义协议，或者仅作为日志输出）
                    if isinstance(log, dict) and "success" in log:
                         file_success = log["success"]
                         if not file_success:
                             errors.append({"file": input_file, "error": log.get("error", "Unknown error")})
                    else:
                        yield log
                
                if file_success:
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                errors.append({"file": input_file, "error": str(e)})
                yield f"❌ Error processing {input_file}: {str(e)}\n"
        
        # 汇总结果
        yield f"\n\n---\n✅ **Batch Inference Completed**\n"
        yield f"- Success: {success_count}\n"
        yield f"- Failed: {failed_count}\n"
        
        if errors:
            yield "\n**Errors:**\n"
            for e in errors:
                yield f"- {Path(e['file']).name}: {e['error']}\n"

    def _run_single_inference_streaming(
        self, 
        task_id: str,
        input_file: str, 
        algorithm: str,
        args: Dict
    ):
        """执行单个文件推理（流式）"""
        # 构建命令（与 _run_single_inference 保持一致）
        python_exe = settings.PYTHON_UNIFIED if settings.USE_LOCAL_MODULES else settings.PYTHON_ILABEL
        cmd = [
            python_exe, str(self.run_script),
            "--input", input_file,
            "--method", algorithm,
            "--task_name", ""  # Suppress default 'global' subfolder
        ]
        
        # 处理参数映射
        if "n_downsample" not in args:
             cmd.extend(["--n_downsample", str(settings.DEFAULT_DOWNSAMPLE_POINTS)])
        
        # 特殊处理 ChatTS 模型参数
        if algorithm == "chatts":
            # LoRA Adapter
            if args.get("lora_adapter_path"):
                 cmd.extend(["--chatts_lora_adapter_path", str(args["lora_adapter_path"])])
            # Base Model (如果未指定，run.py 会使用默认值，这里显式传递更安全)
            if args.get("base_model_path"):
                 cmd.extend(["--chatts_model_path", str(args["base_model_path"])])
        
        # 新增 Qwen 参数处理
        if algorithm == "qwen":
            if args.get("base_model_path"):
                 cmd.extend(["--qwen_model_path", str(args["base_model_path"])])
            
        for k, v in args.items():
            if v is None or v == "": continue
            
            # 跳过已处理的参数
            if k in ["model", "base_model_path", "lora_adapter_path"]: continue
            if k == "chatts_enabled": continue 
            
            # 特殊处理布尔值
            if k == "timer_streaming":
                if str(v).lower() == "true":
                    cmd.extend(["--timer_streaming", "True"])
                continue
            
            # 常规参数处理
            arg_name = f"--{k}"
            cmd.extend([arg_name, str(v)])

        # 强制指定输出路径，确保与系统统一配置一致
        if "--data_path" not in cmd:
            cmd.extend(["--data_path", settings.DATA_INFERENCE_DIR])
            
        yield f"Running command: `{' '.join(cmd[:10])}...`\n"

        process = None
        try:
            # 使用 Popen 获取实时输出
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # 注册进程
            if task_id:
                self.active_processes[task_id] = process
            
            output_lines = []
            
            # Local variable to track result directory for this specific run
            current_result_dir = None

            # 实时读取日志
            for line in iter(process.stdout.readline, ''):
                if line:
                    stripped_line = line.rstrip()
                    output_lines.append(stripped_line)
                    
                    # 捕获结果文件目录
                    # Log format: "Saving results to: /path/to/dir"
                    if "Saving results to:" in stripped_line:
                        parts = stripped_line.split("Saving results to:")
                        if len(parts) >= 2:
                            current_result_dir = parts[-1].strip()

                    # 捕获实际保存的文件名
                    # Log format: "保存结果: filename.csv"
                    if "保存结果" in stripped_line and ".csv" in stripped_line:
                        parts = stripped_line.split(":")
                        if len(parts) >= 2:
                            filename = parts[-1].strip()
                            # 构建完整路径
                            if current_result_dir:
                                full_path = f"{current_result_dir}/{filename}"
                            else:
                                # Fallback: use algorithm specific directory
                                default_dir = os.path.join(settings.DATA_INFERENCE_DIR, algorithm)
                                full_path = f"{default_dir}/{filename}"
                            yield {"file_path": full_path, "file_name": filename}

                    # 过滤并格式化有用信息
                    # 1. 进度条 (tqdm)
                    if "%" in stripped_line or "it/s" in stripped_line:
                        # 对于进度条，使用行内代码块或特定格式，避免刷屏
                         if "ChatTS 处理进度" in stripped_line or "Loading" in stripped_line:
                             yield f"> {stripped_line}\n"
                         continue

                    # 2. 关键状态信息
                    if stripped_line.startswith("[ChatTS]") or "Data shape" in stripped_line or "Saving results" in stripped_line:
                         yield f"- {stripped_line}\n"
                    elif "Error" in stripped_line or "Exception" in stripped_line:
                         yield f"❌ **{stripped_line}**\n"
                    # 3. 允许更多调试信息通过
                    elif any(k in stripped_line for k in ["Traceback", "File \"", "line ", "KeyError", "ValueError", "DEBUG", "INFO"]):
                        yield f"```\n{stripped_line}\n```\n"
                    # 4. 默认显示其他非空行 (作为引言，避免太乱)
                    else:
                        yield f"> {stripped_line}\n"
            
            process.wait()
            
            # 进程结束后移除
            if task_id and task_id in self.active_processes:
                del self.active_processes[task_id]

            # 解析最后的结果（仅为了判断成功与否，不用于返回大量数据）
            # 注意：流式模式下我们无法像 run_batch_inference 那样方便地返回结构化结果
            # 这里我们只返回一个状态标记
            if process.returncode == 0:
                yield {"success": True}
                yield "✅ Finished processing file.\n"
            else:
                yield {"success": False, "error": f"Process exited with code {process.returncode}"}
                yield f"❌ Process failed with code {process.returncode}\n"
                # 输出最后几行日志作为错误上下文
                yield "```\n" + "\n".join(output_lines[-10:]) + "\n```\n"
                
        except Exception as e:
            yield {"success": False, "error": str(e)}
            yield f"❌ Execution error: {str(e)}\n"
