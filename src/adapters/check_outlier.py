"""
check_outlier 项目适配器
封装推理检测功能
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from configs.settings import settings


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
                print(f"Error stopping task {task_id}: {e}")
                return False
        return False
    
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
                    algorithm_args
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
        args: Dict
    ) -> Dict:
        """执行单个文件推理"""
        # 使用 Python 解释器（统一模式使用 PYTHON_UNIFIED）
        python_exe = settings.PYTHON_UNIFIED if settings.USE_LOCAL_MODULES else settings.PYTHON_ILABEL
        cmd = [
            python_exe, str(self.run_script),
            "--input", input_file,
            "--method", algorithm
        ]
        
        # 处理参数映射
        # 1. 必需参数
        # 默认降采样点数，如果没有传入 n_downsample，则使用 settings 的默认值
        if "n_downsample" not in args:
             cmd.extend(["--n_downsample", str(settings.DEFAULT_DOWNSAMPLE_POINTS)])
        
        # 2. 遍历 args 添加参数
        for k, v in args.items():
            if v is None or v == "":
                continue
            
            # 特殊处理内部标记
            if k == "chatts_enabled":
                cmd.append("--use-chatts")
                continue
            if k == "model_path": # 兼容旧代码，虽然上面改成了 chatts_model_path
                 cmd.extend(["--model", str(v)])
                 continue

            # 处理布尔值参数 (例如 --chatts_load_in_4bit)
            # 注意：run.py 中某些布尔参数可能是接收字符串 "true"/"false" 或 action="store_true"
            # 根据 default_params.json 分析，大部分是字符串类型的 true/false 或 auto
            
            # 将下划线转换为连字符，例如 chatts_load_in_4bit -> --chatts-load-in-4bit
            arg_name = f"--{k.replace('_', '-')}"
            
            # 防止重复添加 method
            if arg_name == "--method":
                continue
            
            cmd.extend([arg_name, str(v)])
        
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
            "--method", algorithm
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
                            result_dir = parts[-1].strip()
                            # 存储目录供后续使用
                            if not hasattr(self, '_current_result_dir'):
                                self._current_result_dir = result_dir

                    # 捕获实际保存的文件名
                    # Log format: "保存结果: filename.csv"
                    if "保存结果" in stripped_line and ".csv" in stripped_line:
                        parts = stripped_line.split(":")
                        if len(parts) >= 2:
                            filename = parts[-1].strip()
                            # 构建完整路径
                            if hasattr(self, '_current_result_dir'):
                                full_path = f"{self._current_result_dir}/{filename}"
                            else:
                                # 默认目录
                                full_path = f"/home/share/results/data/global/chatts/{filename}"
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

