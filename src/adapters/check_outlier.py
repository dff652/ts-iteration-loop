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
        self.run_script = self.project_path / "run.py"
    
    def run_batch_inference(
        self,
        task_id: str,
        model: str,
        algorithm: str,
        input_files: List[str]
    ) -> Dict:
        """
        执行批量推理任务
        调用 run.py 脚本
        """
        if not self.run_script.exists():
            return {"success": False, "error": f"脚本不存在: {self.run_script}"}
        
        # 根据算法选择参数
        algorithm_args = self._build_algorithm_args(algorithm, model)
        
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
                "model_path": model,
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
        cmd = [
            "python", str(self.run_script),
            "--input", input_file,
            "--method", algorithm
        ]
        
        # 添加其他参数
        if args.get("chatts_enabled"):
            cmd.append("--use-chatts")
            if args.get("model_path"):
                cmd.extend(["--model", args["model_path"]])
        
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
    
    def convert_to_annotation_format(self, inference_result: str) -> str:
        """
        将推理结果转换为标注工具可加载的格式
        用于迭代循环中的反馈机制
        """
        try:
            results = json.loads(inference_result) if isinstance(inference_result, str) else inference_result
        except json.JSONDecodeError:
            raise ValueError("无法解析推理结果")
        
        # 转换为标注格式
        annotations = []
        
        for item in results.get("results", []):
            if not item.get("success"):
                continue
            
            result = item.get("result", {})
            anomalies = result.get("detected_anomalies", [])
            
            annotation = {
                "filename": item["file"],
                "annotations": [],
                "source": "inference"  # 标记来源为推理
            }
            
            for anomaly in anomalies:
                annotation["annotations"].append({
                    "label": {
                        "text": anomaly.get("type", "异常"),
                        "category": "局部变化"
                    },
                    "segments": [{
                        "start": anomaly.get("interval", [0, 0])[0],
                        "end": anomaly.get("interval", [0, 0])[1]
                    }],
                    "analysis": anomaly.get("reason", "")
                })
            
            annotations.append(annotation)
        
        # 保存为 JSON 文件
        output_path = self.project_path.parent / "inference_annotations.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        return str(output_path)
