"""
Data-Processing 项目适配器
封装数据采集与数据集文件管理功能
"""
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from configs.settings import settings
from src.utils.file_filters import is_inference_or_generated_csv


class DataProcessingAdapter:
    """Data-Processing 项目适配器"""
    
    def __init__(self):
        self.project_path = Path(settings.DATA_PROCESSING_PATH)
        self.scripts_path = self.project_path / "scripts"
        # 使用标准化数据目录
        self.data_path = Path(settings.DATA_DOWNSAMPLED_DIR)

    def _resolve_dataset_path(self, filename: str) -> Path:
        """Resolve a dataset file path and ensure it stays under data_path."""
        candidate = (self.data_path / str(filename or "")).resolve()
        root = self.data_path.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("非法文件路径") from exc
        if candidate.suffix.lower() != ".csv":
            raise ValueError("仅支持 CSV 文件")
        return candidate
    
    def list_datasets(self) -> List[Dict]:
        """列出所有数据集"""
        datasets = []
        
        if self.data_path.exists():
            for f in self.data_path.glob("*.csv"):
                try:
                    if not f.exists():
                        continue
                    
                    # 过滤掉推理结果/中间文件
                    if is_inference_or_generated_csv(f.name):
                        continue
                        
                    stat = f.stat()
                    datasets.append({
                        "name": f.stem,
                        "filename": f.name,
                        "path": str(f),
                        "size_bytes": stat.st_size,
                        "modified_time": stat.st_mtime
                    })
                except OSError:
                    # Skip files that cause errors (e.g. deleted during iteration)
                    continue
        
        return datasets
    
    def preview_csv(self, filename: str, limit: int = 5000) -> List[Dict]:
        """预览 CSV 文件"""
        file_path = self._resolve_dataset_path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {filename}")
        
        df = pd.read_csv(file_path, nrows=limit)
        return df.to_dict(orient="records")
    
    def delete_dataset(self, filename: str) -> Dict:
        """删除数据集文件及关联的图片"""
        try:
            file_path = self._resolve_dataset_path(filename)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        if not file_path.exists():
            return {"success": False, "error": f"文件不存在: {filename}"}
        
        try:
            # 删除 CSV 文件
            file_path.unlink()
            
            # 删除关联的图片文件（如果存在）
            img_name = filename.replace(".csv", ".jpg")
            img_path = Path(settings.DATA_IMAGES_DIR) / img_name
            deleted_img = False
            if img_path.exists():
                img_path.unlink()
                deleted_img = True
            
            if deleted_img:
                return {"success": True, "message": f"已删除: {filename} 和关联图片"}
            return {"success": True, "message": f"已删除: {filename}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_acquire_task(
        self,
        task_id: str,
        source: str,
        host: str = "192.168.199.185",
        port: str = "6667",
        user: str = "",
        password: str = "",
        point_name: str = "*",
        target_points: int = 5000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict:
        """
        执行数据采集任务
        调用 get_downsampled.py 脚本
        """
        script_path = self.scripts_path / "acquisition" / "get_downsampled.py"
        
        if not script_path.exists():
            return {"success": False, "error": f"脚本不存在: {script_path}"}
        if not str(user or "").strip() or not str(password or "").strip():
            return {"success": False, "error": "IoTDB 用户名和密码不能为空"}
        
        # 构建命令 - 使用 Python 解释器（统一模式使用 PYTHON_UNIFIED）
        python_exe = settings.PYTHON_UNIFIED if settings.USE_LOCAL_MODULES else settings.PYTHON_DATA_PROCESSING
        cmd = [
            python_exe, str(script_path),
            "--source", source,
            "--host", host,
            "--port", port,
            "--user", user,
            "--password", password,
            "--target-points", str(target_points),
            "--output-dir", str(self.data_path)
        ]
        
        if point_name and point_name != "*":
             cmd.extend(["--column", point_name])
        
        if start_time:
            cmd.extend(["--start-time", start_time])
        if end_time:
            cmd.extend(["--end-time", end_time])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "任务超时"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_acquire_task_streaming(
        self,
        task_id: str,
        source: str,
        host: str = "192.168.199.185",
        port: str = "6667",
        user: str = "",
        password: str = "",
        point_name: str = "*",
        target_points: int = 5000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ):
        """
        执行数据采集任务（流式输出版）
        使用生成器 yield 实时输出日志
        """
        script_path = self.scripts_path / "acquisition" / "get_downsampled.py"
        
        if not script_path.exists():
            yield f"❌ 脚本不存在: {script_path}"
            return
        if not str(user or "").strip() or not str(password or "").strip():
            yield "❌ IoTDB 用户名和密码不能为空"
            return
        
        # 构建命令
        python_exe = settings.PYTHON_UNIFIED if settings.USE_LOCAL_MODULES else settings.PYTHON_DATA_PROCESSING
        cmd = [
            python_exe, str(script_path),
            "--source", source,
            "--host", host,
            "--port", port,
            "--user", user,
            "--password", password,
            "--target-points", str(target_points),
            "--output-dir", str(self.data_path),
            "--image-dir", settings.DATA_IMAGES_DIR,
        ]
        
        if point_name and point_name != "*":
            cmd.extend(["--column", point_name])
        if start_time:
            cmd.extend(["--start-time", start_time])
        if end_time:
            cmd.extend(["--end-time", end_time])
        
        yield f"🚀 Starting acquisition...\n\n**Command:** `{' '.join(cmd[:3])}...`\n\n---\n"
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line.rstrip())
                    # 只显示最近 20 行日志，避免输出过多
                    display_lines = output_lines[-20:]
                    yield f"🔄 **Acquiring data...**\n\n```\n" + "\n".join(display_lines) + "\n```"
            
            process.wait()
            
            if process.returncode == 0:
                yield f"✅ **Acquisition completed!**\n\n```\n" + "\n".join(output_lines[-10:]) + "\n```"
            else:
                yield f"❌ **Acquisition failed** (code: {process.returncode})\n\n```\n" + "\n".join(output_lines[-20:]) + "\n```"
                
        except Exception as e:
            yield f"❌ **Error:** {str(e)}"
