"""
Data-Processing 项目适配器
封装数据采集和转换功能
"""
import os
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from configs.settings import settings


class DataProcessingAdapter:
    """Data-Processing 项目适配器"""
    
    def __init__(self):
        self.project_path = Path(settings.DATA_PROCESSING_PATH)
        self.scripts_path = self.project_path / "scripts"
        self.data_path = self.project_path / "data_downsampled"
    
    def list_datasets(self) -> List[Dict]:
        """列出所有数据集"""
        datasets = []
        
        if self.data_path.exists():
            for f in self.data_path.glob("*.csv"):
                stat = f.stat()
                datasets.append({
                    "name": f.stem,
                    "filename": f.name,
                    "path": str(f),
                    "size_bytes": stat.st_size,
                    "modified_time": stat.st_mtime
                })
        
        return datasets
    
    def preview_csv(self, filename: str, limit: int = 100) -> List[Dict]:
        """预览 CSV 文件"""
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {filename}")
        
        df = pd.read_csv(file_path, nrows=limit)
        return df.to_dict(orient="records")
    
    def run_acquire_task(
        self,
        task_id: str,
        source: str,
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
        
        # 构建命令
        cmd = [
            "python", str(script_path),
            "--source", source,
            "--target-points", str(target_points)
        ]
        
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
    
    def convert_annotations(self, input_dir: str, output_path: str) -> Dict:
        """
        转换标注格式
        调用 convert_annotations.py 脚本
        """
        script_path = self.scripts_path / "transformation" / "convert_annotations.py"
        
        if not script_path.exists():
            return {"success": False, "error": f"脚本不存在: {script_path}"}
        
        cmd = [
            "python", str(script_path),
            "--input-dir", input_dir,
            "--output", output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            return {
                "success": result.returncode == 0,
                "output_path": output_path,
                "stderr": result.stderr if result.returncode != 0 else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
