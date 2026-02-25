"""
标注导出适配器
负责将标注 JSON 转为训练格式（ChatTS/Qwen）。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

from configs.settings import settings


class AnnotationExportAdapter:
    """封装 Data-Processing 的 convert_annotations.py 导出能力。"""

    def __init__(self, project_path: Optional[str] = None):
        self.project_path = Path(project_path or settings.DATA_PROCESSING_PATH)
        self.scripts_path = self.project_path / "scripts"
        self.default_image_dir = Path(settings.DATA_DOWNSAMPLED_DIR)

    def _python_executable(self) -> str:
        return settings.PYTHON_UNIFIED if settings.USE_LOCAL_MODULES else settings.PYTHON_DATA_PROCESSING

    @staticmethod
    def _resolve_output_path_from_stdout(stdout: str, fallback: str) -> str:
        match = re.search(r"所有转换结果已保存到:\s*(.+)", stdout or "")
        if not match:
            match = re.search(r"单文件已更新至:\s*(.+)", stdout or "")
        if match:
            candidate = str(match.group(1)).strip()
            if candidate:
                return candidate
        return fallback

    def convert_annotations(
        self,
        input_dir: str,
        output_path: str,
        image_dir: Optional[str] = None,
        filename: Optional[str] = None,
        model_family: str = "qwen",
        csv_src_dir: Optional[str] = None,
    ) -> Dict:
        script_path = self.scripts_path / "transformation" / "convert_annotations.py"
        if not script_path.exists():
            return {"success": False, "error": f"脚本不存在: {script_path}"}

        resolved_image_dir = image_dir or str(self.default_image_dir)
        cmd = [
            self._python_executable(),
            str(script_path),
            "--input-dir",
            input_dir,
            "--image-dir",
            resolved_image_dir,
            "--output",
            output_path,
            "--format",
            model_family,
        ]

        if csv_src_dir:
            cmd.extend(["--csv-src", csv_src_dir])
        if filename:
            cmd.extend(["--file", filename])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=600,
            )
            output_path_final = self._resolve_output_path_from_stdout(
                result.stdout or "",
                output_path,
            )
            return {
                "success": result.returncode == 0,
                "output_path": output_path_final,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
