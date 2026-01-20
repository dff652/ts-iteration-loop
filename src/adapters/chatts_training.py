"""
ChatTS-Training 项目适配器
封装模型微调功能
"""
import os
import json
import subprocess
import signal
from pathlib import Path
from typing import List, Dict, Optional

from configs.settings import settings


class ChatTSTrainingAdapter:
    """ChatTS-Training 项目适配器"""
    
    def __init__(self):
        self.project_path = Path(settings.CHATTS_TRAINING_PATH)
        self.scripts_path = self.project_path / "scripts"
        self.saves_path = self.project_path / "saves"
        self.data_path = self.project_path / "data"
        
        # 运行中的训练进程
        self._running_processes: Dict[str, subprocess.Popen] = {}
    
    def list_configs(self) -> List[Dict]:
        """列出可用的训练配置"""
        configs = []
        
        # 扫描 lora 目录
        lora_scripts = self.scripts_path / "lora"
        if lora_scripts.exists():
            for f in lora_scripts.glob("*.sh"):
                configs.append({
                    "name": f.stem,
                    "path": str(f),
                    "method": "lora",
                    "description": f"LoRA 微调: {f.stem}"
                })
        
        # 扫描 full 目录
        full_scripts = self.scripts_path / "full"
        if full_scripts.exists():
            for f in full_scripts.glob("*.sh"):
                configs.append({
                    "name": f.stem,
                    "path": str(f),
                    "method": "full",
                    "description": f"Full SFT: {f.stem}"
                })
        
        return configs
    
    def list_models(self) -> List[Dict]:
        """
        列出已训练的模型，包含训练产物信息
        
        返回格式:
        {
            "name": "RTX6000_tune_...",
            "path": "/path/to/model",
            "type": "lora" | "full",
            "checkpoints": ["checkpoint-50", "checkpoint-90"],
            "latest_checkpoint": "checkpoint-90",
            "loss_image": "/path/to/training_loss.png",
            "train_results": {"train_loss": 0.5, ...},
            "trainer_state": {"global_step": 90, ...},
            "created_time": 1704441600.0
        }
        """
        models = []
        
        if not self.saves_path.exists():
            return models
        
        # 递归扫描所有可能的模型目录
        for model_dir in self._find_model_dirs(self.saves_path):
            model_info = self._parse_model_dir(model_dir)
            if model_info:
                models.append(model_info)
        
        # 按创建时间排序（最新的在前）
        models.sort(key=lambda x: x.get("created_time", 0), reverse=True)
        return models
    
    def _find_model_dirs(self, base_path: Path) -> List[Path]:
        """递归查找包含 adapter_config.json 的目录"""
        model_dirs = []
        
        for item in base_path.rglob("adapter_config.json"):
            model_dirs.append(item.parent)
        
        # 也检查 full SFT 模型（包含 config.json 但没有 adapter_config.json）
        for item in base_path.rglob("trainer_state.json"):
            if not (item.parent / "adapter_config.json").exists():
                model_dirs.append(item.parent)
        
        return list(set(model_dirs))
    
    def _parse_model_dir(self, model_dir: Path) -> Optional[Dict]:
        """解析单个模型目录，提取训练产物信息"""
        try:
            # 基本信息
            model_info = {
                "name": model_dir.name,
                "path": str(model_dir),
                "type": "lora" if (model_dir / "adapter_config.json").exists() else "full",
                "created_time": model_dir.stat().st_mtime
            }
            
            # 检查点
            checkpoints = sorted(model_dir.glob("checkpoint-*"), 
                               key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0)
            model_info["checkpoints"] = [cp.name for cp in checkpoints]
            model_info["latest_checkpoint"] = checkpoints[-1].name if checkpoints else None
            
            # Loss 曲线图
            loss_image = model_dir / "training_loss.png"
            model_info["loss_image"] = str(loss_image) if loss_image.exists() else None
            
            # 训练结果
            train_results = model_dir / "train_results.json"
            if train_results.exists():
                model_info["train_results"] = self._load_json(train_results)
            
            # 完整结果
            all_results = model_dir / "all_results.json"
            if all_results.exists():
                model_info["all_results"] = self._load_json(all_results)
            
            # Trainer 状态（包含 global_step 等）
            trainer_state = model_dir / "trainer_state.json"
            if trainer_state.exists():
                state = self._load_json(trainer_state)
                model_info["global_step"] = state.get("global_step")
                model_info["best_metric"] = state.get("best_metric")
                model_info["log_history"] = state.get("log_history", [])[-10:]  # 最后10条日志
            
            # Trainer 日志（用于绘制曲线）
            trainer_log = model_dir / "trainer_log.jsonl"
            if trainer_log.exists():
                model_info["trainer_log"] = str(trainer_log)
            
            return model_info
        except Exception as e:
            return {"name": model_dir.name, "path": str(model_dir), "error": str(e)}
    
    def _load_json(self, file_path: Path) -> Optional[Dict]:
        """安全加载 JSON 文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    
    def get_model_details(self, model_path: str) -> Dict:
        """获取单个模型的详细信息"""
        model_dir = Path(model_path)
        if not model_dir.exists():
            return {"error": f"模型目录不存在: {model_path}"}
        return self._parse_model_dir(model_dir) or {"error": "解析失败"}
    
    def get_training_log(self, model_path: str) -> List[Dict]:
        """获取训练日志（用于绘制 loss 曲线）"""
        model_dir = Path(model_path)
        log_file = model_dir / "trainer_log.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    logs.append(json.loads(line.strip()))
        except Exception:
            pass
        
        return logs
    
    def run_training(
        self,
        task_id: str,
        config_name: str,
        version_tag: Optional[str] = None
    ) -> Dict:
        """
        执行训练任务
        调用训练脚本
        """
        # 查找配置脚本
        script_path = None
        for method in ["lora", "full"]:
            candidate = self.scripts_path / method / f"{config_name}.sh"
            if candidate.exists():
                script_path = candidate
                break
        
        if not script_path:
            return {"success": False, "error": f"配置不存在: {config_name}"}
        
        # 设置输出目录
        output_dir = self.saves_path / f"{config_name}_{version_tag or task_id[:8]}"
        
        # 环境变量
        env = os.environ.copy()
        env["OUTPUT_DIR"] = str(output_dir)
        
        try:
            # 启动训练进程
            process = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=str(self.project_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 保存进程引用
            self._running_processes[task_id] = process
            
            # 等待完成
            stdout, stderr = process.communicate()
            
            # 移除进程引用
            self._running_processes.pop(task_id, None)
            
            return {
                "success": process.returncode == 0,
                "output_dir": str(output_dir),
                "stdout": stdout[-5000:] if stdout else "",  # 最后 5000 字符
                "stderr": stderr[-2000:] if stderr else "",
                "return_code": process.returncode
            }
        except Exception as e:
            self._running_processes.pop(task_id, None)
            return {"success": False, "error": str(e)}
    
    def get_training_progress(self, task_id: str) -> Dict:
        """获取训练进度"""
        process = self._running_processes.get(task_id)
        
        if not process:
            return {"status": "not_running"}
        
        # 检查进程状态
        poll = process.poll()
        if poll is None:
            return {"status": "running"}
        else:
            return {"status": "completed", "return_code": poll}
    
    def stop_training(self, task_id: str) -> bool:
        """停止训练任务"""
        process = self._running_processes.get(task_id)
        
        if not process:
            return False
        
        try:
            # 发送终止信号
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
            self._running_processes.pop(task_id, None)
            return True
        except Exception:
            # 强制终止
            process.kill()
            self._running_processes.pop(task_id, None)
            return True
