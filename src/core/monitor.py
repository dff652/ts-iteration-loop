"""
训练进度监控
提供实时进度查询和 WebSocket 推送
"""
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from src.adapters.chatts_training import ChatTSTrainingAdapter


class TrainingMonitor:
    """训练进度监控器"""
    
    def __init__(self):
        self.adapter = ChatTSTrainingAdapter()
    
    def get_training_progress(self, output_dir: str) -> Dict:
        """
        获取训练进度
        通过解析 trainer_log.jsonl 获取实时进度
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return {"status": "not_started", "progress": 0}
        
        # 检查 trainer_log.jsonl
        log_file = output_path / "trainer_log.jsonl"
        if not log_file.exists():
            return {"status": "initializing", "progress": 0}
        
        # 解析日志获取最新进度
        logs = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    logs.append(json.loads(line.strip()))
        except Exception:
            return {"status": "running", "progress": 0}
        
        if not logs:
            return {"status": "running", "progress": 0}
        
        # 获取最新日志
        latest = logs[-1]
        
        # 计算进度
        current_step = latest.get("current_steps", 0)
        total_steps = latest.get("total_steps", 1)
        progress = min(100, int((current_step / total_steps) * 100)) if total_steps > 0 else 0
        
        return {
            "status": "running",
            "progress": progress,
            "current_step": current_step,
            "total_steps": total_steps,
            "loss": latest.get("loss"),
            "learning_rate": latest.get("learning_rate"),
            "epoch": latest.get("epoch"),
            "elapsed_time": latest.get("elapsed_time"),
            "remaining_time": latest.get("remaining_time"),
            "log_count": len(logs)
        }
    
    def get_loss_history(self, output_dir: str) -> List[Dict]:
        """
        获取 loss 历史记录（用于绘制曲线）
        """
        output_path = Path(output_dir)
        log_file = output_path / "trainer_log.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if "loss" in entry:
                        logs.append({
                            "step": entry.get("current_steps", 0),
                            "loss": entry.get("loss"),
                            "epoch": entry.get("epoch")
                        })
        except Exception:
            pass
        
        return logs
    
    def is_training_complete(self, output_dir: str) -> bool:
        """检查训练是否完成"""
        output_path = Path(output_dir)
        
        # 检查是否存在最终模型文件
        adapter_config = output_path / "adapter_config.json"
        train_results = output_path / "train_results.json"
        
        return adapter_config.exists() and train_results.exists()
    
    def get_final_results(self, output_dir: str) -> Optional[Dict]:
        """获取最终训练结果"""
        output_path = Path(output_dir)
        
        train_results = output_path / "train_results.json"
        if not train_results.exists():
            return None
        
        try:
            with open(train_results, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


# 全局监控器实例
training_monitor = TrainingMonitor()
