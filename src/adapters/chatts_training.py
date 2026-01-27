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
    
    def get_dataset_list(self) -> List[str]:
        """获取所有可用数据集名称 (仅显示 DATA_TRAINING_DIR 下的自动注册数据集)"""
        try:
            # 1. 同步外部数据集 (覆盖模式)
            self._sync_external_datasets()
            
            # 2. 读取最新的 info
            info_path = self.data_path / "dataset_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    data = json.load(f)
                return sorted(list(data.keys()))
        except:
            pass
        return []

    def _sync_external_datasets(self):
        """扫描标准训练数据目录，重写 dataset_info.json (仅包含外部数据)"""
        try:
            training_dir = Path(settings.DATA_TRAINING_DIR)
            if not training_dir.exists():
                return
            
            info_path = self.data_path / "dataset_info.json"
            
            # 构建新的 info 字典
            new_info = {}
            
            # 扫描 json/jsonl 文件
            for f_path in training_dir.glob("*.json*"):
                name = f_path.stem
                
                # 简单推断格式
                file_format = "sharegpt" 
                try:
                     with open(f_path, 'r') as f:
                         sample = f.read(1024)
                         if '"conversations":' in sample or '"messages":' in sample:
                             file_format = "sharegpt"
                         elif '"instruction":' in sample or '"input":' in sample:
                             file_format = "alpaca"
                except:
                    pass
                    
                entry = {
                    "file_name": str(f_path),
                    "formatting": file_format
                }
                if file_format == "sharegpt":
                    entry["columns"] = {
                        "messages": "conversations"  
                    }
                
                new_info[name] = entry
            
            # 覆盖写入
            with open(info_path, 'w') as f:
                json.dump(new_info, f, indent=4, ensure_ascii=False)
                    
        except Exception as e:
            print(f"Sync datasets failed: {e}")

    def get_base_models(self) -> List[str]:
        """获取常用底座模型路径 (Quick Start)"""
        return [
            "/home/data1/llm_models/bytedance-research/ChatTS-8B",
            "/home/share/models/Qwen3-VL-8B-TR"
        ]

    def start_native_webui(self, port: int = 7861) -> Dict:
        """
        启动原生 LLaMA-Factory WebUI
        """
        task_id = "native_webui"
        
        # 检查是否已运行
        if task_id in self._running_processes:
            proc = self._running_processes[task_id]
            if proc.poll() is None:
                 # 尝试获取本机 IP
                 import socket
                 try:
                     ip = socket.gethostbyname(socket.gethostname())
                     if ip.startswith("127"): ip = "localhost"
                 except:
                     ip = "localhost"
                 return {"success": True, "message": "服务已在运行中", "url": f"http://{ip}:{port}"}
            else:
                self._running_processes.pop(task_id)

        try:
            env = os.environ.copy()
            env["GRADIO_SERVER_PORT"] = str(port)
            env["GRADIO_IPV6"] = "0" 
            # 允许外部访问
            env["GRADIO_SERVER_NAME"] = "0.0.0.0"
            # 禁用 LLaMA-Factory 的严格版本检查 (适配当前环境 Transformers 4.57+)
            env["DISABLE_VERSION_CHECK"] = "1" 
            
            # 设置 PYTHONPATH 以便能导入 llamafactory
            # project_path 指向 services/training
            # 源码在 services/training/src
            src_path = self.project_path / "src"
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{str(src_path)}:{current_pythonpath}"
            
            # 日志文件
            log_file = self.project_path / "webui.log"
            
            # 使用当前 Python 解释器直接运行 webui.py
            # 使用项目专属环境的 Python 解释器
            # 避免使用 sys.executable (可能指向错误环境)
            python_executable = "/opt/conda_envs/douff/ts-iteration-loop/bin/python"
            if not Path(python_executable).exists():
                # Fallback to current if dedicated env python missing
                python_executable = sys.executable

            cmd = [python_executable, str(src_path / "webui.py")]
            
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_path),
                env=env,
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
            )
            
            self._running_processes[task_id] = process
            
            # 等待 3 秒检查进程是否存活 (捕获启动初期错误)
            import time
            time.sleep(3)
            
            if process.poll() is not None:
                # 进程已退出，说明启动失败
                error_msg = f"WebUI 进程意外退出 (Exit Code: {process.returncode})\n"
                try:
                    with open(log_file, 'r') as f:
                        logs = f.read()
                        # 取最后 20 行日志
                        tail_logs = "\n".join(logs.splitlines()[-20:])
                        error_msg += f"--- 错误日志 (最后 20 行) ---\n{tail_logs}"
                except:
                    error_msg += "无法读取日志文件"
                
                print(f"[ERROR] {error_msg}") # 打印到后台控制台
                self._running_processes.pop(task_id)
                return {"success": False, "message": "启动失败，请查看日志", "error": error_msg}
            
            # 使用固定 IP，停止猜测
            ip = "192.168.199.126"
                
            return {"success": True, "message": f"已启动原生 WebUI (PID: {process.pid})", "url": f"http://{ip}:{port}"}
            
        except Exception as e:
            return {"success": False, "error": f"启动失败: {str(e)}"}

    def run_training(
        self,
        task_id: str,
        config_name: str,
        version_tag: Optional[str] = None,
        # Overrides
        override_model_path: Optional[str] = None,
        override_dataset: Optional[str] = None,
        override_learning_rate: Optional[str] = None,
        override_epochs: Optional[float] = None,
        override_batch_size: Optional[int] = None,
        override_lora_rank: Optional[int] = None,
        override_lora_alpha: Optional[int] = None,
    ) -> Dict:
        """
        执行训练任务 (支持 Quick Start 参数覆盖)
        """
        try:
            # 查找配置脚本
            script_path = None
            for method in ["lora", "full"]:
                candidate = self.scripts_path / method / f"{config_name}.sh"
                if candidate.exists():
                    script_path = candidate
                    break
            
            if not script_path:
                return {"success": False, "error": f"配置不存在: {config_name}"}
            
            # 读取原始脚本
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()

            # 如果有覆盖参数，进行替换
            if override_model_path:
                 script_content = self._replace_script_var(script_content, "MODEL_PATH", override_model_path)
            if override_dataset:
                 script_content = self._replace_script_var(script_content, "DATASET", override_dataset)
            
            # 命令行参数替换
            if override_learning_rate:
                script_content = self._replace_arg_val(script_content, "--learning_rate", str(override_learning_rate))
            if override_epochs:
                script_content = self._replace_arg_val(script_content, "--num_train_epochs", str(override_epochs))
            if override_batch_size:
                script_content = self._replace_arg_val(script_content, "--per_device_train_batch_size", str(override_batch_size))
            if override_lora_rank:
                script_content = self._replace_arg_val(script_content, "--lora_rank", str(override_lora_rank))
            if override_lora_alpha:
                script_content = self._replace_arg_val(script_content, "--lora_alpha", str(override_lora_alpha))

            # 生成临时运行脚本
            temp_script_name = f"run_tmp_{task_id[:8]}.sh"
            temp_script_path = self.project_path / temp_script_name
            
            with open(temp_script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # 设置输出目录
            output_dir = self.saves_path / f"{config_name}_{version_tag or task_id[:8]}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 日志文件
            log_file = output_dir / "train.log"
            
            # 环境变量
            env = os.environ.copy()
            env["OUTPUT_DIR"] = str(output_dir)
            # 强制禁用缓存以实时显示日志
            env["PYTHONUNBUFFERED"] = "1"
            # 禁用 LLaMA-Factory 的严格版本检查 (适配当前环境 Transformers 4.57+)
            env["DISABLE_VERSION_CHECK"] = "1"
            
            # 关键修复: 强制 PATH 指向正确的 Conda 环境
            # 解决 torchrun / python 使用系统默认解释器导致 ModuleNotFoundError 
            conda_bin = "/opt/conda_envs/douff/ts-iteration-loop/bin"
            if os.path.exists(conda_bin):
                env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
                
            # 启动训练进程 (输出重定向到文件)
            cmd = ["bash", str(temp_script_path)]
            
            f_log = open(log_file, "w")
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_path),
                env=env,
                stdout=f_log,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # 保存进程引用和日志文件句柄
            self._running_processes[task_id] = {
                "process": process,
                "log_file": str(log_file),
                "file_handle": f_log
            }
            
            return {
                "success": True,
                "message": f"训练已启动 (PID: {process.pid})，正在写入日志: {log_file.name}",
                "output_dir": str(output_dir),
                "return_code": 0
            }
        except Exception as e:
             return {"success": False, "error": str(e)}

    def get_training_log(self, task_id: str, offset: int = 0) -> Dict:
        """获取增量日志"""
        if task_id not in self._running_processes:
            return {"log": "", "status": "stopped", "offset": offset}
            
        info = self._running_processes[task_id]
        log_path = Path(info["log_file"])
        process = info["process"]
        
        status = "running"
        if process.poll() is not None:
            status = "completed"
        
        content = ""
        new_offset = offset
        
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(offset)
                    chunk = f.read()
                    if chunk:
                        content = chunk
                        new_offset = f.tell()
            except:
                pass
                
        return {"log": content, "status": status, "offset": new_offset}

    def stop_training(self, task_id: str) -> Dict:
        """停止训练任务"""
        if task_id not in self._running_processes:
            return {"success": False, "message": "任务未运行"}
            
        info = self._running_processes[task_id]
        process = info["process"]
        f_handle = info["file_handle"]
        
        try:
            # 尝试优雅终止
            import signal
            os.kill(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.kill(process.pid, signal.SIGKILL)
                
            # 关闭日志句柄
            try:
                f_handle.close()
            except:
                pass
                
            self._running_processes.pop(task_id)
            return {"success": True, "message": "训练任务已终止"}
        except Exception as e:
            return {"success": False, "error": f"终止失败: {str(e)}"}

    def _replace_script_var(self, content: str, var_name: str, new_value: str) -> str:
        """替换 Shell 变量定义 MODEL_PATH="..." """
        import re
        pattern = re.compile(f'^{var_name}=["\']?.*?["\']?$', re.MULTILINE)
        if pattern.search(content):
            return pattern.sub(f'{var_name}="{new_value}"', content)
        return content

    def _replace_arg_val(self, content: str, arg_name: str, new_value: str) -> str:
        """替换命令行参数 --arg val """
        import re
        pattern = re.compile(f'{arg_name}\s+["\']?.*?["\']?(\s|\\\\|$)', re.MULTILINE)
        if pattern.search(content):
            return pattern.sub(f'{arg_name} {new_value}\\1', content)
        return content
