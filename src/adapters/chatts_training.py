"""
ChatTS-Training 项目适配器
封装模型微调功能
"""
import os
import json
import subprocess
import signal
import sys
import threading
from pathlib import Path
from typing import List, Dict, Optional

from configs.settings import settings


class ChatTSTrainingAdapter:
    """ChatTS-Training 项目适配器"""

    def __init__(self, model_family: str = "chatts"):
        self.model_family = model_family
        self.project_path = Path(settings.CHATTS_TRAINING_PATH)
        self.scripts_root = self.project_path / "scripts"
        self.scripts_path = self.scripts_root / model_family
        self.saves_path = self.project_path / "saves" / model_family
        self.data_path = Path(self._get_data_dir())
        
        # 运行中的训练进程
        self._running_processes: Dict[str, subprocess.Popen] = {}
    
    def list_configs(self) -> List[Dict]:
        """列出可用的训练配置"""
        configs = []

        # 扫描 lora 目录
        lora_scripts = self._get_scripts_dir("lora")
        if lora_scripts.exists():
            for f in lora_scripts.glob("*.sh"):
                configs.append({
                    "name": f.stem,
                    "path": str(f),
                    "method": "lora",
                    "description": f"LoRA 微调 ({self.model_family}): {f.stem}"
                })

        # 扫描 full 目录
        full_scripts = self._get_scripts_dir("full")
        if full_scripts.exists():
            for f in full_scripts.glob("*.sh"):
                configs.append({
                    "name": f.stem,
                    "path": str(f),
                    "method": "full",
                    "description": f"Full SFT ({self.model_family}): {f.stem}"
                })
        
        return configs
    
    def list_models(self) -> List[Dict]:
        """
        列出已训练的模型，包含训练产物信息
        """
        models = []
        
        # 扫描全局 saves 目录
        saves_root = self.project_path / "saves"
        if not saves_root.exists():
            return models
        
        # 递归扫描所有可能的模型目录
        # 限制搜索深度以提高性能，或者直接查找 adapter_config.json
        # 这里复用 _find_model_dirs，它使用 rglob，能找到 saves/chatts-8b/lora/xxx 中的模型
        for model_dir in self._find_model_dirs(saves_root):
            model_info = self._parse_model_dir(model_dir)
            if model_info:
                # 简单的过滤：只显示当前 model_family 或相关的模型
                # 如果 model_family 是 chatts，显示 path 中包含 chatts 的
                # 或者不做过滤，全部显示
                models.append(model_info)
        
        # 按创建时间排序（最新的在前）
        models.sort(key=lambda x: x.get("created_time", 0), reverse=True)
        return models
    
    def _find_model_dirs(self, base_path: Path) -> List[Path]:
        """递归查找包含 adapter_config.json 的目录"""
        model_dirs = []
        
        if not base_path.exists():
            return []

        # 1. 查找 LoRA 模型 (含有 adapter_config.json)
        for item in base_path.rglob("adapter_config.json"):
            model_dirs.append(item.parent)
        
        # 2. 查找 Full SFT 模型 (含有 trainer_state.json 但没有 adapter_config.json)
        for item in base_path.rglob("trainer_state.json"):
            parent = item.parent
            if parent not in model_dirs:
                 # 再次确认没有 adapter_config.json (避免重复)
                 if not (parent / "adapter_config.json").exists():
                     model_dirs.append(parent)
        
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
        """获取所有可用数据集名称"""
        try:
            # 1. 同步数据集
            self._sync_datasets()

            # 2. 读取最新的 info
            info_path = self.data_path / "dataset_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    data = json.load(f)
                return sorted(list(data.keys()))
        except:
            pass
        return []

    def _sync_datasets(self):
        """按模型类型同步数据集，并写入对应的 dataset_info.json"""
        try:
            if self.model_family == "qwen":
                training_dir = Path(settings.DATA_TRAINING_QWEN_DIR)
                info_path = training_dir / "dataset_info.json"
                new_info = self._build_sharegpt_dataset_info(training_dir)
            else:
                training_dir = Path(settings.DATA_TRAINING_CHATTS_DIR)
                info_path = training_dir / "dataset_info.json"
                new_info = self._build_chatts_dataset_info(training_dir)

            if not training_dir.exists():
                return

            # 覆盖写入
            with open(info_path, 'w') as f:
                json.dump(new_info, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Sync datasets failed: {e}")

    def _build_chatts_dataset_info(self, training_dir: Path) -> Dict[str, Dict]:
        """构建 ChatTS 数据集信息（支持文件/目录）"""
        new_info: Dict[str, Dict] = {}
        if not training_dir.exists():
            return new_info

        for item in sorted(training_dir.iterdir()):
            if item.name.startswith("dataset_info") or item.name.startswith(".") or item.name.startswith("_"):
                continue
            if item.is_file() and item.suffix in {".json", ".jsonl"}:
                name = item.stem
                new_info[name] = {
                    "file_name": item.name,
                    "columns": {
                        "prompt": "input",
                        "response": "output",
                        "timeseries": "timeseries"
                    }
                }
            elif item.is_dir():
                dataset_file = self._pick_dataset_file(item)
                name = item.name
                rel_path = item.name if dataset_file is None else f"{item.name}/{dataset_file.name}"
                new_info[name] = {
                    "file_name": rel_path,
                    "columns": {
                        "prompt": "input",
                        "response": "output",
                        "timeseries": "timeseries"
                    }
                }
        return new_info

    def _pick_dataset_file(self, dataset_dir: Path) -> Optional[Path]:
        """为目录型数据集选择最合适的文件"""
        candidates = sorted([p for p in dataset_dir.glob("*.json*") if p.is_file()])
        if not candidates:
            return None

        # 优先 train.jsonl / train.json
        for name in ["train.jsonl", "train.json"]:
            for c in candidates:
                if c.name == name:
                    return c

        # 其次选择包含 train 的文件
        for c in candidates:
            if "train" in c.name:
                return c

        # 如果只有一个文件就用它，否则交给目录加载
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _build_sharegpt_dataset_info(self, training_dir: Path) -> Dict[str, Dict]:
        """构建 ShareGPT / Alpaca 类数据集信息"""
        new_info: Dict[str, Dict] = {}
        if not training_dir.exists():
            return new_info

        for f_path in training_dir.glob("*.json*"):
            name = f_path.stem
            if name.startswith("dataset_info"):
                continue

            # 简单推断格式
            file_format = "sharegpt"
            messages_key = "conversations"
            role_tag = "from"
            content_tag = "value"
            user_tag = "human"
            assistant_tag = "gpt"
            try:
                with open(f_path, 'r') as f:
                    sample = f.read(1024)
                    if '"conversations":' in sample or '"messages":' in sample:
                        file_format = "sharegpt"
                        if '"messages":' in sample:
                            messages_key = "messages"
                    elif '"instruction":' in sample or '"input":' in sample:
                        file_format = "alpaca"

                    # ShareGPT tags detection (user/assistant vs human/gpt)
                    if '"role":' in sample:
                        role_tag = "role"
                    if '"content":' in sample:
                        content_tag = "content"
                    if f'\"{role_tag}\": \"user\"' in sample or f'\"{role_tag}\":\"user\"' in sample:
                        user_tag = "user"
                        assistant_tag = "assistant"
                    elif f'\"{role_tag}\": \"human\"' in sample or f'\"{role_tag}\":\"human\"' in sample:
                        user_tag = "human"
                        assistant_tag = "gpt"
            except Exception:
                pass

            entry = {
                "file_name": str(f_path),
                "formatting": file_format
            }
            if file_format == "sharegpt":
                entry["columns"] = {
                    "messages": messages_key
                }
                entry["tags"] = {
                    "role_tag": role_tag,
                    "content_tag": content_tag,
                    "user_tag": user_tag,
                    "assistant_tag": assistant_tag
                }

            new_info[name] = entry

        return new_info

    def get_base_models(self) -> List[str]:
        """获取常用底座模型路径 (Quick Start)"""
        if self.model_family == "qwen":
            return ["/home/share/models/Qwen3-VL-8B-TR"]
        return ["/home/share/llm_models/bytedance-research/ChatTS-8B"]

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
            # 使用训练环境 Python 解释器（优先专属环境）
            # 避免使用 sys.executable (可能指向错误环境)
            python_executable = self._get_training_python() or settings.PYTHON_UNIFIED
            if not Path(python_executable).exists():
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
            
            # 使用配置 IP
            ip = settings.WEBUI_HOST
                
            return {"success": True, "message": f"已启动原生 WebUI (PID: {process.pid})", "url": f"http://{ip}:{port}"}
            
        except Exception as e:
            return {"success": False, "error": f"启动失败: {str(e)}"}

    def run_training(
        self,
        task_id: str,
        config_name: str,
        version_tag: Optional[str] = None,
        auto_eval: bool = False,
        eval_truth_dir: Optional[str] = None,
        eval_data_dir: Optional[str] = None,
        eval_dataset_name: Optional[str] = None,
        eval_output_dir: Optional[str] = None,
        eval_device: Optional[str] = None,
        eval_method: Optional[str] = None,
        # Overrides
        override_model_path: Optional[str] = None,
        override_dataset: Optional[str] = None,
        override_learning_rate: Optional[str] = None,
        override_epochs: Optional[float] = None,
        override_batch_size: Optional[int] = None,
        override_lora_rank: Optional[int] = None,
        override_lora_alpha: Optional[int] = None,
        override_grad_accum_steps: Optional[int] = None,
        override_cutoff_len: Optional[int] = None,
        override_precision: Optional[str] = None,
        override_image_max_pixels: Optional[int] = None,
        override_image_min_pixels: Optional[int] = None,
        override_nproc_per_node: Optional[int] = None,
        override_cuda_visible_devices: Optional[str] = None,
        override_extra_args: Optional[str] = None,
        override_logging_steps: Optional[str] = None,
        override_save_steps: Optional[str] = None,
        override_warmup_steps: Optional[str] = None,
        override_warmup_ratio: Optional[str] = None,
        override_lr_scheduler_type: Optional[str] = None,
        override_lora_dropout: Optional[str] = None,
        override_lora_target: Optional[str] = None,
        override_freeze_vision_tower: Optional[str] = None,
        override_freeze_multi_modal_projector: Optional[str] = None,
        override_freeze_trainable_layers: Optional[str] = None,
        override_freeze_trainable_modules: Optional[str] = None,
    ) -> Dict:
        """
        执行训练任务 (支持 Quick Start 参数覆盖)
        """
        try:
            # 查找配置脚本
            script_path = None
            for method in ["lora", "full"]:
                candidate = self._get_scripts_dir(method) / f"{config_name}.sh"
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
            if override_grad_accum_steps:
                script_content = self._replace_or_append_arg(
                    script_content, "--gradient_accumulation_steps", str(override_grad_accum_steps)
                )
            if override_cutoff_len:
                script_content = self._replace_or_append_arg(script_content, "--cutoff_len", str(override_cutoff_len))
            if override_image_max_pixels:
                script_content = self._replace_or_append_arg(
                    script_content, "--image_max_pixels", str(override_image_max_pixels)
                )
            if override_image_min_pixels:
                script_content = self._replace_or_append_arg(
                    script_content, "--image_min_pixels", str(override_image_min_pixels)
                )
            if override_precision:
                script_content = self._apply_precision(script_content, override_precision)
            if override_nproc_per_node:
                script_content = self._replace_arg_val(
                    script_content, "--nproc_per_node", str(override_nproc_per_node)
                )
                script_content = self._replace_arg_val(script_content, "--num_gpus", str(override_nproc_per_node))
            if override_extra_args:
                script_content = self._append_to_train_command(script_content, override_extra_args.strip())
            if override_logging_steps is not None and str(override_logging_steps).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--logging_steps", str(override_logging_steps).strip()
                )
            if override_save_steps is not None and str(override_save_steps).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--save_steps", str(override_save_steps).strip()
                )
            if override_warmup_steps is not None and str(override_warmup_steps).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--warmup_steps", str(override_warmup_steps).strip()
                )
            if override_warmup_ratio is not None and str(override_warmup_ratio).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--warmup_ratio", str(override_warmup_ratio).strip()
                )
            if override_lr_scheduler_type is not None and str(override_lr_scheduler_type).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--lr_scheduler_type", str(override_lr_scheduler_type).strip()
                )
            if override_lora_dropout is not None and str(override_lora_dropout).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--lora_dropout", str(override_lora_dropout).strip()
                )
            if override_lora_target is not None and str(override_lora_target).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--lora_target", f"\"{str(override_lora_target).strip()}\""
                )
            if override_freeze_vision_tower is not None and str(override_freeze_vision_tower).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--freeze_vision_tower", str(override_freeze_vision_tower).strip()
                )
            if override_freeze_multi_modal_projector is not None and str(override_freeze_multi_modal_projector).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--freeze_multi_modal_projector", str(override_freeze_multi_modal_projector).strip()
                )
            if override_freeze_trainable_layers is not None and str(override_freeze_trainable_layers).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--freeze_trainable_layers", str(override_freeze_trainable_layers).strip()
                )
            if override_freeze_trainable_modules is not None and str(override_freeze_trainable_modules).strip() != "":
                script_content = self._replace_or_append_arg(
                    script_content, "--freeze_trainable_modules", str(override_freeze_trainable_modules).strip()
                )

            # 设置输出目录
            output_dir = self.saves_path / f"{config_name}_{version_tag or task_id[:8]}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 生成临时运行脚本
            temp_script_name = f"run_tmp_{task_id[:8]}.sh"
            temp_script_path = self.project_path / temp_script_name
            
            # 强制输出目录隔离（模型族）
            script_content = self._replace_script_var(script_content, "OUTPUT_DIR", str(output_dir))

            with open(temp_script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # 日志文件
            log_file = output_dir / "train.log"
            
            # 环境变量
            env = os.environ.copy()
            env["OUTPUT_DIR"] = str(output_dir)
            env["DATASET_DIR"] = str(self._get_data_dir())
            env["IMAGE_DIR"] = str(self._get_images_dir())
            if override_nproc_per_node:
                env["NPROC_PER_NODE"] = str(override_nproc_per_node)
            if override_cuda_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = override_cuda_visible_devices
            # 强制禁用缓存以实时显示日志
            env["PYTHONUNBUFFERED"] = "1"
            # 禁用 LLaMA-Factory 的严格版本检查 (适配当前环境 Transformers 4.57+)
            env["DISABLE_VERSION_CHECK"] = "1"
            
            # 关键修复: 强制 PATH 指向正确的 Conda 环境
            # 解决 torchrun / python 使用系统默认解释器导致 ModuleNotFoundError 
            python_bin = Path(self._get_training_python() or settings.PYTHON_UNIFIED).parent
            if python_bin.exists():
                env["PATH"] = f"{python_bin}:{env.get('PATH', '')}"
                
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
                "file_handle": f_log,
                "output_dir": str(output_dir),
                "auto_eval": bool(auto_eval),
                "auto_eval_status": "pending" if auto_eval else "disabled",
            }

            if auto_eval:
                self._schedule_auto_eval(
                    task_id=task_id,
                    model_path=str(output_dir),
                    truth_dir=eval_truth_dir or settings.EVAL_GOLDEN_TRUTH_DIR,
                    data_dir=eval_data_dir or settings.EVAL_GOLDEN_DATA_DIR,
                    dataset_name=eval_dataset_name or settings.EVAL_DEFAULT_DATASET_NAME,
                    output_dir=eval_output_dir or settings.EVAL_DEFAULT_OUTPUT_DIR,
                    device=eval_device,
                    method=eval_method,
                )
            
            return {
                "success": True,
                "message": f"训练已启动 (PID: {process.pid})，正在写入日志: {log_file.name}",
                "output_dir": str(output_dir),
                "return_code": 0
            }
        except Exception as e:
             return {"success": False, "error": str(e)}

    def _schedule_auto_eval(
        self,
        task_id: str,
        model_path: str,
        truth_dir: Optional[str],
        data_dir: Optional[str],
        dataset_name: str,
        output_dir: Optional[str],
        device: Optional[str],
        method: Optional[str],
    ) -> None:
        if not truth_dir or not data_dir:
            info = self._running_processes.get(task_id)
            if info is not None:
                info["auto_eval_status"] = "skipped_missing_paths"
            return

        def _watch_and_eval():
            info = self._running_processes.get(task_id)
            if not info:
                return
            process = info.get("process")
            if not process:
                return
            process.wait()
            f_handle = info.get("file_handle")
            if f_handle:
                try:
                    f_handle.close()
                except Exception:
                    pass
            if process.returncode != 0:
                info["auto_eval_status"] = "skipped_failed_training"
                return
            try:
                from src.utils.model_eval import evaluate_model_on_golden
                info["auto_eval_status"] = "running"
                result = evaluate_model_on_golden(
                    model_path=model_path,
                    model_family=self.model_family,
                    truth_dir=truth_dir,
                    data_dir=data_dir,
                    dataset_name=dataset_name,
                    output_dir=output_dir or None,
                    device=device,
                    method=method,
                )
                info["auto_eval_result"] = result
                info["auto_eval_status"] = "completed" if result.get("success") else "failed"
            except Exception as e:
                info["auto_eval_status"] = f"failed: {e}"

        t = threading.Thread(target=_watch_and_eval, daemon=True)
        t.start()

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
            except Exception as e:
                content += f"\n[System Error] 无法读取日志文件: {e}\n"
                
        if status == "completed":
            ret_code = process.returncode
            if ret_code != 0:
                content += f"\n\n[System] Training process terminated with error code {ret_code}.\n"
                
        return {"log": content, "status": status, "offset": new_offset}

    def get_training_progress(self, task_id: str) -> Dict:
        """获取训练进度（基于当前任务状态，简化版）"""
        if task_id not in self._running_processes:
            return {"status": "unknown", "progress": 0}

        info = self._running_processes[task_id]
        process = info["process"]
        status = "running" if process.poll() is None else "completed"
        return {"status": status, "progress": 0}

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

    def _replace_or_append_arg(self, content: str, arg_name: str, new_value: str) -> str:
        """替换参数值，若不存在则追加到训练命令"""
        updated = self._replace_arg_val(content, arg_name, new_value)
        if updated != content:
            return updated
        return self._append_to_train_command(updated, f"{arg_name} {new_value}")

    def _apply_precision(self, content: str, precision: str) -> str:
        """设置精度：bf16/fp16/fp32"""
        import re
        normalized = precision.lower().strip()
        content = re.sub(r"\s--bf16\b", "", content)
        content = re.sub(r"\s--fp16\b", "", content)
        if normalized == "bf16":
            return self._append_to_train_command(content, "--bf16")
        if normalized == "fp16":
            return self._append_to_train_command(content, "--fp16")
        return content

    def _append_to_train_command(self, content: str, arg_line: str) -> str:
        """Append an arg line to the last train command block."""
        if not arg_line:
            return content
        lines = content.splitlines()
        start_idx = None
        for i, line in enumerate(lines):
            if "src/train.py" in line or "${RUN_CMD[@]}" in line:
                start_idx = i
                break
        if start_idx is None:
            return content
        last_arg_idx = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].lstrip().startswith("--"):
                last_arg_idx = j
        end_idx = last_arg_idx if last_arg_idx is not None else start_idx
        if lines[end_idx].rstrip().endswith("\\"):
            lines.insert(end_idx + 1, f"    {arg_line}")
        else:
            lines[end_idx] = lines[end_idx] + " \\"
            lines.insert(end_idx + 1, f"    {arg_line}")
        return "\n".join(lines) + "\n"

    def _get_scripts_dir(self, method: str) -> Path:
        """获取脚本目录（优先模型专属目录）"""
        candidate = self.scripts_root / self.model_family / method
        if candidate.exists():
            return candidate
        # fallback for legacy layout
        return self.scripts_root / method

    def _get_data_dir(self) -> str:
        """获取训练数据目录"""
        if self.model_family == "qwen":
            return settings.DATA_TRAINING_QWEN_DIR
        return settings.DATA_TRAINING_CHATTS_DIR

    def _get_images_dir(self) -> str:
        """获取训练图片目录"""
        if self.model_family == "qwen":
            return settings.DATA_TRAINING_QWEN_IMAGES_DIR
        return ""

    def _get_training_python(self) -> str:
        """获取训练环境的 Python 解释器路径"""
        if self.model_family == "qwen":
            return settings.PYTHON_TRAINING_QWEN
        return settings.PYTHON_TRAINING_CHATTS
