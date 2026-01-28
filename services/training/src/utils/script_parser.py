import os
import re
import shlex
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from configs.settings import settings

class ScriptParser:
    """
    解析 Shell 脚本并转换为 LLaMA-Factory WebUI 兼容的配置 (YAML)
    """

    def __init__(self, training_root: str = settings.CHATTS_TRAINING_PATH):
        self.training_root = Path(training_root)
        self.dataset_info_path = self.training_root / "data" / "dataset_info.json"
        
        # 加载 DataSet Info 用于验证
        self.known_datasets = set()
        if self.dataset_info_path.exists():
            try:
                with open(self.dataset_info_path, 'r') as f:
                    import json
                    info = json.load(f)
                    self.known_datasets = set(info.keys())
            except:
                pass

    def parse_script(self, script_path: str) -> Dict[str, Any]:
        """
        解析 Shell 脚本提取训练参数
        """
        p = Path(script_path)
        if not p.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        content = p.read_text(encoding="utf-8")
        
        # 1. 提取变量定义 (e.g. MODEL_PATH="...")
        variables = self._extract_variables(content)
        
        # 2. 提取命令行参数
        # 查找包含 'train.py' 或 'llamafactory-cli train' 的行
        cmd_lines = []
        capture = False
        for line in content.splitlines():
            line = line.strip()
            # 简单状态机捕获多行命令
            if ("torchrun" in line or "deepspeed" in line or "llamafactory-cli" in line or "python" in line) and not line.startswith("#"):
                capture = True
            
            if capture:
                # 去除行尾反斜杠
                clean_line = line.rstrip('\\').strip()
                if clean_line:
                    cmd_lines.append(clean_line)
                if not line.endswith('\\'):
                    capture = False
                    # 只解析第一个找到的训练命令
                    if cmd_lines: break
        
        full_cmd = " ".join(cmd_lines)
        
        # 3. 解析参数
        args = self._parse_cmd_args(full_cmd, variables)
        
        # 4. 后处理与映射
        config = self._map_to_webui_config(args, script_name=p.stem)
        
        return config

    def _extract_variables(self, content: str) -> Dict[str, str]:
        """提取 Shell 变量定义"""
        vars = {}
        # 匹配 key="value" 或 key=value
        pattern = re.compile(r'^([A-Z_][A-Z0-9_]*)=["\']?(.*?)["\']?$', re.MULTILINE)
        for match in pattern.finditer(content):
            key, val = match.groups()
            # 简单的变量替换 (e.g. ${MODEL_PATH})
            for k, v in vars.items():
                val = val.replace(f"${{{k}}}", v).replace(f"${k}", v)
            vars[key] = val
        return vars

    def _parse_cmd_args(self, cmd: str, variables: Dict[str, str]) -> Dict[str, str]:
        """使用 shlex 解析命令行参数"""
        # 替换变量引用
        for k, v in variables.items():
            cmd = cmd.replace(f"${{{k}}}", v).replace(f"${k}", v)
            
        parts = shlex.split(cmd)
        args = {}
        
        i = 0
        while i < len(parts):
            item = parts[i]
            if item.startswith("--"):
                key = item[2:]
                # 检查下一个是否是值
                if i + 1 < len(parts) and not parts[i+1].startswith("--"):
                    val = parts[i+1]
                    args[key] = val
                    i += 2
                else:
                    # bool flag
                    args[key] = "True"
                    i += 1
            else:
                i += 1
        return args

    def _map_to_webui_config(self, args: Dict[str, str], script_name: str) -> Dict[str, Any]:
        """映射提取的参数到 WebUI Config 格式"""
        config = {}
        
        # === 基础映射表 (Arg Name -> UI Key) ===
        # LLaMA-Factory WebUI 使用的 Key 可能与 CLI 参数名不同
        # 参考 src/llamafactory/webui/components/train.py 的 _parse_train_args
        
        # 1. 路径处理
        if "model_name_or_path" in args:
            raw_path = args["model_name_or_path"]
            # 自动修复相对路径
            abs_path = self._resolve_path(raw_path)
            config["top.model_path"] = str(abs_path)
            # 尝试推断 model_name (仅用于 UI 显示)
            config["top.model_name"] = "Custom Path" 
        
        if "output_dir" in args:
            # 仅保留目录名，因为 WebUI 会基于 saves/前缀自动拼接
            # 但这里我们存完整路径更安全吗？ WebUI 的 get_save_dir 会强制拼接...
            # 这里我们让 output_dir 只保留最后一级目录名
            p = Path(args["output_dir"])
            config["train.output_dir"] = p.name
            
        # 2. 数据集处理
        if "dataset" in args:
            ds_name = args["dataset"]
            if ds_name not in self.known_datasets:
                # 模糊匹配修复逻辑
                match = self._find_closest_dataset(ds_name)
                if match:
                    ds_name = match # 自动修复
            config["train.dataset"] = [ds_name]

        # 3. DeepSpeed
        if "deepspeed" in args:
            ds_path = args["deepspeed"]
            # 推断 stage
            if "ds_config_3" in ds_path or "stage3" in ds_path or "z3" in ds_path:
                config["train.ds_stage"] = "3"
            elif "ds_config_2" in ds_path or "stage2" in ds_path or "z2" in ds_path:
                config["train.ds_stage"] = "2"
            
            if "offload" in ds_path:
                config["train.ds_offload"] = True

        # 4. 直接参数映射
        direct_map = {
            "learning_rate": "train.learning_rate",
            "num_train_epochs": "train.num_train_epochs",
            "per_device_train_batch_size": "train.batch_size",
            "gradient_accumulation_steps": "train.gradient_accumulation_steps",
            "lr_scheduler_type": "train.lr_scheduler_type",
            "logging_steps": "train.logging_steps",
            "save_steps": "train.save_steps",
            "warmup_ratio": "train.warmup_ratio", # WebUI 可能用 steps? 需检查 conversion
            "warmup_steps": "train.warmup_steps",
            "lora_rank": "train.lora_rank",
            "lora_alpha": "train.lora_alpha",
            "lora_dropout": "train.lora_dropout",
            "lora_target": "train.lora_target",
            "template": "top.template",
            "finetuning_type": "top.finetuning_type",
            "cutoff_len": "train.cutoff_len",
            "fp16": "train.compute_type", # special handling below
            "bf16": "train.compute_type",
        }

        for arg_k, ui_k in direct_map.items():
            if arg_k in args:
                config[ui_k] = args[arg_k]
                
        # 特殊映射修正
        if "fp16" in args and args["fp16"] == "True":
            config["train.compute_type"] = "fp16"
        if "bf16" in args and args["bf16"] == "True":
            config["train.compute_type"] = "bf16"
            
        # 语言与默认值
        config["top.lang"] = "zh"
        
        return config

    def _resolve_path(self, path_str: str) -> Path:
        """解析路径，处理相对路径"""
        p = Path(path_str)
        if p.is_absolute():
            return p
        # 尝试相对于 training_root
        abs_p = (self.training_root / p).resolve()
        if abs_p.exists():
            return abs_p
        # 找不到则返回原始拼装结果，让用户自己由 UI 处理
        return abs_p

    def _find_closest_dataset(self, name: str) -> Optional[str]:
        """在已知数据集中查找最相似的"""
        candidates = [d for d in self.known_datasets if name in d]
        if candidates:
            return candidates[0]
        return None

    def save_config(self, config: Dict[str, Any], output_name: str):
        """保存为 YAML"""
        config_dir = self.training_root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = config_dir / f"{output_name}.yaml"
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        return str(output_path)
