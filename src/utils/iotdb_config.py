"""
IoTDB 配置加载器
提供统一的 IoTDB 连接配置管理
"""
import os
import json
from typing import Dict, Any, Optional

# 默认配置文件路径
_CONFIG_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'configs', 'iotdb_config.json'),
    os.path.join(os.path.dirname(__file__), 'configs', 'iotdb_config.json'),
    '/home/douff/ts/ts-iteration-loop/configs/iotdb_config.json',
]

_cached_config: Optional[Dict[str, Any]] = None


def load_iotdb_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载 IoTDB 配置
    
    Args:
        config_path: 可选的配置文件路径，如果不指定则自动搜索
        
    Returns:
        配置字典，包含 host, port, user, password 等
    """
    global _cached_config
    
    if _cached_config is not None and config_path is None:
        return _cached_config
    
    # 搜索配置文件
    search_paths = [config_path] if config_path else _CONFIG_PATHS
    
    for path in search_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    _cached_config = config
                    return config
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
                continue
    
    # 返回默认配置
    default_config = {
        "host": "192.168.199.185",
        "port": "6667",
        "user": "root",
        "password": "root",
        "default_path": "root.supcon.nb.whlj.LJSJ",
        "fetch_size": 2000000,
        "limit": 1000000000
    }
    _cached_config = default_config
    return default_config


def get_iotdb_connection_params() -> Dict[str, str]:
    """
    获取 IoTDB 连接参数
    
    Returns:
        只包含连接所需的参数: host, port, user, password
    """
    config = load_iotdb_config()
    return {
        "host": config.get("host", "192.168.199.185"),
        "port": config.get("port", "6667"),
        "user": config.get("user", "root"),
        "password": config.get("password", "root"),
    }
