"""
配置加载模块
用于读取和管理项目配置
"""

import json
import os
from typing import Dict, Any

class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 返回默认配置
                return self._get_default_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            Dict[str, Any]: 默认配置字典
        """
        return {
            "server": {
                "listenport": 8010,
                "max_session": 1,
                "transport": "rtcpush",
                "push_url": "http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream"
            },
            "avatar": {
                "default_avatar_id": "avator_1",
                "preload_enabled": False,
                "preload_queue_size": 10,
                "cache_max_size": 5
            },
            "model": {
                "default_model": "musetalk",
                "batch_size": 16,
                "fps": 50
            },
            "tts": {
                "default_engine": "edgetts",
                "default_voice": "zh-CN-YunxiaNeural",
                "server": "http://127.0.0.1:9880",
                "edge_voice_by_lang": {
                    "zh-CN": "zh-CN-YunxiaNeural",
                    "en": "en-US-JennyNeural",
                    "ja": "ja-JP-NanamiNeural",
                    "ko": "ko-KR-SunHiNeural"
                }
            },
            "performance": {
                "level": "balanced",
                "enable_parallel_processing": True,
                "enable_streaming": True
            },
            "gui": {
                "width": 450,
                "height": 450
            },
            "sliding_window": {
                "left": 10,
                "middle": 8,
                "right": 10
            },
            "liveportrait": {
                "expression_enabled": False,
                "mouth_from_liveportrait": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的路径
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔的路径
            value: 配置值
            
        Returns:
            bool: 是否设置成功
        """
        try:
            keys = key.split('.')
            config = self.config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            return True
        except Exception as e:
            print(f"设置配置失败: {e}")
            return False
    
    def save(self) -> bool:
        """
        保存配置到文件
        
        Returns:
            bool: 是否保存成功
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def reload(self) -> bool:
        """
        重新加载配置文件
        
        Returns:
            bool: 是否加载成功
        """
        try:
            self.config = self._load_config()
            return True
        except Exception as e:
            print(f"重新加载配置失败: {e}")
            return False


# 全局配置实例
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """
    获取全局配置实例
    
    Returns:
        ConfigLoader: 配置加载器实例
    """
    return config


def get_config_value(key: str, default: Any = None) -> Any:
    """
    获取配置值的便捷函数
    
    Args:
        key: 配置键
        default: 默认值
        
    Returns:
        Any: 配置值
    """
    return config.get(key, default)


def set_config_value(key: str, value: Any) -> bool:
    """
    设置配置值的便捷函数
    
    Args:
        key: 配置键
        value: 配置值
        
    Returns:
        bool: 是否设置成功
    """
    return config.set(key, value)


def save_config() -> bool:
    """
    保存配置的便捷函数
    
    Returns:
        bool: 是否保存成功
    """
    return config.save()


def reload_config() -> bool:
    """
    重新加载配置的便捷函数
    
    Returns:
        bool: 是否加载成功
    """
    return config.reload()