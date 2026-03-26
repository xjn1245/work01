"""
性能优化配置文件
提供不同级别的性能优化选项
"""

import os
from typing import Dict, Any

class PerformanceConfig:
    """性能优化配置类"""

    def __init__(self, level: str = "balanced"):
        """
        初始化性能配置

        Args:
            level: 优化级别，可选 "speed"（速度优先）、"balanced"（平衡）、"quality"（质量优先）
        """
        self.level = level.lower()
        self.configs = self._get_configs()

    def _get_configs(self) -> Dict[str, Any]:
        """获取对应级别的配置"""

        if self.level == "speed":
            # 速度优先配置
            return {
                "tts_engine": "azuretts",  # 使用更快的TTS引擎
                "llm_model": "qwen-turbo",  # 使用更快的LLM模型
                "max_response_length": 300,  # 限制回答长度
                "fps": 20,  # 降低帧率
                "video_quality": "low",  # 低视频质量
                "enable_cache": True,  # 启用缓存
                "parallel_processing": True,  # 并行处理
                "stream_tts": True,  # 流式TTS
            }
        elif self.level == "quality":
            # 质量优先配置 - 使用Azure TTS替代有问题的EdgeTTS
            return {
                "tts_engine": "azuretts",  # 使用Azure TTS替代EdgeTTS
                "llm_model": "qwen-plus",  # 使用更强的LLM模型
                "max_response_length": 500,  # 允许更长的回答
                "fps": 30,  # 标准帧率
                "video_quality": "high",  # 高视频质量
                "enable_cache": False,  # 禁用缓存
                "parallel_processing": False,  # 串行处理
                "stream_tts": False,  # 完整TTS
            }
        else:
            # 平衡配置（默认）
            return {
                "tts_engine": "azuretts",  # 平衡的TTS引擎
                "llm_model": "qwen-turbo",  # 平衡的LLM模型
                "max_response_length": 350,  # 适中的回答长度
                "fps": 35,  # 平衡的帧率
                "video_quality": "medium",  # 中等视频质量
                "enable_cache": True,  # 启用缓存
                "parallel_processing": True,  # 并行处理
                "stream_tts": True,  # 流式TTS
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.configs.get(key, default)
    
    def apply_to_opt(self, opt) -> None:
        """将配置应用到opt对象"""
        # 设置TTS引擎
        # 注意：opt.tts 需要真正影响 basereal/BaseReal 中 TTS 实例选择。
        tts_engine = self.get("tts_engine")
        if tts_engine == "azuretts":
            # AzureTTS 依赖环境变量；缺失时不要强行切换，避免直接报错导致完全不说话。
            has_azure_creds = bool(os.getenv("AZURE_SPEECH_KEY")) and bool(os.getenv("AZURE_TTS_REGION"))
            if has_azure_creds:
                opt.tts = "azuretts"
                opt.REF_FILE = "zh-CN-XiaoxiaoMultilingualNeural"
            else:
                print(
                    "AzureTTS credentials not found (need AZURE_SPEECH_KEY & AZURE_TTS_REGION). "
                    "Keep current opt.tts=" + str(getattr(opt, "tts", None))
                )
        else:
            # 默认情况下只做直接覆盖（当前配置主要就是 azuretts）
            if tts_engine:
                opt.tts = tts_engine
        
        # 设置帧率
        opt.fps = self.get("fps", 25)
        
        # 设置质量级别
        opt.quality = self.get("video_quality", "medium")
        
        # 记录配置信息
        print(f"应用性能配置 - 级别: {self.level}")
        print(f"TTS引擎: {self.get('tts_engine')}")
        print(f"LLM模型: {self.get('llm_model')}")
        print(f"帧率: {self.get('fps')} FPS")
        print(f"视频质量: {self.get('video_quality')}")


def get_performance_config() -> PerformanceConfig:
    """获取性能配置实例"""
    # 从环境变量获取优化级别
    level = os.getenv("PERFORMANCE_LEVEL", "balanced")
    return PerformanceConfig(level)


def optimize_llm_response(performance_config: PerformanceConfig) -> Dict[str, Any]:
    """优化LLM响应配置"""
    return {
        "model": performance_config.get("llm_model"),
        "max_tokens": performance_config.get("max_response_length"),
        "temperature": 0.7,
        "stream": True
    }


def optimize_tts_config(performance_config: PerformanceConfig) -> Dict[str, Any]:
    """优化TTS配置"""
    return {
        "engine": performance_config.get("tts_engine"),
        "stream": performance_config.get("stream_tts", True),
        "quality": "fast" if performance_config.level == "speed" else "standard"
    }


if __name__ == "__main__":
    # 测试配置
    config = get_performance_config()
    print(f"当前配置级别: {config.level}")
    print("配置详情:", config.configs)