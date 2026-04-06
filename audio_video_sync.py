"""
音画同步与情感表达模块
实现语音、口型与文本语义的精准匹配，生成符合情感倾向的面部表情
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any
from logger import logger

class AudioVideoSynchronizer:
    """音画同步优化类"""
    
    def __init__(self):
        # 音画同步配置
        self.sync_config = {
            "max_audio_delay": 0.1,  # 最大音频延迟(秒)
            "max_video_delay": 0.05,  # 最大视频延迟(秒)
            "target_sync_threshold": 0.02,  # 目标同步阈值
            "adjustment_interval": 0.5  # 调整间隔(秒)
        }
        
        # 同步状态跟踪
        self.sync_state = {
            "audio_timestamps": [],
            "video_timestamps": [],
            "last_adjustment": time.time()
        }
    
    def calculate_sync_offset(self, audio_timestamp: float, video_timestamp: float) -> float:
        """计算音画同步偏移量"""
        offset = audio_timestamp - video_timestamp
        
        # 记录时间戳用于统计分析
        self.sync_state["audio_timestamps"].append(audio_timestamp)
        self.sync_state["video_timestamps"].append(video_timestamp)
        
        # 保持最近100个样本
        if len(self.sync_state["audio_timestamps"]) > 100:
            self.sync_state["audio_timestamps"].pop(0)
            self.sync_state["video_timestamps"].pop(0)
        
        return offset
    
    def needs_sync_adjustment(self, offset: float) -> bool:
        """判断是否需要同步调整"""
        current_time = time.time()
        
        # 检查偏移是否超过阈值
        if abs(offset) > self.sync_config["target_sync_threshold"]:
            # 检查是否达到调整间隔
            if current_time - self.sync_state["last_adjustment"] > self.sync_config["adjustment_interval"]:
                self.sync_state["last_adjustment"] = current_time
                return True
        
        return False
    
    def calculate_adjustment(self, offset: float) -> Dict[str, float]:
        """计算同步调整参数"""
        adjustment = {}
        
        if offset > 0:  # 音频领先
            adjustment["video_speed"] = 1.0 + min(offset * 0.5, 0.1)  # 加快视频
            adjustment["audio_delay"] = 0.0
        else:  # 视频领先
            adjustment["video_speed"] = 1.0 - min(abs(offset) * 0.5, 0.1)  # 减慢视频
            adjustment["audio_delay"] = min(abs(offset), self.sync_config["max_audio_delay"])
        
        logger.info(f"音画同步调整: 偏移{offset:.3f}s, 视频速度{adjustment['video_speed']:.2f}")
        
        return adjustment


class EmotionalExpressionGenerator:
    """情感表达生成类"""
    
    def __init__(self):
        # 情感词汇与表情映射
        self.emotion_mapping = self._load_emotion_mapping()
        
        # 语音情感分析参数
        self.speech_emotion_params = {
            "pitch_range": (80, 300),  # 音高范围(Hz)
            "energy_range": (0.1, 1.0),  # 能量范围
            "speech_rate_range": (100, 300)  # 语速范围(字/分钟)
        }
    
    def _load_emotion_mapping(self) -> Dict[str, Dict]:
        """加载情感映射配置"""
        return {
            "高兴": {
                "facial_expression": "smile",
                "intensity": 0.8,
                "lip_movement": "energetic",
                "eye_expression": "bright"
            },
            "专业": {
                # 原先 professional 映射为 neutral，导致多数 LLM 回复（“建议/分析/规划”）
                # 都会落到 neutral，看不到表情变化。这里改为 encouraging，提高可见度。
                "facial_expression": "encouraging",
                "intensity": 0.65,
                "lip_movement": "precise",
                "eye_expression": "focused"
            },
            "关切": {
                "facial_expression": "concerned",
                "intensity": 0.7,
                "lip_movement": "gentle",
                "eye_expression": "caring"
            },
            "鼓励": {
                "facial_expression": "encouraging",
                "intensity": 0.75,
                "lip_movement": "emphatic",
                "eye_expression": "supportive"
            }
        }
    
    def analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """分析文本情感倾向"""
        # 情感关键词检测
        emotion_keywords = {
            "高兴": ["恭喜", "优秀", "成功", "好消息", "祝贺"],
            "专业": ["建议", "分析", "评估", "规划", "策略"],
            "关切": ["担心", "关注", "注意", "谨慎", "风险"],
            "鼓励": ["加油", "相信", "努力", "坚持", "进步"]
        }
        
        emotion_scores = {emotion: 0.0 for emotion in emotion_keywords.keys()}
        
        # 计算情感得分（关键词 + 标点/常见短语兜底）
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[emotion] += 1.0

        # 标点/语气强制倾向：让自动表情更容易“看得见”
        # 注意：最终 facial_expression 仍由 emotion_mapping 决定。
        if "！" in text or "太棒" in text or "太好了" in text or "好极了" in text:
            emotion_scores["高兴"] += 2.5
        if "？" in text or "吗" in text or "要不要" in text:
            emotion_scores["关切"] += 1.2
        if "别担心" in text or "没问题" in text or "不用担心" in text:
            emotion_scores["鼓励"] += 1.8
        if "风险" in text or "注意" in text or "小心" in text:
            emotion_scores["关切"] += 1.0
        
        # 归一化得分
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score
        
        # 如果没有检测到明显情感，默认使用专业情感
        if total_score == 0:
            emotion_scores["专业"] = 1.0
        
        return emotion_scores
    
    def generate_facial_expression(self, emotion_scores: Dict[str, float], 
                                 text_semantics: str) -> Dict[str, Any]:
        """生成面部表情参数"""
        # 确定主要情感
        main_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        emotion_config = self.emotion_mapping[main_emotion]
        
        # 根据语义调整表情强度
        intensity = self._adjust_intensity_by_semantics(emotion_config["intensity"], text_semantics)
        
        expression_params = {
            "emotion_type": main_emotion,
            "facial_expression": emotion_config["facial_expression"],
            "intensity": intensity,
            "lip_movement_style": emotion_config["lip_movement"],
            "eye_expression": emotion_config["eye_expression"],
            "duration": self._calculate_expression_duration(text_semantics)
        }
        
        logger.info(f"生成面部表情: {expression_params}")
        
        return expression_params
    
    def _adjust_intensity_by_semantics(self, base_intensity: float, semantics: str) -> float:
        """根据语义调整表情强度"""
        # 根据文本语义特征调整强度
        intensity_modifiers = {
            "强烈": 1.2,  # 感叹号、强调词汇
            "温和": 0.8,   # 平缓叙述
            "中性": 1.0    # 一般陈述
        }
        
        # 简单的语义强度判断
        if "！" in semantics or "强烈" in semantics or "重要" in semantics:
            modifier = intensity_modifiers["强烈"]
        elif "建议" in semantics or "可能" in semantics or "考虑" in semantics:
            modifier = intensity_modifiers["温和"]
        else:
            modifier = intensity_modifiers["中性"]
        
        return min(1.0, base_intensity * modifier)
    
    def _calculate_expression_duration(self, semantics: str) -> float:
        """计算表情持续时间"""
        # 基于文本长度和语义复杂度计算持续时间
        text_length = len(semantics)
        base_duration = max(1.0, text_length * 0.1)  # 基础持续时间
        
        # 语义复杂度调整
        complexity_factors = ["因为", "所以", "但是", "虽然", "如果"]
        complexity = sum(1 for factor in complexity_factors if factor in semantics)
        
        return base_duration * (1 + complexity * 0.1)


class LipSyncOptimizer:
    """口型同步优化类"""
    
    def __init__(self):
        # 音素-口型映射
        self.phoneme_lip_mapping = self._load_phoneme_mapping()
    
    def _load_phoneme_mapping(self) -> Dict[str, List[float]]:
        """加载音素-口型映射"""
        # 简化的音素到口型参数映射
        return {
            "a": [0.8, 0.6, 0.3],  # 开口度, 嘴唇宽度, 嘴唇圆度
            "i": [0.4, 0.8, 0.1],
            "u": [0.5, 0.5, 0.9],
            "e": [0.6, 0.7, 0.2],
            "o": [0.7, 0.5, 0.8],
            "m": [0.3, 0.6, 0.4],
            "n": [0.4, 0.7, 0.3],
            "s": [0.5, 0.8, 0.1]
        }
    
    def generate_lip_movements(self, audio_data: np.ndarray, 
                             text_phonemes: List[str]) -> List[Dict[str, Any]]:
        """生成口型运动序列"""
        lip_movements = []
        
        # 基于音频能量和音素序列生成口型
        for i, phoneme in enumerate(text_phonemes):
            if phoneme in self.phoneme_lip_mapping:
                lip_params = self.phoneme_lip_mapping[phoneme]
                
                # 根据音频能量调整口型强度
                audio_energy = self._calculate_audio_energy(audio_data, i, len(text_phonemes))
                intensity = 0.5 + audio_energy * 0.5
                
                lip_movement = {
                    "phoneme": phoneme,
                    "lip_parameters": lip_params,
                    "intensity": intensity,
                    "duration": 0.1,  # 基础持续时间
                    "transition": "smooth"  # 平滑过渡
                }
                
                lip_movements.append(lip_movement)
        
        return lip_movements
    
    def _calculate_audio_energy(self, audio_data: np.ndarray, 
                              segment_index: int, total_segments: int) -> float:
        """计算音频能量"""
        # 简化实现：计算对应片段的音频能量
        segment_size = len(audio_data) // total_segments
        start_idx = segment_index * segment_size
        end_idx = min((segment_index + 1) * segment_size, len(audio_data))
        
        if start_idx < end_idx:
            segment_energy = np.mean(np.abs(audio_data[start_idx:end_idx]))
            return min(1.0, segment_energy * 10)  # 归一化
        
        return 0.5  # 默认值


# 使用示例
def demo_audio_video_sync():
    """演示音画同步功能"""
    
    # 音画同步测试
    synchronizer = AudioVideoSynchronizer()
    offset = synchronizer.calculate_sync_offset(1.0, 0.95)  # 音频领先0.05秒
    
    if synchronizer.needs_sync_adjustment(offset):
        adjustment = synchronizer.calculate_adjustment(offset)
        print(f"同步调整参数: {adjustment}")
    
    # 情感表达测试
    emotion_generator = EmotionalExpressionGenerator()
    test_text = "恭喜您获得录取！这是非常好的消息。"
    
    emotion_scores = emotion_generator.analyze_text_emotion(test_text)
    print(f"情感分析结果: {emotion_scores}")
    
    expression_params = emotion_generator.generate_facial_expression(emotion_scores, test_text)
    print(f"面部表情参数: {expression_params}")
    
    # 口型同步测试
    lip_sync_optimizer = LipSyncOptimizer()
    test_phonemes = ["g", "o", "n", "g", "x", "i"]
    test_audio = np.random.random(1000)  # 模拟音频数据
    
    lip_movements = lip_sync_optimizer.generate_lip_movements(test_audio, test_phonemes)
    print(f"口型运动序列: {lip_movements[:3]}...")  # 显示前3个


if __name__ == "__main__":
    demo_audio_video_sync()