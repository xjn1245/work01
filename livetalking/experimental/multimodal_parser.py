"""
多模态输入统一解析模块
实现文本、弹幕、语音等不同渠道输入的高效解析与标准化处理
"""

import re
from typing import Dict, List, Any
from datetime import datetime
from logger import logger


class MultimodalInputParser:
    """多模态输入解析类"""

    def __init__(self):
        self.input_type_patterns = {
            "text": r"^[\w\W]{1,500}$",
            "bullet_chat": r"^\[.*?\].*",
            "voice": r"^VOICE:",
            "command": r"^/(\w+)",
            "question": r"^.*[？?].*$",
            "statement": r"^.*[。！!].*$",
        }
        self.cleaning_rules = {
            "remove_emojis": True,
            "normalize_spaces": True,
            "remove_special_chars": False,
            "max_length": 500,
        }

    def parse_input(self, raw_input: str, input_source: str = "unknown") -> Dict[str, Any]:
        input_type = self._identify_input_type(raw_input)
        cleaned_content = self._clean_content(raw_input)
        parsed_result = {
            "original_input": raw_input,
            "input_source": input_source,
            "input_type": input_type,
            "cleaned_content": cleaned_content,
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(f"解析输入: {input_type} from {input_source}")
        return parsed_result

    def _identify_input_type(self, raw_input: str) -> str:
        if re.match(self.input_type_patterns["command"], raw_input):
            return "command"
        if re.match(self.input_type_patterns["voice"], raw_input):
            return "voice"
        if re.match(self.input_type_patterns["bullet_chat"], raw_input):
            return "bullet_chat"
        if re.match(self.input_type_patterns["question"], raw_input):
            return "question"
        if re.match(self.input_type_patterns["statement"], raw_input):
            return "statement"
        return "text"

    def _clean_content(self, raw_input: str) -> str:
        cleaned = raw_input
        if cleaned.startswith("VOICE:"):
            cleaned = cleaned[6:].strip()
        if cleaned.startswith("/"):
            cleaned = cleaned[1:].strip()
        if cleaned.startswith("[") and "]" in cleaned:
            end_bracket = cleaned.find("]")
            cleaned = cleaned[end_bracket + 1 :].strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        max_len = self.cleaning_rules["max_length"]
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len] + "..."
        return cleaned

