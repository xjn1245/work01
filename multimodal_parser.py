"""
多模态输入统一解析模块
实现文本、弹幕、语音等不同渠道输入的高效解析与标准化处理
"""

import re
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
from logger import logger

class MultimodalInputParser:
    """多模态输入解析类"""
    
    def __"""
多模态输入统一解析模块
实现文本、弹幕、语音等不同渠道输入的高效解析与标准化处理
"""

import re
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
from logger import logger

class MultimodalInputParser:
    """多模态输入解析类"""
    
    def __init__(self):
        # 输入类型识别规则
        self.input_type_patterns = {
            "text": r"^[\w\W]{1,500}$",  # 普通文本
            "bullet_chat": r"^\[.*?\].*",  # 弹幕格式
            "voice": r"^VOICE:",  # 语音输入标记
            "command": r"^/(\w+)",  # 命令格式
            "question": r"^.*[？?].*$",  # 问题格式
            "statement": r"^.*[。！!].*$"  # 陈述格式
        }
        
        # 内容清洗规则
        self.cleaning_rules = {
            "remove_emojis": True,
            "normalize_spaces": True,
            "remove_special_chars": False,
            "max_length": 500
        }
        
        # 语义解析配置
        self.semantic_config = {
            "extract_entities": True,
            "detect_intent": True,
            "classify_topic": True,
            "assess_urgency": True
        }
    
    def parse_input(self, raw_input: str, input_source: str = "unknown") -> Dict[str, Any]:
        """解析多模态输入"""
        
        # 1. 识别输入类型
        input_type = self._identify_input_type(raw_input)
        
        # 2. 内容清洗和标准化
        cleaned_content = self._clean_content(raw_input)
        
        # 3. 语义解析
        semantic_analysis = self._analyze_semantics(cleaned_content)
        
        # 4. 构建标准化输出
        parsed_result = {
            "original_input": raw_input,
            "input_source": input_source,
            "input_type": input_type,
            "cleaned_content": cleaned_content,
            "timestamp": datetime.now().isoformat(),
            "semantic_analysis": semantic_analysis,
            "processing_steps": ["类型识别", "内容清洗", "语义解析"]
        }
        
        logger.info(f"解析输入: {input_type} from {input_source}")
        
        return parsed_result
    
    def _identify_input_type(self, raw_input: str) -> str:
        """识别输入类型"""
        
        # 检查命令格式
        if re.match(self.input_type_patterns["command"], raw_input):
            return "command"
        
        # 检查语音输入标记
        if re.match(self.input_type_patterns["voice"], raw_input):
            return "voice"
        
        # 检查弹幕格式
        if re.match(self.input_type_patterns["bullet_chat"], raw_input):
            return "bullet_chat"
        
        # 检查问题格式
        if re.match(self.input_type_patterns["question"], raw_input):
            return "question"
        
        # 检查陈述格式
        if re.match(self.input_type_patterns["statement"], raw_input):
            return "statement"
        
        # 默认为普通文本
        return "text"
    
    def _clean_content(self, raw_input: str) -> str:
        """内容清洗和标准化"""
        
        cleaned = raw_input
        
        # 移除语音标记
        if cleaned.startswith("VOICE:"):
            cleaned = cleaned[6:].strip()
        
        # 移除命令标记
        if cleaned.startswith("/"):
            # 保留命令内容但移除斜杠
            cleaned = cleaned[1:].strip()
        
        # 移除弹幕标记
        if cleaned.startswith("[") and "]" in cleaned:
            # 提取弹幕内容
            end_bracket = cleaned.find("]")
            cleaned = cleaned[end_bracket + 1:].strip()
        
        # 移除表情符号
        if self.cleaning_rules["remove_emojis"]:
            cleaned = self._remove_emojis(cleaned)
        
        # 标准化空格
        if self.cleaning_rules["normalize_spaces"]:
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        # 长度限制
        max_len = self.cleaning_rules["max_length"]
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len] + "..."
        
        return cleaned
    
    def _remove_emojis(self, text: str) -> str:
        """移除表情符号"""
        # 简单的表情符号移除
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # 表情符号
            "\U0001F300-\U0001F5FF"  # 符号和象形文字
            "\U0001F680-\U0001F6FF"  # 交通和地图符号
            "\U0001F1E0-\U0001F1FF"  # 旗帜符号
            "]+", flags=re.UNICODE)
        
        return emoji_pattern.sub(r'', text)
    
    def _analyze_semantics(self, content: str) -> Dict[str, Any]:
        """语义解析"""
        
        analysis = {}
        
        # 实体抽取
        if self.semantic_config["extract_entities"]:
            analysis["entities"] = self._extract_entities(content)
        
        # 意图识别
        if self.semantic_config["detect_intent"]:
            analysis["intent"] = self._detect_intent(content)
        
        # 主题分类
        if self.semantic_config["classify_topic"]:
            analysis["topic"] = self._classify_topic(content)
        
        # 紧急程度评估
        if self.semantic_config["assess_urgency"]:
            analysis["urgency"] = self._assess_urgency(content)
        
        # 情感分析
        analysis["sentiment"] = self._analyze_sentiment(content)
        
        return analysis
    
    def _extract_entities(self, content: str) -> List[Dict[str, str]]:
        """抽取实体"""
        
        entities = []
        
        # 国家/地区
        countries = ["美国", "英国", "加拿大", "澳大利亚", "新加坡", "香港", "日本"]
        for country in countries:
            if country in content:
                entities.append({"type": "国家", "value": country})
        
        # 学位类型
        degrees = ["本科", "硕士", "博士", "MBA", "PhD"]
        for degree in degrees:
            if degree in content:
                entities.append({"type": "学位", "value": degree})
        
        # 专业领域
        majors = ["计算机", "商科", "工程", "医学", "法律", "艺术", "教育"]
        for major in majors:
            if major in content:
                entities.append({"type": "专业", "value": major})
        
        # 时间相关
        time_patterns = [
            (r"(\d{4})年", "年份"),
            (r"(春季|秋季|夏季)", "入学季节"),
            (r"(\d+)月", "月份")
        ]
        
        for pattern, entity_type in time_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                entities.append({"type": entity_type, "value": match})
        
        return entities
    
    def _detect_intent(self, content: str) -> str:
        """识别意图"""
        
        intent_patterns = {
            "咨询": [r"咨询", r"请问", r"想问", r"了解"],
            "比较": [r"哪个好", r"比较", r"对比", r"区别"],
            "申请": [r"申请", r"怎么申请", r"需要什么", r"条件"],
            "费用": [r"多少钱", r"费用", r"学费", r"预算"],
            "时间": [r"什么时候", r"时间", r"截止", r"准备多久"]
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    return intent
        
        return "一般咨询"
    
    def _classify_topic(self, content: str) -> List[str]:
        """分类主题"""
        
        topics = []
        topic_keywords = {
            "院校选择": ["大学", "学校", "院校", "排名", "QS"],
            "专业规划": ["专业", "方向", "就业", "前景"],
            "申请流程": ["申请", "材料", "推荐信", "文书"],
            "签证移民": ["签证", "移民", "绿卡", "身份"],
            "生活指南": ["生活", "住宿", "保险", "文化"]
        }
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    topics.append(topic)
                    break
        
        return topics if topics else ["综合咨询"]
    
    def _assess_urgency(self, content: str) -> str:
        """评估紧急程度"""
        
        urgent_keywords = ["紧急", "尽快", "马上", "立刻", "着急"]
        for keyword in urgent_keywords:
            if keyword in content:
                return "高"
        
        question_marks = content.count("?") + content.count("？")
        if question_marks >= 3:
            return "中"
        
        return "低"
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """情感分析"""
        
        # 简化的情感分析
        positive_words = ["好", "优秀", "恭喜", "感谢", "满意"]
        negative_words = ["差", "不好", "担心", "问题", "困难"]
        
        positive_score = sum(1 for word in positive_words if word in content)
        negative_score = sum(1 for word in negative_words if word in content)
        
        total_words = len(content)
        
        return {
            "positive": positive_score / max(1, total_words) * 10,
            "negative": negative_score / max(1, total_words) * 10,
            "neutral": 1.0 - (positive_score + negative_score) / max(1, total_words) * 10
        }


class InputRouter:
    """输入路由类"""
    
    def __init__(self, parser: MultimodalInputParser):
        self.parser = parser
        
        # 路由规则
        self.routing_rules = {
            "command": self._route_command,
            "voice": self._route_voice,
            "bullet_chat": self._route_bullet_chat,
            "question": self._route_question,
            "statement": self._route_statement,
            "text": self._route_text
        }
    
    def route_input(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """路由输入到相应处理模块"""
        
        input_type = parsed_input["input_type"]
        routing_function = self.routing_rules.get(input_type, self._route_default)
        
        routing_result = routing_function(parsed_input)
        
        # 添加路由信息
        routing_result.update({
            "routing_decision": input_type,
            "routed_at": datetime.now().isoformat()
        })
        
        return routing_result
    
    def _route_command(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """路由命令输入"""
        return {
            "handler": "command_handler",
            "priority": "high",
            "requires_immediate_response": True,
            "processing_timeout": 5.0
        }
    
    def _route_voice(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """路由语音输入"""
        return {
            "handler": "voice_processor",
            "priority": "medium",
            "requires_audio_response": True,
            "processing_timeout": 10.0
        }
    
    def _route_bullet_chat(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """路由弹幕输入"""
        return {
            "handler": "bullet_chat_processor",
            "priority": "low",
            "batch_processing": True,
            "processing_timeout": 3.0
        }
    
    def _route_question(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """路由问题输入"""
        urgency = parsed_input["semantic_analysis"].get("urgency", "低")
        
        return {
            "handler": "question_answerer",
            "priority": "high" if urgency == "高" else "medium",
            "requires_detailed_response": True,
            "processing_timeout": 15.0
        }
    
    def _route_statement(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """路由陈述输入"""
        return {
            "handler": "conversation_manager",
            "priority": "medium",
            "maintain_context": True,
            "processing_timeout": 8.0
        }
    
    def _route_text(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """路由普通文本输入"""
        return {
            "handler": "text_processor",
            "priority": "medium",
            "processing_timeout": 6.0
        }
    
    def _route_default(self, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """默认路由"""
        return {
            "handler": "default_processor",
            "priority": "low",
            "processing_timeout": 5.0
        }


# 使用示例
def demo_multimodal_parsing():
    """演示多模态输入解析功能"""
    
    parser = MultimodalInputParser()
    router = InputRouter(parser)
    
    # 测试不同输入类型
    test_inputs = [
        ("请问美国计算机硕士申请需要什么条件？", "text"),
        ("VOICE:我想了解英国商科留学费用", "voice"),
        ("[用户123]加拿大签证好办吗？", "bullet_chat"),
        ("/help 显示帮助信息", "command"),
        ("恭喜你获得录取！🎉", "statement")
    ]
    
    for raw_input, source in test_inputs:
        print(f"\n=== 解析输入: {source} ===")
        print(f"原始输入: {raw_input}")
        
        # 解析输入
        parsed_input = parser.parse_input(raw_input, source)
        print(f"解析结果:")
        print(f"  类型: {parsed_input['input_type']}")
        print(f"  清洗后: {parsed_input['cleaned_content']}")
        print(f"  语义分析: {parsed_input['semantic_analysis']}")
        
        # 路由输入
        routing_result = router.route_input(parsed_input)
        print(f"  路由决策: {routing_result['handler']} (优先级: {routing_result['priority']})")


if __name__ == "__main__":
    demo_multimodal_parsing()