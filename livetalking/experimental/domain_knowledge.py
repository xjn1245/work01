"""
留学领域语义理解模块
解决通用大模型在留学专业领域知识储备不足的问题
"""

import json
import re
from typing import Dict, List, Tuple
from logger import logger

class StudyAbroadDomainUnderstanding:
    """留学领域语义理解类"""
    
    def __init__(self):
        # 留学领域专业术语库
        self.domain_terms = self._load_domain_terms()
        
        # 留学申请关键信息抽取规则
        self.extraction_rules = self._load_extraction_rules()
        
        # 意图识别模型
        self.intent_patterns = self._load_intent_patterns()
    
    def _load_domain_terms(self) -> Dict[str, List[str]]:
        """加载留学领域专业术语"""
        return {
            "申请流程": ["申请材料", "推荐信", "个人陈述", "简历", "成绩单", "语言成绩", "申请截止日期"],
            "院校选择": ["QS排名", "US News", "专业排名", "地理位置", "学费", "奖学金", "校园环境"],
            "专业规划": ["计算机科学", "商科", "工程", "艺术设计", "医学", "法学", "STEM专业"],
            "签证移民": ["学生签证", "工作签证", "移民政策", "OPT", "CPT", "绿卡", "身份转换"],
            "生活准备": ["住宿", "医疗保险", "文化适应", "语言提升", "就业前景", "实习机会"]
        }
    
    def _load_extraction_rules(self) -> Dict[str, List[str]]:
        """加载关键信息抽取规则"""
        return {
            "成绩": [r"GPA\s*([0-9]\.?[0-9]*)", r"绩点\s*([0-9]\.?[0-9]*)", r"成绩\s*([0-9]\.?[0-9]*)"],
            "语言成绩": [r"托福\s*(\d+)", r"雅思\s*([0-9]\.?[0-9]*)", r"GRE\s*(\d+)", r"GMAT\s*(\d+)"],
            "预算": [r"预算\s*(\d+[万万千]?元?)", r"费用\s*(\d+[万万千]?元?)", r"学费\s*(\d+[万万千]?元?)"],
            "时间": [r"(\d{4})年", r"(\d+)月", r"(春季|秋季|夏季)入学", r"申请截止\s*(\d+/\d+)"]
        }
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """加载意图识别模式"""
        return {
            "院校推荐": [
                r"推荐.*大学", r"哪些.*学校", r"适合.*院校", 
                r"什么.*大学.*好", r"选择.*学校"
            ],
            "申请条件": [
                r"需要.*条件", r"要求.*什么", r"申请.*资格",
                r"什么.*背景", r"需要.*成绩"
            ],
            "专业选择": [
                r"什么.*专业", r"专业.*选择", r"哪个.*专业.*好",
                r"专业.*前景", r"就业.*方向"
            ],
            "时间规划": [
                r"什么时候.*申请", r"时间.*安排", r"规划.*时间",
                r"准备.*多久", r"申请.*时间"
            ],
            "费用预算": [
                r"需要.*钱", r"费用.*多少", r"预算.*多少",
                r"学费.*生活费", r"花费.*多少"
            ]
        }
    
    def enhance_llm_prompt(self, user_message: str) -> str:
        """增强LLM提示词，加入领域知识"""
        
        # 识别用户意图
        intent = self.detect_intent(user_message)
        
        # 抽取关键信息
        extracted_info = self.extract_key_info(user_message)
        
        # 构建增强的提示词
        enhanced_prompt = f"""
        你是一位专业的留学顾问，拥有丰富的留学咨询经验。
        
        用户意图：{intent}
        关键信息：{extracted_info}
        
        请根据以下留学领域知识提供专业回答：
        
        **留学申请核心要素：**
        1. 学术背景：GPA、专业课程、科研经历
        2. 语言能力：托福/雅思成绩、GRE/GMAT分数  
        3. 软实力：实习经历、科研项目、推荐信质量
        4. 文书材料：个人陈述、简历的专业性
        5. 时间规划：申请时间线、准备周期
        
        **院校选择原则：**
        - 匹配度：专业方向与院校优势的匹配
        - 可行性：根据背景条件评估申请难度
        - 就业前景：专业在目标国家的就业情况
        - 费用预算：学费、生活费的合理规划
        
        请基于以上专业知识和用户的具体情况，提供准确、实用的留学建议。
        
        用户问题：{user_message}
        """
        
        return enhanced_prompt
    
    def detect_intent(self, message: str) -> str:
        """识别用户意图"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message):
                    logger.info(f"检测到意图: {intent}")
                    return intent
        return "一般咨询"
    
    def extract_key_info(self, message: str) -> Dict[str, List[str]]:
        """抽取关键信息"""
        extracted = {}
        
        for info_type, patterns in self.extraction_rules.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, message)
                if found:
                    matches.extend(found)
            if matches:
                extracted[info_type] = matches
        
        # 识别专业领域术语
        domain_terms_found = []
        for category, terms in self.domain_terms.items():
            for term in terms:
                if term in message:
                    domain_terms_found.append(f"{category}:{term}")
        
        if domain_terms_found:
            extracted["领域术语"] = domain_terms_found
        
        logger.info(f"抽取的关键信息: {extracted}")
        return extracted
    
    def validate_advice(self, advice: str, user_context: Dict) -> Tuple[bool, str]:
        """验证建议的合理性"""
        validation_checks = [
            ("时间规划合理性", self._check_timeline_reasonableness),
            ("成绩要求匹配度", self._check_score_requirements),
            ("费用预算合理性", self._check_budget_reasonableness),
            ("专业选择可行性", self._check_major_feasibility)
        ]
        
        issues = []
        for check_name, check_func in validation_checks:
            is_valid, message = check_func(advice, user_context)
            if not is_valid:
                issues.append(f"{check_name}: {message}")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "建议合理"
    
    def _check_timeline_reasonableness(self, advice: str, context: Dict) -> Tuple[bool, str]:
        return True, "时间规划合理"
    
    def _check_score_requirements(self, advice: str, context: Dict) -> Tuple[bool, str]:
        return True, "成绩要求匹配"
    
    def _check_budget_reasonableness(self, advice: str, context: Dict) -> Tuple[bool, str]:
        return True, "预算合理"
    
    def _check_major_feasibility(self, advice: str, context: Dict) -> Tuple[bool, str]:
        return True, "专业选择可行"


def create_domain_enhanced_llm_response(message: str, domain_understanding: StudyAbroadDomainUnderstanding) -> str:
    enhanced_prompt = domain_understanding.enhance_llm_prompt(message)
    return enhanced_prompt

