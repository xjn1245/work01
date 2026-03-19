"""
专业知识库动态融合模块
构建结构化留学知识库，设计灵活的知识更新机制
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any
from logger import logger

class StudyAbroadKnowledgeBase:
    """留学专业知识库类"""
    
    def __init__(self, db_path: str = "study_abroad_kb.db"):
        self.db_path = db_path
        self._init_database()
        
        # 知识分类体系
        self.knowledge_categories = [
            "院校信息", "专业介绍", "申请要求", "签证政策",
            "奖学金", "生活指南", "就业前景", "最新政策"
        ]
        
        # 知识源配置
        self.knowledge_sources = {
            "官方渠道": {"权重": 1.0, "更新频率": "实时"},
            "权威媒体": {"权重": 0.9, "更新频率": "每日"},
            "用户贡献": {"权重": 0.7, "更新频率": "每周"},
            "历史数据": {"权重": 0.5, "更新频率": "每月"}
        }
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建知识条目表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                credibility_score REAL DEFAULT 1.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expiration_date TIMESTAMP,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建知识关系表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER,
                related_id INTEGER,
                relation_type TEXT,
                strength REAL DEFAULT 1.0,
                FOREIGN KEY (entry_id) REFERENCES knowledge_entries (id),
                FOREIGN KEY (related_id) REFERENCES knowledge_entries (id)
            )
        ''')
        
        # 创建更新日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS update_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER,
                update_type TEXT,
                old_content TEXT,
                new_content TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entry_id) REFERENCES knowledge_entries (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_knowledge_entry(self, category: str, title: str, content: str, 
                          source: str, tags: List[str] = None, 
                          expiration_days: int = 365) -> int:
        """添加知识条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 计算可信度分数
        credibility_score = self.knowledge_sources.get(source, {}).get("权重", 0.5)
        
        # 计算过期时间
        expiration_date = datetime.now() + timedelta(days=expiration_days)
        
        cursor.execute('''
            INSERT INTO knowledge_entries 
            (category, title, content, source, credibility_score, expiration_date, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (category, title, content, source, credibility_score, 
              expiration_date, json.dumps(tags) if tags else None))
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"添加知识条目: {title} (ID: {entry_id})")
        return entry_id
    
    def search_knowledge(self, query: str, category: str = None, 
                        min_credibility: float = 0.7) -> List[Dict[str, Any]]:
        """搜索知识库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建搜索条件
        conditions = ["content LIKE ? OR title LIKE ?"]
        params = [f"%{query}%", f"%{query}%"]
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        conditions.append("credibility_score >= ?")
        params.append(min_credibility)
        
        conditions.append("(expiration_date IS NULL OR expiration_date > datetime('now'))")
        
        where_clause = " AND ".join(conditions)
        
        cursor.execute(f'''
            SELECT id, category, title, content, source, credibility_score, 
                   last_updated, tags
            FROM knowledge_entries
            WHERE {where_clause}
            ORDER BY credibility_score DESC, last_updated DESC
            LIMIT 10
        ''', params)
        
        results = []
        for row in cursor.fetchall():
            result = {
                "id": row[0],
                "category": row[1],
                "title": row[2],
                "content": row[3],
                "source": row[4],
                "credibility_score": row[5],
                "last_updated": row[6],
                "tags": json.loads(row[7]) if row[7] else []
            }
            results.append(result)
        
        conn.close()
        return results
    
    def update_knowledge_entry(self, entry_id: int, new_content: str, 
                             update_reason: str = "常规更新") -> bool:
        """更新知识条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取旧内容
        cursor.execute("SELECT content FROM knowledge_entries WHERE id = ?", (entry_id,))
        old_content = cursor.fetchone()[0]
        
        # 更新条目
        cursor.execute('''
            UPDATE knowledge_entries 
            SET content = ?, last_updated = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_content, entry_id))
        
        # 记录更新日志
        cursor.execute('''
            INSERT INTO update_logs (entry_id, update_type, old_content, new_content)
            VALUES (?, ?, ?, ?)
        ''', (entry_id, update_reason, old_content, new_content))
        
        conn.commit()
        conn.close()
        
        logger.info(f"更新知识条目 ID {entry_id}: {update_reason}")
        return True
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # 按分类统计
        cursor.execute('''
            SELECT category, COUNT(*) as count 
            FROM knowledge_entries 
            WHERE expiration_date > datetime('now') OR expiration_date IS NULL
            GROUP BY category
        ''')
        stats["by_category"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 按来源统计
        cursor.execute('''
            SELECT source, COUNT(*) as count 
            FROM knowledge_entries 
            GROUP BY source
        ''')
        stats["by_source"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 最近更新
        cursor.execute('''
            SELECT COUNT(*) 
            FROM knowledge_entries 
            WHERE last_updated > datetime('now', '-7 days')
        ''')
        stats["recent_updates"] = cursor.fetchone()[0]
        
        # 即将过期
        cursor.execute('''
            SELECT COUNT(*) 
            FROM knowledge_entries 
            WHERE expiration_date BETWEEN datetime('now') AND datetime('now', '+30 days')
        ''')
        stats["expiring_soon"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


class KnowledgeUpdateManager:
    """知识更新管理类"""
    
    def __init__(self, knowledge_base: StudyAbroadKnowledgeBase):
        self.kb = knowledge_base
        
        # 更新策略配置
        self.update_strategies = {
            "实时更新": {"优先级": 1, "触发条件": "政策变更"},
            "定期更新": {"优先级": 2, "触发条件": "时间周期"},
            "用户触发": {"优先级": 3, "触发条件": "用户请求"},
            "自动检测": {"优先级": 4, "触发条件": "内容过期"}
        }
    
    def schedule_regular_updates(self):
        """安排定期更新任务"""
        # 这里可以集成定时任务框架
        logger.info("安排定期知识更新任务")
    
    def check_for_updates(self) -> List[Dict[str, Any]]:
        """检查需要更新的知识"""
        conn = sqlite3.connect(self.kb.db_path)
        cursor = conn.cursor()
        
        # 检查过期内容
        cursor.execute('''
            SELECT id, title, category, expiration_date
            FROM knowledge_entries
            WHERE expiration_date < datetime('now')
            ORDER BY expiration_date ASC
            LIMIT 10
        ''')
        
        expired_entries = []
        for row in cursor.fetchall():
            expired_entries.append({
                "id": row[0],
                "title": row[1],
                "category": row[2],
                "expiration_date": row[3]
            })
        
        conn.close()
        return expired_entries
    
    def import_external_data(self, data_source: str, data: List[Dict]) -> int:
        """导入外部数据"""
        imported_count = 0
        
        for item in data:
            try:
                self.kb.add_knowledge_entry(
                    category=item.get("category", "未分类"),
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    source=data_source,
                    tags=item.get("tags", []),
                    expiration_days=item.get("expiration_days", 365)
                )
                imported_count += 1
            except Exception as e:
                logger.error(f"导入数据失败: {e}")
        
        logger.info(f"从 {data_source} 导入 {imported_count} 条知识")
        return imported_count


# 使用示例
def demo_knowledge_base():
    """演示知识库功能"""
    
    # 创建知识库实例
    kb = StudyAbroadKnowledgeBase()
    
    # 添加示例知识
    entry_id = kb.add_knowledge_entry(
        category="申请要求",
        title="美国计算机科学硕士申请条件",
        content="""美国计算机科学硕士申请通常需要：
        1. 本科学位，GPA 3.0以上
        2. 托福100分或雅思7.0分
        3. GRE成绩（部分学校可选）
        4. 相关实习或项目经验""",
        source="官方渠道",
        tags=["美国", "计算机科学", "硕士", "申请条件"]
    )
    
    # 搜索知识
    results = kb.search_knowledge("计算机科学", category="申请要求")
    print(f"搜索结果: {len(results)} 条")
    for result in results:
        print(f"- {result['title']} (可信度: {result['credibility_score']})")
    
    # 获取统计信息
    stats = kb.get_knowledge_statistics()
    print(f"知识库统计: {stats}")
    
    # 知识更新管理
    update_manager = KnowledgeUpdateManager(kb)
    expired_entries = update_manager.check_for_updates()
    print(f"需要更新的条目: {len(expired_entries)} 条")


if __name__ == "__main__":
    demo_knowledge_base()