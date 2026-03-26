"""
专业知识库动态融合模块
构建结构化留学知识库，设计灵活的知识更新机制
"""

import json
import sqlite3
import hashlib
import threading
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
from logger import logger

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

        # --- Vector index cache (for hybrid retrieval) ---
        self._vector_index_lock = threading.Lock()
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._index_entry_ids: List[int] = []
        self._index_entries_meta: Dict[int, Dict[str, Any]] = {}
        self._db_revision_key: str | None = None
        self._last_vector_build_at: str | None = None
    
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

    def _get_db_revision_key(self) -> str:
        """
        用 (总数, 最新更新时间) 作为“粗略变更”信号。
        这样能在热更新（插入/更新）后自动重建向量索引。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), MAX(last_updated) FROM knowledge_entries")
        cnt, max_ts = cursor.fetchone()
        conn.close()
        return f"{cnt}:{max_ts}"

    def _parse_expiration_date(self, expiration_date) -> datetime | None:
        if not expiration_date:
            return None
        if isinstance(expiration_date, datetime):
            return expiration_date
        # SQLite may return strings like "2026-03-26 12:34:56"
        s = str(expiration_date)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt)
            except Exception:
                continue
        return None

    def _maybe_rebuild_vector_index(self, max_features: int = 5000) -> None:
        revision = self._get_db_revision_key()
        if self._db_revision_key == revision and self._vectorizer is not None and self._tfidf_matrix is not None:
            return

        with self._vector_index_lock:
            # re-check after acquiring lock
            revision = self._get_db_revision_key()
            if self._db_revision_key == revision and self._vectorizer is not None and self._tfidf_matrix is not None:
                return

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, category, title, content, source, credibility_score, last_updated, tags, expiration_date
                FROM knowledge_entries
                """
            )
            rows = cursor.fetchall()
            conn.close()

            texts: List[str] = []
            entry_ids: List[int] = []
            entries_meta: Dict[int, Dict[str, Any]] = {}

            for row in rows:
                entry_id = row[0]
                category = row[1]
                title = row[2]
                content = row[3]
                source = row[4]
                credibility_score = row[5]
                last_updated = row[6]
                tags_raw = row[7]
                expiration_date = row[8]
                tags = json.loads(tags_raw) if tags_raw else []

                entry_ids.append(entry_id)
                texts.append(f"{title}\n{content}")
                entries_meta[entry_id] = {
                    "id": entry_id,
                    "category": category,
                    "title": title,
                    "content": content,
                    "source": source,
                    "credibility_score": credibility_score,
                    "last_updated": last_updated,
                    "tags": tags,
                    "expiration_date": expiration_date,
                }

            # If KB is empty, keep index empty
            if not texts:
                self._vectorizer = None
                self._tfidf_matrix = None
                self._index_entry_ids = []
                self._index_entries_meta = {}
                self._db_revision_key = revision
                return

            # TF-IDF vectors: lightweight, no external embedding model required.
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=1,
                stop_words=None,
            )
            tfidf_matrix = vectorizer.fit_transform(texts)

            self._vectorizer = vectorizer
            self._tfidf_matrix = tfidf_matrix
            self._index_entry_ids = entry_ids
            self._index_entries_meta = entries_meta
            self._db_revision_key = revision
            self._last_vector_build_at = datetime.now().isoformat(timespec="seconds")

    def _compute_keyword_score(self, query: str, title: str, content: str) -> float:
        """
        在没有 embedding 的情况下，用“字符串包含 + 字符重叠”的方式估计相关度。
        """
        q = (query or "").strip()
        if not q:
            return 0.0

        text_all = f"{title}\n{content}"
        score = 0.0

        # Exact substring boost
        if q in title:
            score += 3.0
        if q in content:
            score += 2.0

        # Chinese character overlap (very cheap heuristic)
        chars = re.findall(r"[\u4e00-\u9fff]", q)
        if chars:
            uniq_chars = set(chars)
            overlap = sum(1 for c in uniq_chars if c in text_all)
            score += float(overlap) / max(1, len(uniq_chars)) * 2.0

        # Word-like overlap for alphanumerics
        words = re.findall(r"[a-zA-Z0-9]+", q)
        if words:
            w_uniq = set(words)
            overlap_w = sum(1 for w in w_uniq if w.lower() in text_all.lower())
            score += float(overlap_w) / max(1, len(w_uniq)) * 1.5

        return score

    def search_knowledge_hybrid(
        self,
        query: str,
        category: str = None,
        min_credibility: float = 0.7,
        top_k: int = 5,
        alpha: float = 0.5,
        keyword_candidate_limit: int = 200,
        vector_candidate_limit: int = 50,
        max_vector_candidates_merge: int = 120,
    ) -> List[Dict[str, Any]]:
        """
        混合检索（向量 TF-IDF + 关键词启发式评分）。
        热更新：只要知识库发生插入/更新（last_updated 变化），会自动重建向量索引。
        """
        # Ensure vector index exists (and is rebuilt after KB change)
        self._maybe_rebuild_vector_index()

        now = datetime.now()
        active_ids: set[int] = set()

        # Keyword candidates via LIKE, then compute heuristic keyword score in Python
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        conditions = ["(content LIKE ? OR title LIKE ?)"]
        params = [f"%{query}%", f"%{query}%"]

        if category:
            conditions.append("category = ?")
            params.append(category)

        conditions.append("credibility_score >= ?")
        params.append(min_credibility)

        conditions.append("(expiration_date IS NULL OR expiration_date > datetime('now'))")

        where_clause = " AND ".join(conditions)
        cursor.execute(
            f"""
            SELECT id, category, title, content, source, credibility_score, last_updated, tags, expiration_date
            FROM knowledge_entries
            WHERE {where_clause}
            LIMIT {keyword_candidate_limit}
            """,
            params,
        )
        keyword_rows = cursor.fetchall()
        conn.close()

        keyword_scores: Dict[int, float] = {}
        keyword_meta: Dict[int, Dict[str, Any]] = {}
        for row in keyword_rows:
            entry_id = row[0]
            title = row[2]
            content = row[3]
            credibility_score = row[5]
            last_updated = row[6]
            tags_raw = row[7]
            expiration_date = row[8]
            tags = json.loads(tags_raw) if tags_raw else []

            kw_score = self._compute_keyword_score(query, title, content)
            keyword_scores[entry_id] = kw_score
            keyword_meta[entry_id] = {
                "id": entry_id,
                "category": row[1],
                "title": title,
                "content": content,
                "source": row[4],
                "credibility_score": credibility_score,
                "last_updated": last_updated,
                "tags": tags,
                "expiration_date": expiration_date,
                "keyword_score": kw_score,
            }

        # Vector candidates from TF-IDF cosine similarity
        vector_scores: Dict[int, float] = {}
        vector_meta: Dict[int, Dict[str, Any]] = {}

        if self._vectorizer is not None and self._tfidf_matrix is not None and self._index_entry_ids:
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()

            # Top-N indices by similarity
            top_indices = sims.argsort()[::-1][:vector_candidate_limit]
            for idx in top_indices:
                entry_id = self._index_entry_ids[int(idx)]
                meta = self._index_entries_meta.get(entry_id)
                if not meta:
                    continue

                # Filter category/min credibility and expiration in Python (for correctness)
                if category and meta.get("category") != category:
                    continue
                if meta.get("credibility_score", 0) < min_credibility:
                    continue
                exp_dt = self._parse_expiration_date(meta.get("expiration_date"))
                if exp_dt is not None and exp_dt <= now:
                    continue

                sim_score = float(sims[int(idx)])
                vector_scores[entry_id] = sim_score
                vector_meta[entry_id] = {
                    **{k: v for k, v in meta.items() if k != "expiration_date"},
                    "expiration_date": meta.get("expiration_date"),
                    "vector_score": sim_score,
                }

        # Merge candidates: union
        candidate_ids = set(keyword_scores.keys()) | set(vector_scores.keys())
        if not candidate_ids:
            return []

        # If candidates too many, keep the most promising by either score
        # (simple prune to cap work)
        if len(candidate_ids) > max_vector_candidates_merge:
            scored = []
            for cid in candidate_ids:
                scored.append((vector_scores.get(cid, 0.0), keyword_scores.get(cid, 0.0), cid))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            candidate_ids = {cid for _, _, cid in scored[:max_vector_candidates_merge]}

        # Normalize scores within candidates
        max_vec = max(vector_scores.get(cid, 0.0) for cid in candidate_ids) or 1e-9
        max_kw = max(keyword_scores.get(cid, 0.0) for cid in candidate_ids) or 1e-9

        ranked: List[Dict[str, Any]] = []
        for cid in candidate_ids:
            kw_meta = keyword_meta.get(cid)
            v_meta = vector_meta.get(cid)
            meta = kw_meta or v_meta
            if not meta:
                continue

            # Exclude expired just in case
            exp_dt = self._parse_expiration_date(meta.get("expiration_date"))
            if exp_dt is not None and exp_dt <= now:
                continue
            if category and meta.get("category") != category:
                continue
            if meta.get("credibility_score", 0) < min_credibility:
                continue

            vec_score = vector_scores.get(cid, 0.0)
            kw_score = keyword_scores.get(cid, 0.0)

            vec_norm = vec_score / max_vec if max_vec > 0 else 0.0
            kw_norm = kw_score / max_kw if max_kw > 0 else 0.0
            combined = alpha * vec_norm + (1.0 - alpha) * kw_norm

            entry_out = {
                "id": meta["id"],
                "category": meta["category"],
                "title": meta["title"],
                "content": meta["content"],
                "source": meta["source"],
                "credibility_score": meta["credibility_score"],
                "last_updated": meta["last_updated"],
                "tags": meta.get("tags", []),
                "combined_score": combined,
                "vector_score": vec_score,
                "keyword_score": kw_score,
            }
            ranked.append(entry_out)

        ranked.sort(key=lambda x: (x["combined_score"], str(x.get("last_updated", ""))), reverse=True)
        return ranked[:top_k]
    
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