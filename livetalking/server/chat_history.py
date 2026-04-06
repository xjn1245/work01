from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trace_id TEXT NOT NULL,
  sessionid INTEGER NOT NULL,
  user_id TEXT,
  avatar_id TEXT,
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER,
  ended_at_ms INTEGER,
  status TEXT,
  title TEXT,
  dt_ms INTEGER,
  rag_enabled INTEGER,
  rag_mode TEXT,
  rag_hit_count INTEGER,
  llm_timeout INTEGER DEFAULT 0,
  llm_ms INTEGER,
  tts_ms INTEGER,
  action_ms INTEGER
  ,
  rag_evidence_json TEXT,
  satisfaction INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at_ms);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_session_id INTEGER NOT NULL,
  role TEXT NOT NULL, -- 'user' | 'assistant'
  content TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL,
  FOREIGN KEY(chat_session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(chat_session_id);
"""


class ChatHistoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            # lightweight migrations for old DBs
            cols = [r[1] for r in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()]
            if "updated_at_ms" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN updated_at_ms INTEGER")
            if "ended_at_ms" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN ended_at_ms INTEGER")
            if "status" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN status TEXT")
            if "title" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN title TEXT")
            if "llm_ms" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN llm_ms INTEGER")
            if "tts_ms" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN tts_ms INTEGER")
            if "action_ms" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN action_ms INTEGER")
            if "rag_evidence_json" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN rag_evidence_json TEXT")
            if "satisfaction" not in cols:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN satisfaction INTEGER")
            # backfill defaults (best-effort)
            now_ms = int(time.time() * 1000)
            conn.execute("UPDATE chat_sessions SET status=COALESCE(status,'active') WHERE status IS NULL OR status=''")
            conn.execute("UPDATE chat_sessions SET updated_at_ms=COALESCE(updated_at_ms, created_at_ms, ?) WHERE updated_at_ms IS NULL", (now_ms,))
            conn.commit()

    def create_chat_session(
        self,
        trace_id: str,
        sessionid: int,
        user_id: str,
        avatar_id: str,
        rag_enabled: bool,
        rag_mode: str,
        rag_hit_count: int,
        rag_evidence_json: str = "",
    ) -> int:
        now_ms = int(time.time() * 1000)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chat_sessions(trace_id, sessionid, user_id, avatar_id, created_at_ms, updated_at_ms, status, title, rag_enabled, rag_mode, rag_hit_count, rag_evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    int(sessionid),
                    user_id,
                    avatar_id,
                    now_ms,
                    now_ms,
                    "active",
                    "",
                    1 if rag_enabled else 0,
                    rag_mode,
                    int(rag_hit_count),
                    rag_evidence_json or "",
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def create_empty_session(self, trace_id: str, sessionid: int, user_id: str, avatar_id: str) -> int:
        """Create a conversation shell before the first message."""
        now_ms = int(time.time() * 1000)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chat_sessions(trace_id, sessionid, user_id, avatar_id, created_at_ms, updated_at_ms, status, title, rag_enabled, rag_mode, rag_hit_count, rag_evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, '', 0, '')
                """,
                (trace_id, int(sessionid), user_id, avatar_id, now_ms, now_ms, "active", ""),
            )
            conn.commit()
            return int(cur.lastrowid)

    def add_message(self, chat_session_id: int, role: str, content: str) -> None:
        now_ms = int(time.time() * 1000)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages(chat_session_id, role, content, created_at_ms) VALUES (?, ?, ?, ?)",
                (int(chat_session_id), str(role), str(content), now_ms),
            )
            conn.execute("UPDATE chat_sessions SET updated_at_ms=? WHERE id=?", (now_ms, int(chat_session_id)))
            conn.commit()

    def finish_chat_session(
        self,
        chat_session_id: int,
        dt_ms: int,
        llm_timeout: bool,
        llm_ms: Optional[int] = None,
        tts_ms: Optional[int] = None,
        action_ms: Optional[int] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET dt_ms=?, llm_timeout=?, llm_ms=?, tts_ms=?, action_ms=?, updated_at_ms=? WHERE id=?",
                (
                    int(dt_ms),
                    1 if llm_timeout else 0,
                    int(llm_ms) if llm_ms is not None else None,
                    int(tts_ms) if tts_ms is not None else None,
                    int(action_ms) if action_ms is not None else None,
                    int(time.time() * 1000),
                    int(chat_session_id),
                ),
            )
            conn.commit()

    def list_sessions(
        self,
        start_ms: Optional[int],
        end_ms: Optional[int],
        user_id: str,
        user_id_like: str,
        keyword: str,
        page: int,
        page_size: int,
    ) -> Dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(200, int(page_size)))
        offset = (page - 1) * page_size

        where = []
        args: List[Any] = []

        if start_ms is not None:
            where.append("created_at_ms >= ?")
            args.append(int(start_ms))
        if end_ms is not None:
            where.append("created_at_ms <= ?")
            args.append(int(end_ms))
        user_id = (user_id or "").strip()
        user_id_like = (user_id_like or "").strip()
        if user_id:
            where.append("user_id = ?")
            args.append(user_id)
        elif user_id_like:
            where.append("user_id LIKE ?")
            args.append(f"%{user_id_like}%")

        # keyword search over message content (simple LIKE)
        keyword = (keyword or "").strip()
        if keyword:
            where.append(
                "id IN (SELECT chat_session_id FROM chat_messages WHERE content LIKE ?)"
            )
            args.append(f"%{keyword}%")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM chat_sessions {where_sql}", args).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT id, trace_id, sessionid, user_id, avatar_id, created_at_ms, updated_at_ms, ended_at_ms, status, title,
                       dt_ms, rag_enabled, rag_mode, rag_hit_count, llm_timeout, llm_ms, tts_ms, action_ms, rag_evidence_json, satisfaction
                FROM chat_sessions
                {where_sql}
                ORDER BY created_at_ms DESC
                LIMIT ? OFFSET ?
                """,
                args + [page_size, offset],
            ).fetchall()

        data = []
        with self._connect() as conn:
            for r in rows:
                sid = int(r[0])
                title = (r[9] or "").strip()
                q = conn.execute(
                    "SELECT content FROM chat_messages WHERE chat_session_id=? AND role='user' ORDER BY id ASC LIMIT 1",
                    (sid,),
                ).fetchone()
                a = conn.execute(
                    "SELECT content FROM chat_messages WHERE chat_session_id=? AND role='assistant' ORDER BY id ASC LIMIT 1",
                    (sid,),
                ).fetchone()
                data.append(
                    {
                        "id": sid,
                        "trace_id": r[1],
                        "sessionid": r[2],
                        "user_id": r[3],
                        "avatar_id": r[4],
                        "created_at_ms": r[5],
                        "updated_at_ms": r[6],
                        "ended_at_ms": r[7],
                        "status": r[8] or "active",
                        "title": title,
                        "dt_ms": r[10],
                        "rag_enabled": bool(r[11]),
                        "rag_mode": r[12],
                        "rag_hit_count": r[13],
                        "llm_timeout": bool(r[14]),
                        "llm_ms": r[15],
                        "tts_ms": r[16],
                        "action_ms": r[17],
                        "rag_evidence": r[18] if len(r) > 18 else "",
                        "satisfaction": r[19] if len(r) > 19 else None,
                        "question_summary": ((title or (q[0] if q else "")) or "")[:120],
                        "answer_summary": (a[0] if a else "")[:120],
                    }
                )

        return {
            "page": page,
            "page_size": page_size,
            "total": int(total),
            "items": data,
        }

    def get_session_detail(self, chat_session_id: int) -> Dict[str, Any]:
        with self._connect() as conn:
            s = conn.execute(
                """
                SELECT id, trace_id, sessionid, user_id, avatar_id, created_at_ms, updated_at_ms, ended_at_ms, status, title,
                       dt_ms, rag_enabled, rag_mode, rag_hit_count, llm_timeout, llm_ms, tts_ms, action_ms, rag_evidence_json, satisfaction
                FROM chat_sessions WHERE id=?
                """,
                (int(chat_session_id),),
            ).fetchone()
            if not s:
                raise KeyError("not found")
            msgs = conn.execute(
                """
                SELECT role, content, created_at_ms
                FROM chat_messages
                WHERE chat_session_id=?
                ORDER BY id ASC
                """,
                (int(chat_session_id),),
            ).fetchall()

        return {
            "session": {
                "id": s[0],
                "trace_id": s[1],
                "sessionid": s[2],
                "user_id": s[3],
                "avatar_id": s[4],
                "created_at_ms": s[5],
                "updated_at_ms": s[6],
                "ended_at_ms": s[7],
                "status": s[8] or "active",
                "title": (s[9] or "").strip(),
                "dt_ms": s[10],
                "rag_enabled": bool(s[11]),
                "rag_mode": s[12],
                "rag_hit_count": s[13],
                "llm_timeout": bool(s[14]),
                "llm_ms": s[15],
                "tts_ms": s[16],
                "action_ms": s[17],
                "rag_evidence_json": s[18] if len(s) > 18 else "",
                "satisfaction": s[19] if len(s) > 19 else None,
            },
            "messages": [{"role": m[0], "content": m[1], "created_at_ms": m[2]} for m in msgs],
        }

    def delete_session(self, chat_session_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE chat_session_id=?", (int(chat_session_id),))
            conn.execute("DELETE FROM chat_sessions WHERE id=?", (int(chat_session_id),))
            conn.commit()

    def end_session(self, chat_session_id: int) -> bool:
        now_ms = int(time.time() * 1000)
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM chat_sessions WHERE id=?", (int(chat_session_id),)).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE chat_sessions SET status='ended', ended_at_ms=?, updated_at_ms=? WHERE id=?",
                (now_ms, now_ms, int(chat_session_id)),
            )
            conn.commit()
        return True

    def rename_session(self, chat_session_id: int, new_title: str) -> bool:
        title = str(new_title or "").strip()
        if not title:
            return False
        with self._connect() as conn:
            conn.execute("UPDATE chat_sessions SET title=?, updated_at_ms=? WHERE id=?", (title, int(time.time() * 1000), int(chat_session_id)))
            conn.commit()
        return True

    def analytics_overview(self) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        day_start_ms = now_ms - (now_ms % (24 * 3600 * 1000))
        with self._connect() as conn:
            today_sessions = conn.execute(
                "SELECT COUNT(*) FROM chat_sessions WHERE created_at_ms >= ?",
                (int(day_start_ms),),
            ).fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]
            total_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM chat_sessions WHERE user_id IS NOT NULL AND user_id != ''").fetchone()[0]
            avg_dt = conn.execute("SELECT AVG(dt_ms) FROM chat_sessions WHERE dt_ms IS NOT NULL").fetchone()[0]
            avg_sat = conn.execute("SELECT AVG(satisfaction) FROM chat_sessions WHERE satisfaction IS NOT NULL").fetchone()[0]
        return {
            "today_sessions": int(today_sessions),
            "total_sessions": int(total_sessions),
            "total_users": int(total_users),
            "avg_dt_ms": int(avg_dt) if avg_dt is not None else None,
            "avg_satisfaction": round(float(avg_sat), 2) if avg_sat is not None else None,
        }

    def set_satisfaction(self, chat_session_id: int, score: int) -> None:
        s = max(1, min(5, int(score)))
        with self._connect() as conn:
            conn.execute("UPDATE chat_sessions SET satisfaction=? WHERE id=?", (s, int(chat_session_id)))
            conn.commit()

