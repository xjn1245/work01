from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Optional

from livetalking.server.auth import generate_salt, hash_password_pbkdf2_sha256
import hmac


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE,
  role TEXT NOT NULL, -- 'admin' | 'student'
  salt TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  iterations INTEGER NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  last_login_at_ms INTEGER,
  student_id TEXT UNIQUE,
  real_name TEXT,
  gender TEXT,
  college TEXT,
  major TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
"""


class AuthStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            # lightweight migrations for older DBs
            cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
            migrations = [
                ("last_login_at_ms", "ALTER TABLE users ADD COLUMN last_login_at_ms INTEGER"),
                ("student_id", "ALTER TABLE users ADD COLUMN student_id TEXT"),
                ("real_name", "ALTER TABLE users ADD COLUMN real_name TEXT"),
                ("gender", "ALTER TABLE users ADD COLUMN gender TEXT"),
                ("college", "ALTER TABLE users ADD COLUMN college TEXT"),
                ("major", "ALTER TABLE users ADD COLUMN major TEXT"),
            ]
            for col, sql in migrations:
                if col not in cols:
                    conn.execute(sql)
            # Create indexes after all migrations to avoid missing-column failures.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_student_id ON users(student_id)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_users_student_id ON users(student_id)")
            conn.commit()

    def ensure_user(self, username: str, password: str, role: str, iterations: int = 200_000) -> None:
        username = str(username).strip()
        role = str(role).strip()
        if not username or not password or role not in ("admin", "student"):
            return
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            if row:
                return
            salt = generate_salt()
            pw_hash = hash_password_pbkdf2_sha256(password=password, salt=salt, iterations=int(iterations))
            conn.execute(
                "INSERT INTO users(username, role, salt, password_hash, iterations, enabled) VALUES (?, ?, ?, ?, ?, 1)",
                (username, role, salt, pw_hash, int(iterations)),
            )
            conn.commit()

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        username = str(username).strip()
        if not username:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, role, salt, password_hash, iterations, enabled FROM users WHERE username=?",
                (username,),
            ).fetchone()
        if not row:
            return None
        return {
            "username": row[0],
            "role": row[1],
            "salt": row[2],
            "password_hash": row[3],
            "iterations": row[4],
            "enabled": bool(row[5]),
        }

    def verify_login(self, username: str, password: str, expected_role: str) -> bool:
        user = self.get_user(username)
        if not user:
            return False
        if not user.get("enabled", False):
            return False
        if user.get("role") != expected_role:
            return False
        salt = str(user.get("salt", ""))
        iterations = int(user.get("iterations", 200_000))
        expected = str(user.get("password_hash", ""))
        got = hash_password_pbkdf2_sha256(password=password, salt=salt, iterations=iterations)
        ok = hmac.compare_digest(got, expected)
        if ok:
            import time

            with self._connect() as conn:
                conn.execute(
                    "UPDATE users SET last_login_at_ms=? WHERE username=?",
                    (int(time.time() * 1000), str(username)),
                )
                conn.commit()
        return ok

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        user = self.get_user(username)
        if not user:
            return False
        if not user.get("enabled", False):
            return False
        salt = str(user.get("salt", ""))
        iterations = int(user.get("iterations", 200_000))
        expected = str(user.get("password_hash", ""))
        got = hash_password_pbkdf2_sha256(password=old_password, salt=salt, iterations=iterations)
        if not hmac.compare_digest(got, expected):
            return False

        new_salt = generate_salt()
        new_hash = hash_password_pbkdf2_sha256(password=new_password, salt=new_salt, iterations=iterations)
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET salt=?, password_hash=? WHERE username=?",
                (new_salt, new_hash, username),
            )
            conn.commit()
        return True

    def student_register(
        self,
        student_id: str,
        real_name: str,
        gender: str,
        college: str,
        major: str,
        username: str,
        password: str,
        enabled: bool = True,
        iterations: int = 200_000,
    ) -> bool:
        student_id = str(student_id).strip()
        username = str(username).strip()
        if not student_id or not username or not password:
            return False
        if self.get_user(username):
            return False
        with self._connect() as conn:
            exists_sid = conn.execute("SELECT id FROM users WHERE student_id=?", (student_id,)).fetchone()
            if exists_sid:
                return False
            salt = generate_salt()
            pw_hash = hash_password_pbkdf2_sha256(password=password, salt=salt, iterations=int(iterations))
            conn.execute(
                """
                INSERT INTO users(username, role, salt, password_hash, iterations, enabled, student_id, real_name, gender, college, major)
                VALUES (?, 'student', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    salt,
                    pw_hash,
                    int(iterations),
                    1 if enabled else 0,
                    student_id,
                    str(real_name or "").strip(),
                    str(gender or "").strip(),
                    str(college or "").strip(),
                    str(major or "").strip(),
                ),
            )
            conn.commit()
        return True

    def list_students(
        self,
        student_id: str = "",
        real_name: str = "",
        college: str = "",
        major: str = "",
        status: str = "",
        page: int = 0,
        page_size: int = 0,
    ) -> list[Dict[str, Any]]:
        conds = ["role='student'"]
        params: list[Any] = []
        if student_id:
            conds.append("student_id LIKE ?")
            params.append(f"%{student_id.strip()}%")
        if real_name:
            conds.append("real_name LIKE ?")
            params.append(f"%{real_name.strip()}%")
        if college:
            conds.append("college LIKE ?")
            params.append(f"%{college.strip()}%")
        if major:
            conds.append("major LIKE ?")
            params.append(f"%{major.strip()}%")
        if status in ("enabled", "disabled"):
            conds.append("enabled=?")
            params.append(1 if status == "enabled" else 0)
        where_sql = " AND ".join(conds)
        sql = f"""
            SELECT id, student_id, real_name, gender, college, major, username, enabled, created_at_ms, last_login_at_ms
            FROM users
            WHERE {where_sql}
            ORDER BY id DESC
        """
        if int(page) > 0 and int(page_size) > 0:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(page_size), (int(page) - 1) * int(page_size)])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r[0]),
                    "student_id": r[1] or "",
                    "real_name": r[2] or "",
                    "gender": r[3] or "",
                    "college": r[4] or "",
                    "major": r[5] or "",
                    "username": r[6] or "",
                    "enabled": bool(r[7]),
                    "status": "正常" if int(r[7] or 0) == 1 else "禁用",
                    "created_at_ms": r[8],
                    "last_login_at_ms": r[9],
                }
            )
        return out

    def count_students(self, student_id: str = "", real_name: str = "", college: str = "", major: str = "", status: str = "") -> int:
        conds = ["role='student'"]
        params: list[Any] = []
        if student_id:
            conds.append("student_id LIKE ?")
            params.append(f"%{student_id.strip()}%")
        if real_name:
            conds.append("real_name LIKE ?")
            params.append(f"%{real_name.strip()}%")
        if college:
            conds.append("college LIKE ?")
            params.append(f"%{college.strip()}%")
        if major:
            conds.append("major LIKE ?")
            params.append(f"%{major.strip()}%")
        if status in ("enabled", "disabled"):
            conds.append("enabled=?")
            params.append(1 if status == "enabled" else 0)
        where_sql = " AND ".join(conds)
        with self._connect() as conn:
            n = conn.execute(f"SELECT COUNT(*) FROM users WHERE {where_sql}", params).fetchone()[0] or 0
        return int(n)

    def get_student_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, student_id, real_name, gender, college, major, username, enabled, created_at_ms, last_login_at_ms
                FROM users WHERE id=? AND role='student'
                """,
                (int(user_id),),
            ).fetchone()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "student_id": row[1] or "",
            "real_name": row[2] or "",
            "gender": row[3] or "",
            "college": row[4] or "",
            "major": row[5] or "",
            "username": row[6] or "",
            "enabled": bool(row[7]),
            "status": "正常" if int(row[7] or 0) == 1 else "禁用",
            "created_at_ms": row[8],
            "last_login_at_ms": row[9],
        }

    def update_student(
        self,
        user_id: int,
        student_id: str,
        real_name: str,
        gender: str,
        college: str,
        major: str,
        username: str,
        enabled: bool,
    ) -> bool:
        user_id = int(user_id)
        student_id = str(student_id).strip()
        username = str(username).strip()
        if not student_id or not username:
            return False
        with self._connect() as conn:
            dup_sid = conn.execute("SELECT id FROM users WHERE student_id=? AND id<>?", (student_id, user_id)).fetchone()
            if dup_sid:
                return False
            dup_user = conn.execute("SELECT id FROM users WHERE username=? AND id<>?", (username, user_id)).fetchone()
            if dup_user:
                return False
            conn.execute(
                """
                UPDATE users
                SET student_id=?, real_name=?, gender=?, college=?, major=?, username=?, enabled=?
                WHERE id=? AND role='student'
                """,
                (
                    student_id,
                    str(real_name or "").strip(),
                    str(gender or "").strip(),
                    str(college or "").strip(),
                    str(major or "").strip(),
                    username,
                    1 if enabled else 0,
                    user_id,
                ),
            )
            conn.commit()
        return True

    def reset_password(self, user_id: int, new_password: str) -> bool:
        user = self.get_student_by_id(int(user_id))
        if not user:
            return False
        new_salt = generate_salt()
        iterations = 200_000
        new_hash = hash_password_pbkdf2_sha256(password=new_password, salt=new_salt, iterations=iterations)
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET salt=?, password_hash=?, iterations=? WHERE id=?",
                (new_salt, new_hash, iterations, int(user_id)),
            )
            conn.commit()
        return True

    def set_user_enabled(self, user_id: int, enabled: bool) -> bool:
        with self._connect() as conn:
            conn.execute("UPDATE users SET enabled=? WHERE id=? AND role='student'", (1 if enabled else 0, int(user_id)))
            conn.commit()
        return True

    def delete_student(self, user_id: int) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM users WHERE id=? AND role='student'", (int(user_id),))
            conn.commit()
        return True

