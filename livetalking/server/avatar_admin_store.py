from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS avatar_profiles (
  avatar_id TEXT PRIMARY KEY,
  name TEXT,
  model_type TEXT,
  identity_type TEXT,
  identity_desc TEXT,
  enabled INTEGER NOT NULL DEFAULT 1,
  updated_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
);

CREATE TABLE IF NOT EXISTS avatar_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  avatar_id TEXT NOT NULL,
  action_key TEXT NOT NULL,
  label TEXT,
  audiotype INTEGER NOT NULL,
  sort_order INTEGER NOT NULL DEFAULT 0,
  note TEXT,
  enabled INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_avatar_actions_avatar ON avatar_actions(avatar_id, sort_order);

CREATE TABLE IF NOT EXISTS avatar_tts_configs (
  avatar_id TEXT PRIMARY KEY,
  speed REAL,
  tone REAL,
  voice TEXT,
  keyword_pron TEXT,
  updated_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
);

CREATE TABLE IF NOT EXISTS tts_allowed_voices (
  voice TEXT PRIMARY KEY,
  label TEXT,
  enabled INTEGER NOT NULL DEFAULT 1,
  sort_order INTEGER NOT NULL DEFAULT 0,
  updated_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  locale_tag TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS avatar_tts_locale (
  avatar_id TEXT NOT NULL,
  ui_lang TEXT NOT NULL,
  voice TEXT NOT NULL DEFAULT '',
  updated_at_ms INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (avatar_id, ui_lang)
);
CREATE INDEX IF NOT EXISTS idx_avatar_tts_locale_avatar ON avatar_tts_locale(avatar_id);
"""


def _now_ms_sql() -> str:
    return "(CAST(strftime('%s','now') AS INTEGER) * 1000)"


class AvatarAdminStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _migrate_legacy_columns(self) -> None:
        """旧库补列 / 新表（CREATE IF NOT EXISTS 对已存在表不生效时）。"""
        with self._connect() as conn:
            rows = conn.execute("PRAGMA table_info(tts_allowed_voices)").fetchall()
            cols = {str(r[1]) for r in rows}
            if cols and "locale_tag" not in cols:
                conn.execute("ALTER TABLE tts_allowed_voices ADD COLUMN locale_tag TEXT NOT NULL DEFAULT ''")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS avatar_tts_locale (
                  avatar_id TEXT NOT NULL,
                  ui_lang TEXT NOT NULL,
                  voice TEXT NOT NULL DEFAULT '',
                  updated_at_ms INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY (avatar_id, ui_lang)
                );
                CREATE INDEX IF NOT EXISTS idx_avatar_tts_locale_avatar ON avatar_tts_locale(avatar_id);
                """
            )
            conn.commit()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        self._migrate_legacy_columns()
        with self._connect() as conn:
            # seed default voices (idempotent)
            default_voices = [
                ("zh-CN-YunxiaNeural", "中文女声 Yunxia", 1, 0, "zh-CN"),
                ("zh-CN-XiaoxiaoNeural", "中文女声 Xiaoxiao", 1, 1, "zh-CN"),
                ("zh-CN-YunjianNeural", "中文男声 Yunjian", 1, 2, "zh-CN"),
                ("zh-CN-YunyangNeural", "中文男声 Yunyang", 1, 3, "zh-CN"),
                ("zh-CN-YunxiNeural", "中文男声 Yunxi", 1, 4, "zh-CN"),
                ("zh-CN-YunhaoNeural", "中文男声 Yunhao", 1, 5, "zh-CN"),
                ("ja-JP-NanamiNeural", "日语女声 Nanami", 1, 10, "ja"),
                ("ja-JP-KeitaNeural", "日语男声 Keita", 1, 11, "ja"),
                ("ko-KR-SunHiNeural", "韩语女声 SunHi", 1, 20, "ko"),
                ("ko-KR-InJoonNeural", "韩语男声 InJoon", 1, 21, "ko"),
                ("ko-KR-HyunsuNeural", "韩语男声 Hyunsu", 1, 22, "ko"),
                ("en-US-JennyNeural", "英语女声 Jenny", 1, 30, "en"),
                ("en-US-AriaNeural", "英语女声 Aria", 1, 31, "en"),
                ("en-US-GuyNeural", "英语男声 Guy", 1, 32, "en"),
                ("en-US-ChristopherNeural", "英语男声 Christopher", 1, 33, "en"),
            ]
            for voice, label, enabled, sort_order, locale_tag in default_voices:
                conn.execute(
                    f"""
                    INSERT INTO tts_allowed_voices(voice, label, enabled, sort_order, updated_at_ms, locale_tag)
                    VALUES (?, ?, ?, ?, {_now_ms_sql()}, ?)
                    ON CONFLICT(voice) DO NOTHING
                    """,
                    (voice, label, enabled, sort_order, locale_tag),
                )
            conn.commit()

    def ensure_avatar(self, avatar_id: str, name: str, model_type: str) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT avatar_id FROM avatar_profiles WHERE avatar_id=?", (avatar_id,)).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO avatar_profiles(avatar_id, name, model_type, enabled) VALUES (?, ?, ?, 1)",
                    (avatar_id, name, model_type),
                )
                conn.commit()

    def list_profiles(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT avatar_id, name, model_type, identity_type, identity_desc, enabled, updated_at_ms FROM avatar_profiles ORDER BY avatar_id ASC"
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "avatar_id": r[0],
                    "name": r[1] or r[0],
                    "model_type": r[2] or "",
                    "identity_type": r[3] or "",
                    "identity_desc": r[4] or "",
                    "enabled": bool(r[5]),
                    "updated_at_ms": r[6],
                }
            )
        return out

    def upsert_profile(
        self,
        avatar_id: str,
        name: str,
        model_type: str,
        identity_type: str,
        identity_desc: str,
        enabled: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO avatar_profiles(avatar_id, name, model_type, identity_type, identity_desc, enabled, updated_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, {_now_ms_sql()})
                ON CONFLICT(avatar_id) DO UPDATE SET
                  name=excluded.name,
                  model_type=excluded.model_type,
                  identity_type=excluded.identity_type,
                  identity_desc=excluded.identity_desc,
                  enabled=excluded.enabled,
                  updated_at_ms={_now_ms_sql()}
                """,
                (avatar_id, name, model_type, identity_type, identity_desc, 1 if enabled else 0),
            )
            conn.commit()

    def delete_profile(self, avatar_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM avatar_profiles WHERE avatar_id=?", (avatar_id,))
            conn.execute("DELETE FROM avatar_actions WHERE avatar_id=?", (avatar_id,))
            conn.execute("DELETE FROM avatar_tts_configs WHERE avatar_id=?", (avatar_id,))
            conn.execute("DELETE FROM avatar_tts_locale WHERE avatar_id=?", (avatar_id,))
            conn.commit()

    def copy_profile(self, source_avatar_id: str, target_avatar_id: str, target_name: str) -> None:
        with self._connect() as conn:
            src = conn.execute(
                "SELECT name, model_type, identity_type, identity_desc, enabled FROM avatar_profiles WHERE avatar_id=?",
                (source_avatar_id,),
            ).fetchone()
            if not src:
                raise KeyError("source avatar not found")
            conn.execute(
                f"""
                INSERT INTO avatar_profiles(avatar_id, name, model_type, identity_type, identity_desc, enabled, updated_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, {_now_ms_sql()})
                ON CONFLICT(avatar_id) DO UPDATE SET
                  name=excluded.name,
                  model_type=excluded.model_type,
                  identity_type=excluded.identity_type,
                  identity_desc=excluded.identity_desc,
                  enabled=excluded.enabled,
                  updated_at_ms={_now_ms_sql()}
                """,
                (target_avatar_id, target_name or src[0], src[1], src[2], src[3], src[4]),
            )

            actions = conn.execute(
                "SELECT action_key, label, audiotype, sort_order, note, enabled FROM avatar_actions WHERE avatar_id=? ORDER BY sort_order ASC",
                (source_avatar_id,),
            ).fetchall()
            conn.execute("DELETE FROM avatar_actions WHERE avatar_id=?", (target_avatar_id,))
            for a in actions:
                conn.execute(
                    """
                    INSERT INTO avatar_actions(avatar_id, action_key, label, audiotype, sort_order, note, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (target_avatar_id, a[0], a[1], a[2], a[3], a[4], a[5]),
                )

            tts = conn.execute(
                "SELECT speed, tone, voice, keyword_pron FROM avatar_tts_configs WHERE avatar_id=?",
                (source_avatar_id,),
            ).fetchone()
            if tts:
                conn.execute(
                    f"""
                    INSERT INTO avatar_tts_configs(avatar_id, speed, tone, voice, keyword_pron, updated_at_ms)
                    VALUES (?, ?, ?, ?, ?, {_now_ms_sql()})
                    ON CONFLICT(avatar_id) DO UPDATE SET
                      speed=excluded.speed, tone=excluded.tone, voice=excluded.voice, keyword_pron=excluded.keyword_pron, updated_at_ms={_now_ms_sql()}
                    """,
                    (target_avatar_id, tts[0], tts[1], tts[2], tts[3]),
                )
            locales = conn.execute(
                "SELECT ui_lang, voice FROM avatar_tts_locale WHERE avatar_id=?",
                (source_avatar_id,),
            ).fetchall()
            conn.execute("DELETE FROM avatar_tts_locale WHERE avatar_id=?", (target_avatar_id,))
            for lr in locales:
                conn.execute(
                    f"""
                    INSERT INTO avatar_tts_locale(avatar_id, ui_lang, voice, updated_at_ms)
                    VALUES (?, ?, ?, {_now_ms_sql()})
                    """,
                    (target_avatar_id, lr[0], lr[1]),
                )
            conn.commit()

    def get_actions(self, avatar_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT action_key, label, audiotype, sort_order, note, enabled
                FROM avatar_actions
                WHERE avatar_id=?
                ORDER BY sort_order ASC, id ASC
                """,
                (avatar_id,),
            ).fetchall()
        return [
            {
                "key": r[0],
                "label": r[1],
                "audiotype": int(r[2]),
                "sort_order": int(r[3]),
                "note": r[4] or "",
                "enabled": bool(r[5]),
            }
            for r in rows
        ]

    def save_actions(self, avatar_id: str, actions: List[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM avatar_actions WHERE avatar_id=?", (avatar_id,))
            for idx, a in enumerate(actions):
                conn.execute(
                    """
                    INSERT INTO avatar_actions(avatar_id, action_key, label, audiotype, sort_order, note, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        avatar_id,
                        str(a.get("key", f"action_{idx}")),
                        str(a.get("label", "")),
                        int(a.get("audiotype", 0)),
                        int(a.get("sort_order", idx)),
                        str(a.get("note", "")),
                        1 if bool(a.get("enabled", True)) else 0,
                    ),
                )
            conn.commit()

    def get_tts(self, avatar_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT speed, tone, voice, keyword_pron, updated_at_ms FROM avatar_tts_configs WHERE avatar_id=?",
                (avatar_id,),
            ).fetchone()
            locale_rows = conn.execute(
                "SELECT ui_lang, voice FROM avatar_tts_locale WHERE avatar_id=?",
                (avatar_id,),
            ).fetchall()
        voices_by_lang: Dict[str, str] = {}
        for lr in locale_rows:
            lk = str(lr[0] or "").strip()
            vv = str(lr[1] or "").strip()
            if lk and vv:
                voices_by_lang[lk] = vv
        legacy_voice = ""
        if row:
            legacy_voice = str(row[2] or "").strip()
        if "zh-CN" not in voices_by_lang and legacy_voice:
            voices_by_lang["zh-CN"] = legacy_voice
        if not row:
            return {
                "speed": None,
                "tone": None,
                "voice": legacy_voice,
                "keyword_pron": "",
                "updated_at_ms": None,
                "voices_by_lang": voices_by_lang,
            }
        return {
            "speed": row[0],
            "tone": row[1],
            "voice": legacy_voice,
            "keyword_pron": row[3] or "",
            "updated_at_ms": row[4],
            "voices_by_lang": voices_by_lang,
        }

    def save_tts(self, avatar_id: str, speed: Optional[float], tone: Optional[float], voice: str, keyword_pron: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO avatar_tts_configs(avatar_id, speed, tone, voice, keyword_pron, updated_at_ms)
                VALUES (?, ?, ?, ?, ?, {_now_ms_sql()})
                ON CONFLICT(avatar_id) DO UPDATE SET
                  speed=excluded.speed, tone=excluded.tone, voice=excluded.voice, keyword_pron=excluded.keyword_pron, updated_at_ms={_now_ms_sql()}
                """,
                (avatar_id, speed, tone, voice, keyword_pron),
            )
            conn.commit()

    def _enabled_voice_set(self, conn: sqlite3.Connection) -> set:
        rows = conn.execute("SELECT voice FROM tts_allowed_voices WHERE enabled=1").fetchall()
        return {str(r[0]) for r in rows if r and r[0]}

    def save_tts_locales(self, avatar_id: str, voices_by_lang: Dict[str, Any]) -> None:
        with self._connect() as conn:
            allowed = self._enabled_voice_set(conn)
            conn.execute("DELETE FROM avatar_tts_locale WHERE avatar_id=?", (avatar_id,))
            for lang, raw in (voices_by_lang or {}).items():
                lang_k = str(lang or "").strip()
                vv = str(raw or "").strip()
                if not lang_k or not vv:
                    continue
                if vv not in allowed:
                    raise ValueError(f"voice not in enabled allowed list: {vv}")
                conn.execute(
                    f"""
                    INSERT INTO avatar_tts_locale(avatar_id, ui_lang, voice, updated_at_ms)
                    VALUES (?, ?, ?, {_now_ms_sql()})
                    """,
                    (avatar_id, lang_k, vv),
                )
            conn.commit()

    def list_allowed_voices(self, enabled_only: bool = False, locale_pool: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = """
            SELECT voice, label, enabled, sort_order, updated_at_ms, locale_tag
            FROM tts_allowed_voices
            WHERE 1=1
        """
        params: List[Any] = []
        if enabled_only:
            sql += " AND enabled=1"
        lp = str(locale_pool or "").strip()
        if lp:
            sql += " AND (IFNULL(locale_tag,'') = '' OR locale_tag = ?)"
            params.append(lp)
        sql += " ORDER BY sort_order ASC, updated_at_ms ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            {
                "voice": str(r[0] or ""),
                "label": str(r[1] or r[0] or ""),
                "enabled": bool(r[2]),
                "sort_order": int(r[3] or 0),
                "updated_at_ms": r[4],
                "locale_tag": str(r[5] or ""),
            }
            for r in rows
        ]

    def save_allowed_voices(self, items: List[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM tts_allowed_voices")
            for idx, item in enumerate(items or []):
                voice = str((item or {}).get("voice", "")).strip()
                if not voice:
                    continue
                label = str((item or {}).get("label", "")).strip() or voice
                enabled = 1 if bool((item or {}).get("enabled", True)) else 0
                sort_order = int((item or {}).get("sort_order", idx))
                locale_tag = str((item or {}).get("locale_tag", "") or "").strip()
                conn.execute(
                    f"""
                    INSERT INTO tts_allowed_voices(voice, label, enabled, sort_order, updated_at_ms, locale_tag)
                    VALUES (?, ?, ?, ?, {_now_ms_sql()}, ?)
                    """,
                    (voice, label, enabled, sort_order, locale_tag),
                )
            conn.commit()

