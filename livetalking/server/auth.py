from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def hash_password_pbkdf2_sha256(
    password: str,
    salt: str,
    iterations: int = 200_000,
) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return dk.hex()


def generate_salt() -> str:
    # Enough for local demo purposes.
    return os.urandom(16).hex()


def ensure_users_file(
    users_file: str,
    default_users: list[dict],
) -> None:
    """
    Ensure a users JSON exists.
    Format:
      {
        "users": [
          {"username":"admin","role":"admin","salt":"...","password_hash":"...","iterations":200000}
        ]
      }
    """
    if os.path.exists(users_file):
        return

    parent = os.path.dirname(users_file)
    if parent:
        os.makedirs(parent, exist_ok=True)

    users_out: list[dict] = []
    for u in default_users:
        username = str(u.get("username", "")).strip()
        role = str(u.get("role", "")).strip()
        password = str(u.get("password", "")).strip()
        if not username or not role or not password:
            continue
        salt = generate_salt()
        iterations = int(u.get("iterations", 200_000))
        password_hash = hash_password_pbkdf2_sha256(password=password, salt=salt, iterations=iterations)
        users_out.append(
            {
                "username": username,
                "role": role,
                "salt": salt,
                "password_hash": password_hash,
                "iterations": iterations,
            }
        )

    with open(users_file, "w", encoding="utf-8") as f:
        json.dump({"users": users_out}, f, ensure_ascii=False, indent=2)


def load_users(users_file: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(users_file):
        return {}
    with open(users_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    users = {}
    for item in data.get("users", []):
        username = str(item.get("username", "")).strip()
        if not username:
            continue
        users[username] = item
    return users


def verify_password_pbkdf2_sha256(user: Dict[str, Any], password: str) -> bool:
    try:
        salt = str(user.get("salt", ""))
        expected = str(user.get("password_hash", ""))
        iterations = int(user.get("iterations", 200_000))
        if not salt or not expected:
            return False
        got = hash_password_pbkdf2_sha256(password=password, salt=salt, iterations=iterations)
        # constant-time compare
        return hmac.compare_digest(got, expected)
    except Exception:
        return False


def create_token(payload: Dict[str, Any], token_secret: str, ttl_seconds: int) -> str:
    now = int(time.time())
    payload2 = dict(payload)
    payload2["iat"] = now
    payload2["exp"] = now + int(ttl_seconds)

    payload_json = json.dumps(payload2, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)
    sig = hmac.new(token_secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{payload_b64}.{sig_b64}"


def decode_token(token: str, token_secret: str) -> Optional[Dict[str, Any]]:
    try:
        if not token or "." not in token:
            return None
        payload_b64, sig_b64 = token.split(".", 1)
        expected_sig = hmac.new(token_secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_encode(expected_sig), sig_b64):
            return None
        payload_json = _b64url_decode(payload_b64)
        payload = json.loads(payload_json.decode("utf-8"))
        now = int(time.time())
        if int(payload.get("exp", 0)) < now:
            return None
        return payload
    except Exception:
        return None


def get_bearer_token_from_auth_header(auth_header: str | None) -> str:
    if not auth_header:
        return ""
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return ""

