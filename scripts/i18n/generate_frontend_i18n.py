import argparse
import hashlib
import json
import os
import re
from pathlib import Path


HTML_EXTS = {".html", ".htm"}
JS_EXTS = {".js", ".mjs", ".cjs"}


def _is_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def _looks_like_ui(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if len(s) < 2:
        return False
    # too short punctuation/noise
    if all(ch in "，。！？：；、（）()[]{}+-/*" or ch.isspace() for ch in s):
        return False
    # avoid huge lines
    if len(s) > 300:
        return False
    return _is_cjk(s)


def _escape_regex(s: str) -> str:
    return re.sub(r"([.^$*+?{}\[\]\\|()])", r"\\\1", s)


def extract_from_html(text: str) -> set[str]:
    # Remove script/style blocks for HTML text extraction.
    stripped = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    stripped = re.sub(r"<style[\s\S]*?</style>", "", stripped, flags=re.I)

    out: set[str] = set()

    # Tag text nodes: >...<
    for m in re.finditer(r">([^<>]+)<", stripped):
        s = m.group(1).strip()
        if _looks_like_ui(s):
            out.add(s)

    # placeholder/title/aria-label attributes
    for attr in ("placeholder", "title", "aria-label"):
        for m in re.finditer(rf'{attr}\s*=\s*"(.*?)"', stripped, flags=re.I | re.S):
            v = m.group(1).strip()
            if _looks_like_ui(v):
                out.add(v)

    # option text
    for m in re.finditer(r"<option[^>]*>([^<]+)</option>", stripped, flags=re.I):
        s = m.group(1).strip()
        if _looks_like_ui(s):
            out.add(s)

    return out


def extract_from_js(text: str) -> set[str]:
    out: set[str] = set()

    # Quick skip obvious minified/vendor JS in caller
    # Extract string literals: "..." / '...' / `...`
    # Template literals may include ${...} (dynamic). We will still extract static HTML text inside.
    for m in re.finditer(r'"([^"\\\n]{1,800})"', text):
        s = m.group(1).strip()
        if "${" in s:
            continue
        if _looks_like_ui(s):
            out.add(s)
    for m in re.finditer(r"'([^'\\\n]{1,800})'", text):
        s = m.group(1).strip()
        if "${" in s:
            continue
        if _looks_like_ui(s):
            out.add(s)
    for m in re.finditer(r"`([^`\\\n]{1,1000})`", text):
        s = m.group(1)
        # If it contains ${...}, strip those expressions, then re-run HTML text extraction.
        if "${" in s:
            s2 = re.sub(r"\$\{[^}]*\}", "", s)
            for m2 in re.finditer(r">([^<>]+)<", s2):
                ss = m2.group(1).strip()
                if _looks_like_ui(ss):
                    out.add(ss)
        else:
            ss = s.strip()
            if _looks_like_ui(ss):
                out.add(ss)

    # Also catch text concatenations like: 'xx'+... is not handled; keep scope small.
    # Extra coverage: extract HTML text nodes from JS templates by removing ${...} expressions first.
    # This handles multiline backtick strings such as:
    #   el.innerHTML = `...<button>查看详情</button>...${id}...`
    try:
        tmp = re.sub(r"\$\{[^}]*\}", "", text)
        for m2 in re.finditer(r">([^<>]+)<", tmp):
            ss = m2.group(1).strip()
            if _looks_like_ui(ss):
                out.add(ss)
    except Exception:
        pass
    return out


def batch(lst: list[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "sourceLang": "zh-CN", "strings": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"version": 1, "sourceLang": "zh-CN", "strings": {}}
    if "strings" not in data or not isinstance(data["strings"], dict):
        data["strings"] = {}
    return data


def sha_key(zh_text: str) -> str:
    # Stable key: sha1 of original zh.
    h = hashlib.sha1(zh_text.encode("utf-8")).hexdigest()[:16]
    return f"k_{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-root", default="web", help="web directory")
    ap.add_argument("--out-json", default="web/i18n/frontend_i18n.json", help="output translation JSON")
    ap.add_argument("--translate", action="store_true", help="call LLM to translate missing strings")
    ap.add_argument("--max-batch", type=int, default=25, help="max texts per LLM request")
    ap.add_argument("--skip-inject", action="store_true", help="do not patch html to load i18n3.js")
    ap.add_argument("--inject-only", action="store_true", help="only patch html to load i18n3.js")
    args = ap.parse_args()

    web_root = Path(args.web_root)
    out_json = Path(args.out_json)

    # 1) extract
    zh_texts: set[str] = set()
    for p in web_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in HTML_EXTS:
            # Skip vendor html (none) and read.
            try:
                s = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            zh_texts |= extract_from_html(s)
            # Also extract from inline scripts inside html files.
            for m in re.finditer(r"<script[^>]*>([\s\S]*?)</script>", s, flags=re.I):
                js = m.group(1)
                zh_texts |= extract_from_js(js)
        elif ext in JS_EXTS:
            if "vendor" in str(p).replace("\\", "/"):
                continue
            try:
                s = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            zh_texts |= extract_from_js(s)

    zh_list = sorted(zh_texts)

    existing = load_existing(out_json)
    existing_strings = existing.get("strings", {})

    # existing format: { "zhText": {en,ja,ko} } OR { "zhText": {"key":..., ...} }
    # We'll store as: strings[zh] = {key, en, ja, ko}
    missing: list[str] = []
    for zh in zh_list:
        rec = existing_strings.get(zh)
        # If any language is missing OR translation is still equal to source,
        # treat as missing to re-translate (improves coverage).
        if (not rec) or ("en" not in rec) or ("ja" not in rec) or ("ko" not in rec) or (rec.get("en") == zh) or (rec.get("ja") == zh) or (rec.get("ko") == zh):
            missing.append(zh)

    # If only inject/scan, just dump zh keys.
    if args.inject_only:
        missing = []

    # 2) translate (optional)
    if args.translate and missing:
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise SystemExit("Missing env: DASHSCOPE_API_KEY")

        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        targets = ["en", "ja", "ko"]

        def safe_parse_translations(content: str) -> dict:
            """
            LLM 输出可能带截断/非严格 JSON。
            解析失败时返回 {}，保证主流程不崩溃。
            """
            if not content:
                return {}
            try:
                obj = json.loads(content)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                pass

            # Try extract first JSON object
            try:
                m = re.search(r"\{[\s\S]*\}", content)
                if not m:
                    return {}
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        for chunk in batch(missing, args.max_batch):
            # Ask for pure JSON mapping.
            payload = {"texts": chunk, "targets": targets}
            system = (
                "你是一名专业翻译器。"
                "任务：把输入的中文字符串分别翻译成 English / Japanese / Korean。"
                "要求：只输出合法 JSON，不要输出任何解释。"
                "JSON 结构：{"
                "  \"en\": {\"中文原文\": \"英文翻译\", ...},"
                "  \"ja\": {\"中文原文\": \"日本語翻訳\", ...},"
                "  \"ko\": {\"中文原文\": \"한국어 번역\", ...}"
                "}"
            )
            user = "请翻译并返回 JSON：\n" + json.dumps(payload, ensure_ascii=False)

            resp_content = ""
            try:
                # Prefer deterministic output to reduce formatting issues.
                resp = client.chat.completions.create(
                    model="qwen-turbo",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    stream=False,
                    temperature=0.0,
                    max_tokens=2000,
                )
                resp_content = resp.choices[0].message.content or ""
            except Exception as e:
                resp_content = ""

            translations = safe_parse_translations(resp_content)

            en_map = translations.get("en", {}) if isinstance(translations.get("en", {}), dict) else {}
            ja_map = translations.get("ja", {}) if isinstance(translations.get("ja", {}), dict) else {}
            ko_map = translations.get("ko", {}) if isinstance(translations.get("ko", {}), dict) else {}

            # Debug log: save raw content when parsing fails (best-effort)
            if not translations:
                debug_dir = out_json.parent / "_i18n_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                dbg_path = debug_dir / f"llm_parse_failed_{sha_key(chunk[0])[:8]}.txt"
                try:
                    dbg_path.write_text(
                        f"chunk_size={len(chunk)}\nerror=parse_failed\n\nRAW_CONTENT_START\n{resp_content}\nRAW_CONTENT_END\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass

            for zh in chunk:
                existing_strings.setdefault(zh, {"key": sha_key(zh)})
                existing_strings[zh]["en"] = en_map.get(zh, zh)
                existing_strings[zh]["ja"] = ja_map.get(zh, zh)
                existing_strings[zh]["ko"] = ko_map.get(zh, zh)

            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

    # 3) write final json (even if no translate)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    for zh in zh_list:
        existing_strings.setdefault(zh, {"key": sha_key(zh)})
    existing["strings"] = existing_strings
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    # 4) optional inject i18n3.js into all html files
    if not args.skip_inject and not args.inject_only:
        inject_token = '<script src="/i18n3.js"></script>'
        for p in web_root.rglob("*.html"):
            try:
                s = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if inject_token in s:
                continue
            if "</body>" not in s.lower():
                continue
            # Insert before first </body>
            s2 = re.sub(r"</body>", inject_token + "\n</body>", s, flags=re.I, count=1)
            try:
                p.write_text(s2, encoding="utf-8")
            except Exception:
                continue

    print(f"[i18n] zh strings extracted: {len(zh_list)}, missing translated: {len(missing)}")
    print(f"[i18n] output: {out_json}")


if __name__ == "__main__":
    main()

