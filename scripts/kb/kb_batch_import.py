import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from knowledge_base import StudyAbroadKnowledgeBase  # noqa: E402


def _normalize_tags(tags_val):
    if tags_val is None:
        return []
    if isinstance(tags_val, list):
        return [str(x).strip() for x in tags_val if str(x).strip()]
    if isinstance(tags_val, str):
        return [t.strip() for t in tags_val.split(",") if t.strip()]
    return [str(tags_val).strip()] if str(tags_val).strip() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="study_abroad_kb.db", help="sqlite db path")
    parser.add_argument("--json", required=True, help="input JSON file path (array of entries)")
    parser.add_argument(
        "--default_source", default="用户贡献", help="fallback source if item.source is missing"
    )
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of entry objects.")

    kb = StudyAbroadKnowledgeBase(args.db)

    imported = 0
    skipped = 0

    for item in data:
        try:
            category = item.get("category", "未分类")
            title = item.get("title", "") or ""
            content = item.get("content", "") or ""
            source = item.get("source", args.default_source)
            tags = _normalize_tags(item.get("tags", []))
            expiration_days = int(item.get("expiration_days", 365))

            if not title or not content:
                skipped += 1
                continue

            kb.add_knowledge_entry(
                category=category,
                title=title,
                content=content,
                source=source,
                tags=tags,
                expiration_days=expiration_days,
            )
            imported += 1
        except Exception as e:
            skipped += 1
            print(f"[skip] {e}")

    print(f"imported={imported}, skipped={skipped}")


if __name__ == "__main__":
    main()

