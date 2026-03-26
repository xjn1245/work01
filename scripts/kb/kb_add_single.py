import argparse
import sys
from pathlib import Path


# Ensure project root is importable no matter where you run the script from.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from knowledge_base import StudyAbroadKnowledgeBase  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="study_abroad_kb.db", help="sqlite db path")
    parser.add_argument("--category", required=True, help="e.g. 申请要求/签证政策")
    parser.add_argument("--title", required=True, help="entry title")
    parser.add_argument("--content", required=True, help="entry content")
    parser.add_argument("--source", required=True, help="e.g. 官方渠道/权威媒体/用户贡献")
    parser.add_argument("--tags", default="", help="comma separated tags, e.g. 美国,计算机科学,硕士")
    parser.add_argument("--expiration_days", type=int, default=365)
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    kb = StudyAbroadKnowledgeBase(args.db)
    entry_id = kb.add_knowledge_entry(
        category=args.category,
        title=args.title,
        content=args.content,
        source=args.source,
        tags=tags,
        expiration_days=args.expiration_days,
    )
    print(entry_id)


if __name__ == "__main__":
    main()

