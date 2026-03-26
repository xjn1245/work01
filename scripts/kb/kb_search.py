import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from knowledge_base import StudyAbroadKnowledgeBase  # noqa: E402


def _print_result(r, max_content_chars: int = 300):
    title = r.get("title", "")
    source = r.get("source", "")
    last_updated = r.get("last_updated", "")
    credibility = r.get("credibility_score", "")
    category = r.get("category", "")
    content = r.get("content", "") or ""
    content_excerpt = content[:max_content_chars].replace("\n", " ").strip()

    # extra score fields (hybrid)
    combined_score = r.get("combined_score", None)
    vector_score = r.get("vector_score", None)
    keyword_score = r.get("keyword_score", None)

    parts = []
    if combined_score is not None:
        parts.append(f"combined={combined_score:.4f}")
    if vector_score is not None:
        parts.append(f"vector={vector_score:.4f}")
    if keyword_score is not None:
        parts.append(f"keyword={keyword_score:.4f}")

    score_str = f" ({', '.join(parts)})" if parts else ""

    print(
        f"ID={r.get('id')} | category={category} | credibility={credibility} | updated={last_updated}{score_str}\n"
        f"  title: {title}\n"
        f"  source: {source}\n"
        f"  content: {content_excerpt}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="study_abroad_kb.db", help="sqlite db path")
    parser.add_argument("--query", required=True, help="search query text")
    parser.add_argument("--category", default=None, help="optional category filter")
    parser.add_argument("--min_credibility", type=float, default=0.7, help="minimum credibility_score")
    parser.add_argument("--mode", default="keyword", choices=["keyword", "hybrid"], help="retrieval mode")
    parser.add_argument("--top_k", type=int, default=10, help="how many results to return")
    parser.add_argument("--alpha", type=float, default=0.5, help="hybrid alpha (vector weight)")
    parser.add_argument("--max_content_chars", type=int, default=300, help="print content excerpt length")
    parser.add_argument("--json_out", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    kb = StudyAbroadKnowledgeBase(args.db)

    if args.mode == "hybrid":
        results = kb.search_knowledge_hybrid(
            query=args.query,
            category=args.category,
            min_credibility=args.min_credibility,
            top_k=args.top_k,
            alpha=args.alpha,
        )
    else:
        results = kb.search_knowledge(
            query=args.query,
            category=args.category,
            min_credibility=args.min_credibility,
        )

        # search_knowledge 内部 LIMIT 10；若你想严格 top_k，可以后处理裁剪
        results = results[: args.top_k]

    if args.json_out:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not results:
        print("No results.")
        return

    for r in results:
        _print_result(r, max_content_chars=args.max_content_chars)


if __name__ == "__main__":
    main()

