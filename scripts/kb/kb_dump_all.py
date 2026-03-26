import argparse
import json
import sqlite3
from pathlib import Path


def _maybe_load_json(s):
    if s is None:
        return []
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def dump_table(conn: sqlite3.Connection, table_name: str):
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()

    out = []
    for row in rows:
        obj = dict(zip(cols, row))
        out.append(obj)
    return out, cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="study_abroad_kb.db", help="sqlite db path")
    parser.add_argument("--json_out", default="", help="optional json file path to write")
    parser.add_argument("--limit_entries", type=int, default=0, help="limit knowledge_entries rows (0=all)")
    parser.add_argument("--include_relations", action="store_true", help="also dump knowledge_relations")
    parser.add_argument("--include_update_logs", action="store_true", help="also dump update_logs")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    conn = sqlite3.connect(str(db_path))

    result = {}

    # Dump knowledge_entries first (often the biggest)
    cur = conn.cursor()
    if args.limit_entries and args.limit_entries > 0:
        cur.execute(f"SELECT * FROM knowledge_entries LIMIT {int(args.limit_entries)}")
    else:
        cur.execute("SELECT * FROM knowledge_entries")

    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    entries = []
    for row in rows:
        obj = dict(zip(cols, row))
        # normalize tags for readability
        obj["tags"] = _maybe_load_json(obj.get("tags"))
        entries.append(obj)
    result["knowledge_entries"] = entries

    if args.include_relations:
        relations, _ = dump_table(conn, "knowledge_relations")
        result["knowledge_relations"] = relations

    if args.include_update_logs:
        update_logs, _ = dump_table(conn, "update_logs")
        result["update_logs"] = update_logs

    conn.close()

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"dumped to: {out_path}")
        return

    # Print summary + first N records to avoid terminal overflow
    total_entries = len(result["knowledge_entries"])
    print(f"knowledge_entries total={total_entries}")
    preview_n = min(5, total_entries)
    print(json.dumps(result["knowledge_entries"][:preview_n], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

