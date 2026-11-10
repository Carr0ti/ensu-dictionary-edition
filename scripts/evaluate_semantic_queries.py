#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()

rows = json.loads(args.results.read_text(encoding="utf-8"))
report = {
    "test": "real Ensu Rust RetrievalIndex semantic query test",
    "query_count": len(rows),
    "queries": [],
}
top5_count = 0
top10_count = 0
for row in rows:
    rank = row.get("expected_rank")
    if rank is not None and rank <= 5:
        top5_count += 1
    if rank is not None and rank <= 10:
        top10_count += 1
    item = {
        "query": row["query"],
        "expected_headword": row["expected_headword"],
        "rank": rank,
        "in_top_5": bool(rank is not None and rank <= 5),
        "top_results": row["top_results"][:5],
    }
    report["queries"].append(item)
report["expected_headwords_in_top_5"] = top5_count
report["expected_headwords_in_top_10"] = top10_count
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
