#!/usr/bin/env python3
"""Check that model-facing OpenStax questions are absent from retrieval passages."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.textbook_store import TextbookStore


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", str(text or "").lower()).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.open(encoding="utf-8")]
    passages = TextbookStore.load(args.corpus).passages
    normalized_passages = [(row.passage_id, normalize(row.text)) for row in passages]
    leaks = []
    split_counts = Counter()
    ids = Counter()
    for row in questions:
        question_id = str(row["question_id"])
        ids[question_id] += 1
        split_counts[str(row["split"])] += 1
        question = normalize(str(row.get("question") or ""))
        if len(question) < 40:
            continue
        for passage_id, passage in normalized_passages:
            if question in passage:
                leaks.append({"question_id": question_id, "passage_id": passage_id})
    report = {
        "status": "automatic_exact_normalized_match_audit",
        "n_questions": len(questions),
        "n_passages": len(passages),
        "split_counts": dict(split_counts),
        "duplicate_question_ids": sum(count - 1 for count in ids.values() if count > 1),
        "exact_question_passage_leaks": len(leaks),
        "leaks": leaks,
        "automatic_checks_pass": not leaks and all(count == 1 for count in ids.values()),
        "limitation": "Semantic near-duplicate and answer memorization require a separate contamination study.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["n_questions", "exact_question_passage_leaks", "automatic_checks_pass"]}, indent=2))
    return int(not report["automatic_checks_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
