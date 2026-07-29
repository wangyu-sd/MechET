#!/usr/bin/env python3
"""Gold-data audit for MECH_ET JSONL (NOT model evaluation).

Checks that stored assistant targets are self-consistent. Expect ~100% on valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.metrics import aggregate_by_topology, aggregate_rates, score_mech_et_prediction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/mechet_sft/valid.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/mechet_eval/gold_audit_summary.json")
    args = parser.parse_args()

    cases = []
    with args.data.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = str((row.get("messages") or [{}])[-1].get("content") or "")
            cases.append(score_mech_et_prediction(row, prediction, mode="gold_audit"))
            if args.limit and len(cases) >= args.limit:
                break

    report = {
        "mode": "gold_audit",
        "data": str(args.data),
        "limit": args.limit or len(cases),
        **aggregate_rates(cases),
        "by_topology": aggregate_by_topology(cases),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    case_path = args.out.parent / "gold_audit_cases.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
