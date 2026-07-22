#!/usr/bin/env python3
"""Evaluate gold MECH_ET SFT rows (format / reachability / BE_DELTA / answer)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from mechet.sft import parse_mech_cot_output
from mechet.mech_et import verify_mech_et


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/mechet_sft/valid.jsonl")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/mechet_eval/summary.json")
    args = parser.parse_args()

    rows = []
    with args.data.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    totals = {
        "n": 0,
        "format_ok": 0,
        "reachability_ok": 0,
        "be_delta_exact": 0,
        "electron_conserved": 0,
        "answer_exact": 0,
        "main_product_ok": 0,
    }
    for row in rows:
        assistant = row["messages"][-1]["content"]
        product = ""
        for msg in row["messages"]:
            if msg["role"] == "user" and msg["content"].startswith("TARGET:"):
                product = msg["content"].split("\n", 1)[0].replace("TARGET:", "").strip()
                break
        parsed = parse_mech_cot_output(assistant)
        verified = verify_mech_et(
            mechanism_body=str(parsed.get("mechanism") or ""),
            answer=str(parsed.get("answer") or ""),
            main_product=product or None,
            expected_precursor=str(parsed.get("answer") or ""),
        )
        totals["n"] += 1
        for key in (
            "format_ok",
            "reachability_ok",
            "be_delta_exact",
            "electron_conserved",
            "answer_exact",
            "main_product_ok",
        ):
            if verified.get(key):
                totals[key] += 1

    rates = {f"{k}_rate": (totals[k] / totals["n"] if totals["n"] else 0.0) for k in totals if k != "n"}
    summary = {"data": str(args.data), "limit": args.limit, "totals": totals, "rates": rates}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
