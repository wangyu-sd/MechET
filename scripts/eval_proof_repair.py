#!/usr/bin/env python3
"""Evaluate certificate-guided proof repair trajectories."""
from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path


def edit_stats(before: str, after: str) -> dict[str, int]:
    left = before.splitlines()
    right = after.splitlines()
    matcher = difflib.SequenceMatcher(a=left, b=right)
    changed = inserted = deleted = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
        inserted += max(j2 - j1 - (i2 - i1), 0)
        deleted += max(i2 - i1 - (j2 - j1), 0)
    return {"changed_lines": changed, "inserted_lines": inserted, "deleted_lines": deleted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--over-edit-threshold", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = repair_attempts = success1 = success2 = final_success = over_edit = new_error = 0
    transitions: Counter[str] = Counter()
    cases = []
    for row in rows:
        candidates = list(row.get("candidates") or [])
        for candidate in candidates:
            total += 1
            history = list((candidate.get("metadata") or {}).get("repair_history") or [])
            if len(history) <= 1:
                continue
            repair_attempts += 1
            first = history[0]
            final = history[-1]
            success1 += int(len(history) >= 2 and bool(history[1].get("execute_ok")))
            success2 += int(any(bool(item.get("execute_ok")) for item in history[1:3]))
            final_success += int(bool(final.get("execute_ok")))
            stats = edit_stats(str(first.get("proof") or ""), str(final.get("proof") or ""))
            over_edit += int(stats["changed_lines"] > args.over_edit_threshold)
            failure_codes = []
            for item in history[1:]:
                certificate = str(item.get("certificate") or "")
                code = certificate.splitlines()[0].replace("FAIL ", "") if certificate else ""
                failure_codes.append(code)
            for left, right in zip(failure_codes, failure_codes[1:]):
                if left and right:
                    transitions[f"{left}->{right}"] += 1
            if len(failure_codes) >= 2 and failure_codes[-1] and failure_codes[-1] != failure_codes[0]:
                new_error += 1
            cases.append({
                "target_id": row.get("id"),
                "source_index": candidate.get("source_index"),
                "n_repairs": len(history) - 1,
                "final_execute_ok": bool(final.get("execute_ok")),
                **stats,
            })
    report = {
        "overall": {
            "n_candidates": total,
            "repair_attempt_rate": repair_attempts / max(total, 1),
            "repair_success_at_1": success1 / max(repair_attempts, 1),
            "repair_success_at_2": success2 / max(repair_attempts, 1),
            "final_repair_success": final_success / max(repair_attempts, 1),
            "over_edit_rate": over_edit / max(repair_attempts, 1),
            "new_error_introduction_rate": new_error / max(repair_attempts, 1),
        },
        "failure_transitions": dict(transitions.most_common()),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
