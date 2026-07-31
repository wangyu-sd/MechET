#!/usr/bin/env python3
"""Aggregate proof hypothesis survival, Pass@K, and class diversity."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def candidate_list(row: dict) -> list[dict]:
    return list(row.get("hypotheses") or row.get("candidates") or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="*", default=[1, 4, 16, 64])
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    totals = defaultdict(float)
    pass_at_k = {int(k): {"execute": 0, "endpoint": 0} for k in args.k}
    cases = []
    for row in rows:
        values = candidate_list(row)
        for item in values:
            totals["generated"] += 1
            totals["parseable"] += int(bool(item.get("format_ok")))
            totals["executable"] += int(bool(item.get("execute_ok")))
            totals["endpoint_exact"] += int(bool(item.get("endpoint_exact")))
            totals["repaired"] += int(bool(item.get("repaired")))
        executable_classes = {
            str(item.get("equivalence_digest") or "")
            for item in values
            if item.get("execute_ok") and item.get("equivalence_digest")
        }
        compositions = {
            str(item.get("composition_digest") or "")
            for item in values
            if item.get("execute_ok") and item.get("composition_digest")
        }
        endpoints = {
            str(item.get("derived_precursor") or "")
            for item in values
            if item.get("execute_ok") and item.get("derived_precursor")
        }
        totals["equivalence_classes"] += len(executable_classes)
        totals["compositions"] += len(compositions)
        totals["endpoints"] += len(endpoints)
        for k in args.k:
            prefix = values[: int(k)]
            pass_at_k[int(k)]["execute"] += int(any(item.get("execute_ok") for item in prefix))
            pass_at_k[int(k)]["endpoint"] += int(any(item.get("endpoint_exact") for item in prefix))
        cases.append({
            "id": row.get("id"),
            "n_generated": len(values),
            "n_executable": sum(bool(item.get("execute_ok")) for item in values),
            "n_endpoint_exact": sum(bool(item.get("endpoint_exact")) for item in values),
            "n_equivalence_classes": len(executable_classes),
            "n_compositions": len(compositions),
            "n_endpoints": len(endpoints),
        })
    n_targets = len(rows)
    n_generated = max(totals["generated"], 1)
    report = {
        "overall": {
            "n_targets": n_targets,
            "n_generated": int(totals["generated"]),
            "parse_rate": totals["parseable"] / n_generated,
            "execute_rate": totals["executable"] / n_generated,
            "endpoint_exact_rate": totals["endpoint_exact"] / n_generated,
            "repair_rate": totals["repaired"] / n_generated,
            "mean_executable_classes_per_target": totals["equivalence_classes"] / max(n_targets, 1),
            "mean_compositions_per_target": totals["compositions"] / max(n_targets, 1),
            "mean_endpoints_per_target": totals["endpoints"] / max(n_targets, 1),
        },
        "pass_at_k": {
            str(k): {
                "execute_pass": value["execute"] / max(n_targets, 1),
                "endpoint_pass": value["endpoint"] / max(n_targets, 1),
            }
            for k, value in pass_at_k.items()
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["overall"], "pass_at_k": report["pass_at_k"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
