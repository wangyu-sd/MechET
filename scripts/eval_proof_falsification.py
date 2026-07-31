#!/usr/bin/env python3
"""Evaluate deterministic falsification on controlled proof corruptions."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_diagnostics import diagnose_proof
from mechet.proof_program import execute_proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_type: dict[str, list[dict]] = defaultdict(list)
    invalid_total = invalid_accepted = valid_total = valid_rejected = 0
    failure_code_correct = failure_edge_correct = 0
    failure_labelled = edge_labelled = 0
    cases = []
    for row in rows:
        proof = str(row.get("corrupted_proof") or "")
        expected_execute = bool(row.get("expected_execute"))
        result = execute_proof(proof)
        certificate = diagnose_proof(proof)
        observed_code = certificate.code if certificate else ""
        observed_edge = certificate.edge if certificate else ""
        expected_code = str(row.get("failure_code") or "")
        expected_edge = str(row.get("failure_edge") or "")
        if expected_execute:
            valid_total += 1
            valid_rejected += int(not result.ok)
        else:
            invalid_total += 1
            invalid_accepted += int(result.ok)
        if expected_code:
            failure_labelled += 1
            failure_code_correct += int(expected_code == observed_code)
        if expected_edge:
            edge_labelled += 1
            failure_edge_correct += int(expected_edge == observed_edge)
        case = {
            "source_id": row.get("source_id"),
            "corruption_type": row.get("corruption_type"),
            "expected_execute": expected_execute,
            "observed_execute": result.ok,
            "expected_failure_code": expected_code,
            "observed_failure_code": observed_code,
            "expected_failure_edge": expected_edge,
            "observed_failure_edge": observed_edge,
        }
        cases.append(case)
        by_type[str(row.get("corruption_type") or "unknown")].append(case)

    def aggregate(values: list[dict]) -> dict:
        n = len(values)
        return {
            "n": n,
            "execute_accuracy": sum(item["expected_execute"] == item["observed_execute"] for item in values) / max(n, 1),
            "invalid_acceptance_rate": sum((not item["expected_execute"]) and item["observed_execute"] for item in values) / max(sum(not item["expected_execute"] for item in values), 1),
            "valid_rejection_rate": sum(item["expected_execute"] and (not item["observed_execute"]) for item in values) / max(sum(item["expected_execute"] for item in values), 1),
        }

    report = {
        "overall": {
            "n": len(rows),
            "false_acceptance_rate": invalid_accepted / max(invalid_total, 1),
            "false_rejection_rate": valid_rejected / max(valid_total, 1),
            "failure_code_accuracy": failure_code_correct / max(failure_labelled, 1),
            "failure_edge_accuracy": failure_edge_correct / max(edge_labelled, 1),
        },
        "by_corruption_type": {key: aggregate(value) for key, value in sorted(by_type.items())},
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
