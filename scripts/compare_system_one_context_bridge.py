#!/usr/bin/env python3
"""Pair a principal-product context bridge with the frozen mixture-start run.

This is an input-contract diagnostic on the strict trace view, not the full
3,120-reaction structural endpoint benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import sha256


def read_cases(directory: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads((directory / "report.json").read_text())
    path = directory / "cases.jsonl"
    declared_sha = report.get("cases_sha256")
    if declared_sha is not None and declared_sha != sha256(path):
        raise ValueError(f"case SHA mismatch: {directory}")
    cases = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["id"])
        if reaction_id in cases:
            raise ValueError(f"duplicate reaction ID: {reaction_id}")
        cases[reaction_id] = row
    if len(cases) != report["evaluated_reactions"]:
        raise ValueError(f"case denominator mismatch: {directory}")
    return report, cases


def compare(bridge_dir: Path, baseline_dir: Path) -> dict:
    bridge, bridged = read_cases(bridge_dir)
    baseline, original = read_cases(baseline_dir)
    if bridge["artifact_type"] != "system_one_pr81_principal_product_context_bridge_diagnostic":
        raise ValueError("not a principal-product bridge report")
    if bridge["strict_source"]["sha256"] != baseline["source"]["sha256"]:
        raise ValueError("strict source mismatch")
    if bridge["weights"] != baseline["weights"] or not baseline["legality_backoff"]:
        raise ValueError("policy lineage or executor rule mismatch")
    if len(bridged) != bridge["evaluated_reactions"] or not set(bridged) <= set(original):
        raise ValueError("bridge IDs do not match the frozen baseline")
    counts: Counter[str] = Counter()
    for reaction_id, candidate in bridged.items():
        reference = original[reaction_id]
        if (candidate["strict_reference_final_mixture"] != reference["target"]
                or candidate["expected_strict_full_precursor"] != reference["expected_precursor"]):
            raise ValueError(f"reference/input mismatch: {reaction_id}")
        reconstructed = candidate["inferred_final_mixture"] == reference["target"]
        if reconstructed != candidate["strict_mixture_reconstructed_byte_exact"]:
            raise ValueError(f"incorrect reconstruction flag: {reaction_id}")
        hit = bool(candidate["endpoint_exact_strict_full_precursor"])
        baseline_hit = bool(reference["endpoint_exact"])
        if reconstructed and (candidate["predicted_precursor"] != reference["predicted_precursor"]
                              or candidate["terminal"] != reference["terminal"]):
            raise ValueError(f"same input produced different rollout: {reaction_id}")
        group = "reconstructed" if reconstructed else "nonreconstructed"
        counts[group] += 1
        counts[f"{group}_bridge_hit"] += int(hit)
        counts[f"{group}_baseline_hit"] += int(baseline_hit)
        counts["bridge_hit"] += int(hit)
        counts["baseline_hit"] += int(baseline_hit)
        counts[f"paired_baseline{int(baseline_hit)}_bridge{int(hit)}"] += 1
    if counts["bridge_hit"] != bridge["endpoint_exact_strict_full_precursor"]:
        raise ValueError("bridge report endpoint total mismatch")
    if counts["reconstructed"] != bridge["strict_mixture_reconstructed_byte_exact"]:
        raise ValueError("bridge report reconstruction total mismatch")
    return {
        "artifact_type": "system_one_pr81_context_bridge_paired_audit",
        "scope": bridge["scope"],
        "split": bridge["split"],
        "bridge_report_sha256": sha256(bridge_dir / "report.json"),
        "baseline_report_sha256": sha256(baseline_dir / "report.json"),
        "bridge_cases_sha256": sha256(bridge_dir / "cases.jsonl"),
        "baseline_cases_sha256": sha256(baseline_dir / "cases.jsonl"),
        "evaluated": len(bridged),
        "counts": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.bridge_dir, args.baseline_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
