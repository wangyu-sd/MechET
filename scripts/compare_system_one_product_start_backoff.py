#!/usr/bin/env python3
"""Paired endpoint comparison of frozen PR81 product-start fallback policies."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(path: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads((path / "report.json").read_text())
    cases = {}
    for line in (path / "cases.jsonl").read_text().splitlines():
        case = json.loads(line)
        case_id = str(case["id"])
        if case_id in cases:
            raise ValueError(f"duplicate reaction ID in {path}: {case_id}")
        cases[case_id] = case
    if len(cases) != report["evaluated_reactions"]:
        raise ValueError(f"case count differs from report: {path}")
    if sum(bool(case["endpoint_exact"]) for case in cases.values()) != report["endpoint_exact"]:
        raise ValueError(f"endpoint count differs from report: {path}")
    return report, cases


def action_signature(action: dict) -> dict:
    return {key: action.get(key) for key in (
        "step", "action", "accepted", "state_before", "state_after", "batch",
        "selected_pairs", "execute_ok", "code",
    )}


def compare(base_report: dict, base_cases: dict[str, dict],
            candidate_report: dict, candidate_cases: dict[str, dict]) -> dict:
    for key in ("scope", "split", "source", "train_import_source", "reaction_denominator",
                "evaluated_reactions", "selection", "max_actions", "weights"):
        if base_report[key] != candidate_report[key]:
            raise ValueError(f"not a matched comparison: {key}")
    if (base_report.get("legality_backoff", False)
            or candidate_report.get("legality_backoff") is not True):
        raise ValueError("baseline/candidate fallback modes are not false/true")
    if set(base_cases) != set(candidate_cases):
        raise ValueError("reaction ID sets differ")

    transition_counts: Counter[str] = Counter()
    improved = worsened = rescued = 0
    paired_deltas = []
    for case_id in sorted(base_cases):
        base, candidate = base_cases[case_id], candidate_cases[case_id]
        for key in ("id", "target", "expected_precursor", "reference_decisions"):
            if base[key] != candidate[key]:
                raise ValueError(f"{case_id}: paired case differs in {key}")
        if base["terminal"] != "ELECTRON_EXECUTION_FAILED":
            if any(base.get(key) != candidate.get(key) for key in (
                "terminal", "completed", "predicted_precursor", "endpoint_exact",
                "electron_events", "import_batches",
            )) or [action_signature(action) for action in base["actions"]] != [
                action_signature(action) for action in candidate["actions"]
            ]:
                raise ValueError(f"{case_id}: non-failure episode changed under fallback")
        else:
            failed_at = next(index for index, action in enumerate(base["actions"])
                             if action.get("execute_ok") is False)
            if [action_signature(action) for action in base["actions"][:failed_at]] != [
                action_signature(action) for action in candidate["actions"][:failed_at]
            ]:
                raise ValueError(f"{case_id}: history changed before failed action")
            if (len(candidate["actions"]) <= failed_at
                    or base["actions"][failed_at]["ranked_top8"]
                    != candidate["actions"][failed_at]["ranked_top8"]):
                raise ValueError(f"{case_id}: frozen electron ranking changed")
            rescued += int(candidate["actions"][failed_at].get("execute_ok") is True)
        transition_counts[f"{base['terminal']} -> {candidate['terminal']}"] += 1
        before, after = bool(base["endpoint_exact"]), bool(candidate["endpoint_exact"])
        improved += int(not before and after)
        worsened += int(before and not after)
        paired_deltas.append(int(after) - int(before))

    rng = random.Random(17)
    n = len(paired_deltas)
    bootstrap = sorted(
        sum(paired_deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(5000)
    )
    return {
        "evaluated_reactions": n,
        "baseline_exact": base_report["endpoint_exact"],
        "candidate_exact": candidate_report["endpoint_exact"],
        "endpoint_delta_percentage_points": 100 * sum(paired_deltas) / n,
        "endpoint_delta_bootstrap_95ci_percentage_points": [
            100 * bootstrap[124], 100 * bootstrap[4874],
        ],
        "improved_reactions": improved,
        "worsened_reactions": worsened,
        "baseline_electron_failures": base_report["terminal_counts"].get(
            "ELECTRON_EXECUTION_FAILED", 0
        ),
        "failed_events_locally_rescued": rescued,
        "terminal_transitions": dict(sorted(transition_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base_report, base_cases = read_run(args.baseline)
    candidate_report, candidate_cases = read_run(args.candidate)
    result = {
        "artifact_type": "system_one_product_start_paired_backoff_comparison",
        "baseline_report_sha256": sha256(args.baseline / "report.json"),
        "baseline_cases_sha256": sha256(args.baseline / "cases.jsonl"),
        "candidate_report_sha256": sha256(args.candidate / "report.json"),
        "candidate_cases_sha256": sha256(args.candidate / "cases.jsonl"),
        "source_sha256": base_report["source"]["sha256"],
        "weights": base_report["weights"],
        "bootstrap_reaction_resamples": 5000,
        "bootstrap_seed": 17,
        **compare(base_report, base_cases, candidate_report, candidate_cases),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
