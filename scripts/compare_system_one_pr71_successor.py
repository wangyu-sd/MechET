#!/usr/bin/env python3
"""Paired local-successor comparison of frozen PR81 and PR71 predictions.

The action-count backoff rule is derived only from each model's executor result:
try two flows, then use one flow if the two-flow action fails. The rule was
chosen on validation and frozen before the held-out strict trace-view test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result(directory: Path, split: str) -> tuple[dict, dict[str, dict], dict]:
    report_path = directory / "report.json"
    cases_path = directory / "cases.jsonl"
    report = json.loads(report_path.read_text())
    if report.get("scope") != "reference_current_state_not_product_start":
        raise ValueError(f"{directory}: wrong evaluation scope")
    if report.get("split", split) != split:
        raise ValueError(f"{directory}: split mismatch")
    cases = {}
    with cases_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in cases:
                raise ValueError(f"{directory}: duplicate event ID {row['id']}")
            cases[row["id"]] = row
    if len(cases) != report["evaluated_events"] or report["gold_replay_ok"] != len(cases):
        raise ValueError(f"{directory}: event or GT replay denominator mismatch")
    source = report.get("valid_source") or report.get("source")
    if source["event_decisions"] != len(cases) or source["sha256"] != source["declared_sha256"]:
        raise ValueError(f"{directory}: frozen source mismatch")
    hashes = {"report_sha256": file_sha256(report_path), "cases_sha256": file_sha256(cases_path)}
    return report, cases, hashes


def backoff(row: dict) -> dict:
    policies = row["policies"]
    chosen = policies["fixed2"] if policies["fixed2"]["execute_ok"] else policies["fixed1"]
    recorded = policies.get("validity_backoff_2_to_1")
    if recorded is not None and (
        recorded["execute_ok"] != chosen["execute_ok"]
        or recorded["successor_exact"] != chosen["successor_exact"]
        or recorded["successor"] != chosen["successor"]
    ):
        raise ValueError(f"{row['id']}: recorded backoff differs from frozen rule")
    return chosen


def cluster_bootstrap(clusters: list[tuple[int, int, int]], seed: int = 17,
                      repetitions: int = 5000) -> tuple[float, float]:
    rng = random.Random(seed)
    n_clusters = len(clusters)
    estimates = []
    for _ in range(repetitions):
        total_n = total_small = total_large = 0
        for _ in range(n_clusters):
            n, small, large = clusters[rng.randrange(n_clusters)]
            total_n += n
            total_small += small
            total_large += large
        estimates.append((total_small - total_large) / total_n)
    estimates.sort()
    return estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions) - 1]


def compare(split: str, small_dir: Path, large_dir: Path) -> dict:
    small_report, small, small_hashes = load_result(small_dir, split)
    large_report, large, large_hashes = load_result(large_dir, split)
    small_source = small_report.get("valid_source") or small_report.get("source")
    large_source = large_report.get("valid_source") or large_report.get("source")
    if small_source["sha256"] != large_source["sha256"] or small.keys() != large.keys():
        raise ValueError(f"{split}: model comparisons must use identical frozen event IDs")
    tallies = {"both_exact": 0, "small_only_exact": 0, "large_only_exact": 0,
               "neither_exact": 0, "small_execute": 0, "large_execute": 0,
               "small_top2_gold_set": 0, "large_top2_gold_set": 0, "two_flow_events": 0}
    cluster_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for event_id, small_row in small.items():
        large_row = large[event_id]
        if (small_row["gold_successor"] != large_row["gold_successor"]
                or small_row["pair_targets"] != large_row["pair_targets"]
                or small_row["gold_flow_count"] != large_row["gold_flow_count"]):
            raise ValueError(f"{event_id}: model comparisons differ in reference chemistry")
        small_action, large_action = backoff(small_row), backoff(large_row)
        small_exact = bool(small_action["successor_exact"])
        large_exact = bool(large_action["successor_exact"])
        tallies[("both_exact" if small_exact and large_exact else
                 "small_only_exact" if small_exact else
                 "large_only_exact" if large_exact else "neither_exact")] += 1
        tallies["small_execute"] += int(small_action["execute_ok"])
        tallies["large_execute"] += int(large_action["execute_ok"])
        if small_row["gold_flow_count"] == 2:
            tallies["two_flow_events"] += 1
            for key, row in (("small_top2_gold_set", small_row),
                             ("large_top2_gold_set", large_row)):
                tallies[key] += int(set(row["ranked_top8"][:2]) == set(row["pair_targets"]))
        reaction_id = event_id.split("::", 1)[0]
        cluster_counts[reaction_id][0] += 1
        cluster_counts[reaction_id][1] += int(small_exact)
        cluster_counts[reaction_id][2] += int(large_exact)
    n = len(small)
    small_exact = tallies["both_exact"] + tallies["small_only_exact"]
    large_exact = tallies["both_exact"] + tallies["large_only_exact"]
    ci_low, ci_high = cluster_bootstrap([tuple(value) for value in cluster_counts.values()])
    return {
        "split": split,
        "scope": "reference_current_state_not_product_start",
        "source_sha256": small_source["sha256"],
        "events": n,
        "reactions_with_events": len(cluster_counts),
        "policy": "two_flows_then_one_on_executor_failure_no_reference_count",
        "policy_selected_after_validation": True,
        "small": {
            "checkpoint": small_report["checkpoint_manifest"],
            "pair_r1": small_report["pair_recall"]["pair_r1"],
            "pair_all_r8": small_report["pair_recall"]["pair_all_r8"],
            "execute_ok": tallies["small_execute"],
            "successor_exact": small_exact,
            "successor_exact_rate": small_exact / n,
            "evaluation_elapsed_s": small_report["elapsed_s"],
            "artifact_hashes": small_hashes,
        },
        "large": {
            "checkpoint": large_report["checkpoint"],
            "pair_r1": large_report["pair_recall"]["pair_r1"],
            "pair_all_r8": large_report["pair_recall"]["pair_all_r8"],
            "execute_ok": tallies["large_execute"],
            "successor_exact": large_exact,
            "successor_exact_rate": large_exact / n,
            "evaluation_elapsed_s": large_report["elapsed_s"],
            "artifact_hashes": large_hashes,
        },
        "paired_successor_exact_difference_small_minus_large": (small_exact - large_exact) / n,
        "reaction_cluster_bootstrap_95pct_ci": [ci_low, ci_high],
        "bootstrap_repetitions": 5000,
        "bootstrap_seed": 17,
        "paired_outcomes": tallies,
        "evaluation_wall_ratio_large_over_small": large_report["elapsed_s"] / small_report["elapsed_s"],
        "wall_ratio_note": "Same one-A100 frozen data/executor, but evaluator/model implementations differ; not a causal architecture ablation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("small_valid", "large_valid", "small_test", "large_test"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {
        "artifact_type": "pr81_pr71_paired_local_successor_comparison",
        "valid": compare("valid", args.small_valid, args.large_valid),
        "test": compare("test", args.small_test, args.large_test),
        "limitations": [
            "reference current states, not product-start autonomous trajectories",
            "mech-USPTO-31k strict executable trace subset, not 3120-reaction full endpoint test",
            "small and large differ in backbone size, prior Stage-II adapter, option-state format and training objective",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "valid_difference": result["valid"]["paired_successor_exact_difference_small_minus_large"],
                      "test_difference": result["test"]["paired_successor_exact_difference_small_minus_large"]}))


if __name__ == "__main__":
    main()
