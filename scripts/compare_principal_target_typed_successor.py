#!/usr/bin/env python3
"""Pair old/new typed-v2 local electron successors across target-prompt contracts.

The reference executor states and labels must agree event by event, while the
training-view JSONL hashes intentionally differ because the TARGET line differs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.compare_system_one_pr71_successor import backoff, cluster_bootstrap, load_result
from scripts.train_system_one_electron_flow import verify_source


def summarize_pairs(old: dict[str, dict], new: dict[str, dict]) -> dict:
    if old.keys() != new.keys():
        raise ValueError("old and principal-target policies have different event IDs")
    counts: Counter[str] = Counter()
    clusters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for event_id, before in old.items():
        after = new[event_id]
        for field in ("gold_successor", "pair_targets", "gold_flow_count"):
            if before[field] != after[field]:
                raise ValueError(f"{event_id}: reference chemistry differs in {field}")
        old_outcome, new_outcome = backoff(before), backoff(after)
        old_exact = bool(old_outcome["successor_exact"])
        new_exact = bool(new_outcome["successor_exact"])
        counts["events"] += 1
        counts["old_exact"] += int(old_exact)
        counts["new_exact"] += int(new_exact)
        counts["old_execute"] += int(old_outcome["execute_ok"])
        counts["new_execute"] += int(new_outcome["execute_ok"])
        counts[("both_exact" if old_exact and new_exact else
                "old_only_exact" if old_exact else
                "new_only_exact" if new_exact else "neither_exact")] += 1
        reaction_id = event_id.split("::", 1)[0]
        cluster = clusters[reaction_id]
        cluster[0] += 1
        cluster[1] += int(new_exact)
        cluster[2] += int(old_exact)
    if not counts["events"]:
        raise ValueError("empty paired evaluation")
    ci = cluster_bootstrap([tuple(value) for value in clusters.values()])
    return {
        "counts": dict(counts),
        "reactions_with_events": len(clusters),
        "new_minus_old_successor_exact_rate": (
            counts["new_exact"] - counts["old_exact"]
        ) / counts["events"],
        "reaction_cluster_bootstrap_95pct_ci": list(ci),
        "bootstrap_seed": 17,
        "bootstrap_repetitions": 5000,
    }


def compare_split(split: str, old_dir: Path, new_dir: Path,
                  old_data: Path, new_data: Path) -> dict:
    old_report, old_cases, old_hashes = load_result(old_dir, split)
    new_report, new_cases, new_hashes = load_result(new_dir, split)
    if (old_report.get("artifact_type")
            != "system_one_jev_typed_v2_local_successor_evaluation"
            or new_report.get("artifact_type")
            != "system_one_jev_typed_v2_local_successor_evaluation"):
        raise ValueError(f"{split}: expected paired typed-v2 evaluator outputs")
    old_source = verify_source(old_data / f"{split}.jsonl")
    new_source = verify_source(new_data / f"{split}.jsonl")
    if (old_source["artifact_id"]
            != "mech_uspto_31k_natural_language_electron_event_history_v2"
            or new_source["artifact_id"]
            != "mech_uspto_31k_natural_language_history_principal_target_v2"
            or old_report["source"]["sha256"] != old_source["sha256"]
            or new_report["source"]["sha256"] != new_source["sha256"]
            or old_source["event_decisions"] != new_source["event_decisions"]):
        raise ValueError(f"{split}: source lineage or event denominators differ")
    return {
        "old_source_sha256": old_source["sha256"],
        "new_source_sha256": new_source["sha256"],
        "old_report_sha256": old_hashes["report_sha256"],
        "old_cases_sha256": old_hashes["cases_sha256"],
        "new_report_sha256": new_hashes["report_sha256"],
        "new_cases_sha256": new_hashes["cases_sha256"],
        "paired": summarize_pairs(old_cases, new_cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-dir", "new-dir", "old-data", "new-data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "artifact_type": "pr81_principal_target_typed_v2_paired_local_successor_comparison",
        "scope": "reference_current_state_not_product_start",
        "valid": compare_split("valid", args.old_dir / "valid", args.new_dir / "valid",
                               args.old_data, args.new_data),
        "test": compare_split("test", args.old_dir / "test", args.new_dir / "test",
                              args.old_data, args.new_data),
        "limitations": [
            "same reference executor states and gold actions, different target-line supervision",
            "strict executable trace view only, not the complete 3120-reaction endpoint benchmark",
            "gold-independent two-flow then one-on-execution-failure policy",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "valid": report["valid"]["paired"],
                      "test": report["test"]["paired"]}), flush=True)


if __name__ == "__main__":
    main()
