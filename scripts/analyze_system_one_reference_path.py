#!/usr/bin/env python3
"""Join PR81 router and v1 local electron results along reference trajectories.

This is a teacher-forced reference-path diagnostic, not product-start rollout.
IMPORT fragment arguments are assumed to be the recorded ones and are never
predicted here; alternate chemically valid trajectories are not credited.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.system_one_action_family import ACTION_NAMES, ActionFamilyHead
from scripts.compare_system_one_pr71_successor import backoff, load_result
from scripts.train_system_one_action_family import file_sha256, load_split, metrics


def summarize_reference_paths(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    reactions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reactions[row["reaction_id"]].append(row)
    if not reactions:
        raise ValueError("no reference trajectories")
    totals = Counter()
    first_failure = Counter()
    by_event_count: dict[int, Counter] = defaultdict(Counter)
    prefix_lengths = Counter()
    for reaction_id, trajectory in reactions.items():
        indices = [item["decision_index"] for item in trajectory]
        if indices != list(range(len(trajectory))):
            raise ValueError(f"{reaction_id}: nonconsecutive decision indices")
        if trajectory[-1]["gold_action"] != "finish_trace":
            raise ValueError(f"{reaction_id}: reference trajectory lacks final FINISH")
        events = sum(item["gold_action"] == "apply_electron_flow" for item in trajectory)
        imports = sum(item["gold_action"] == "import_fragments" for item in trajectory)
        route_all = all(item["route_correct"] for item in trajectory)
        event_all = all(item["event_successor_exact"] for item in trajectory
                        if item["gold_action"] == "apply_electron_flow")
        joint = route_all and event_all
        totals["reactions"] += 1
        totals["decisions"] += len(trajectory)
        totals["events"] += events
        totals["imports"] += imports
        totals["all_routes_correct"] += int(route_all)
        totals["all_event_successors_exact"] += int(event_all)
        totals["all_reference_path_local_agreement"] += int(joint)
        bucket = by_event_count[events]
        bucket["reactions"] += 1
        bucket["all_routes_correct"] += int(route_all)
        bucket["all_event_successors_exact"] += int(event_all)
        bucket["all_reference_path_local_agreement"] += int(joint)
        prefix = 0
        for item in trajectory:
            if not item["route_correct"]:
                first_failure["route_" + item["gold_action"]] += 1
                break
            if (item["gold_action"] == "apply_electron_flow"
                    and not item["event_successor_exact"]):
                first_failure["event_successor"] += 1
                break
            prefix += 1
        else:
            first_failure["none"] += 1
        prefix_lengths[prefix] += 1
    n = totals["reactions"]
    return {
        **dict(totals),
        "all_routes_correct_rate": totals["all_routes_correct"] / n,
        "all_event_successors_exact_rate": totals["all_event_successors_exact"] / n,
        "all_reference_path_local_agreement_rate": (
            totals["all_reference_path_local_agreement"] / n
        ),
        "first_failure": dict(sorted(first_failure.items())),
        "correct_decision_prefix_length": {str(k): v for k, v in sorted(prefix_lengths.items())},
        "by_gold_event_count": {
            str(k): {**dict(value), "all_reference_path_local_agreement_rate":
                     value["all_reference_path_local_agreement"] / value["reactions"]}
            for k, value in sorted(by_event_count.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--route-run", type=Path, required=True)
    parser.add_argument("--event-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.data.stem != args.split:
        raise ValueError("data split does not match declared split")

    import torch
    from safetensors.torch import load_file

    route_report = json.loads((args.route_run / "report.json").read_text())
    if route_report.get("artifact_type") != "system_one_phase1a_action_family_result":
        raise ValueError("not a completed Phase-1a router")
    source_sha = file_sha256(args.data)
    if source_sha != route_report["source_sha256"][args.split]:
        raise ValueError("router source SHA differs from decision data")
    feature_path = args.route_run / f"{args.split}_features.safetensors"
    head_path = args.route_run / "action_family_head.pt"
    if (file_sha256(feature_path) != route_report["feature_sha256"][args.split]
            or file_sha256(head_path) != route_report["head_sha256"]):
        raise ValueError("router features/head differ from completed run")
    examples, source_counts = load_split(args.data)
    features = load_file(str(feature_path))["features"]
    if features.ndim != 2 or features.shape[0] != len(examples):
        raise ValueError("router features do not align with decisions")
    head = ActionFamilyHead(features.shape[1])
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    head.eval()
    with torch.inference_mode():
        predictions = head(features.float()).argmax(dim=-1).tolist()
    reproduced = metrics([example.label for example in examples], predictions)
    if reproduced["confusion_gold_rows_predicted_columns"] != (
        route_report[args.split]["confusion_gold_rows_predicted_columns"]
    ):
        raise ValueError("router predictions do not reproduce frozen report")

    event_report, event_cases, event_hashes = load_result(args.event_eval, args.split)
    if (event_report["artifact_type"] != "system_one_phase0_local_successor_evaluation"
            or (event_report.get("source") or event_report.get("valid_source"))["sha256"]
            != source_sha):
        raise ValueError("electron evaluator is not the matching frozen v1 result")

    joined = []
    seen_events = set()
    for example, prediction in zip(examples, predictions, strict=True):
        gold_action = ACTION_NAMES[example.label]
        event_exact = None
        if gold_action == "apply_electron_flow":
            if example.row_id not in event_cases:
                raise ValueError(f"{example.row_id}: missing electron evaluation case")
            event_exact = bool(backoff(event_cases[example.row_id])["successor_exact"])
            seen_events.add(example.row_id)
        joined.append({
            "reaction_id": example.reaction_id,
            "decision_index": example.history_accepted_actions,
            "gold_action": gold_action,
            "route_correct": prediction == example.label,
            "event_successor_exact": event_exact,
        })
    if seen_events != event_cases.keys():
        raise ValueError("electron evaluation has extra or missing event IDs")
    summary = summarize_reference_paths(joined)
    if summary["reactions"] != source_counts["reactions"]:
        raise ValueError("reaction denominator mismatch")
    report = {
        "artifact_type": "system_one_v1_reference_path_local_agreement",
        "scope": "teacher_forced_reference_states_not_product_start_rollout",
        "split": args.split,
        "source_sha256": source_sha,
        "route_head_sha256": route_report["head_sha256"],
        "route_features_sha256": route_report["feature_sha256"][args.split],
        "electron_case_hashes": event_hashes,
        "import_argument_assumed_reference": True,
        "alternate_valid_trajectories_not_credited": True,
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "split": args.split,
        "reactions": summary["reactions"],
        "all_reference_path_local_agreement": summary["all_reference_path_local_agreement"],
        "rate": summary["all_reference_path_local_agreement_rate"],
        "first_failure": summary["first_failure"],
    }), flush=True)


if __name__ == "__main__":
    main()
