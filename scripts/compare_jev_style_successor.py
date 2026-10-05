#!/usr/bin/env python3
"""Pair typed-v2, marker-v1 and PR71 local successor results on identical IDs.

The same validation-selected, gold-independent executor-validity backoff is
applied to each model. Reference move count is never used for this comparison.
This is not product-start endpoint accuracy or an architecture-only ablation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from scripts.compare_system_one_pr71_successor import (
    backoff,
    cluster_bootstrap,
    load_result,
)


ARTIFACT_TYPES = {
    "typed_v2": "system_one_jev_typed_v2_local_successor_evaluation",
    "marker_v1": "system_one_phase0_local_successor_evaluation",
    "pr71_8b": "pr71_conditional_pointer_local_successor_evaluation",
}


def compare_split(split: str, directories: dict[str, Path]) -> dict[str, Any]:
    loaded = {name: load_result(directory, split) for name, directory in directories.items()}
    for name, (report, _, _) in loaded.items():
        if report.get("artifact_type") != ARTIFACT_TYPES[name]:
            raise ValueError(f"{name}: incompatible model/evaluation artifact")
    first = loaded["typed_v2"]
    reference_cases = first[1]
    source = first[0].get("source") or first[0].get("valid_source")
    for name, (report, cases, _) in loaded.items():
        other_source = report.get("source") or report.get("valid_source")
        if other_source["sha256"] != source["sha256"] or cases.keys() != reference_cases.keys():
            raise ValueError(f"{split}: {name} has a different frozen source or event IDs")
        for event_id, row in cases.items():
            reference = reference_cases[event_id]
            for field in ("gold_successor", "pair_targets", "gold_flow_count"):
                if row[field] != reference[field]:
                    raise ValueError(f"{event_id}: {name} differs in reference {field}")

    results: dict[str, dict[str, dict[str, Any]]] = {}
    models = {}
    two_flow_ids = [event_id for event_id, row in reference_cases.items()
                    if row["gold_flow_count"] == 2]
    for name, (report, cases, hashes) in loaded.items():
        selections = {event_id: backoff(row) for event_id, row in cases.items()}
        results[name] = selections
        executed = sum(bool(row["execute_ok"]) for row in selections.values())
        exact = sum(bool(row["successor_exact"]) for row in selections.values())
        recorded = report["policies"]["validity_backoff_2_to_1"]["overall"]
        if (recorded["n"] != len(cases) or recorded["execute_ok"] != executed
                or recorded["successor_exact"] != exact):
            raise ValueError(f"{split}: {name} report and cases disagree")
        top2_gold_set = sum(
            set(cases[event_id]["ranked_top8"][:2]) == set(cases[event_id]["pair_targets"])
            for event_id in two_flow_ids
        )
        models[name] = {
            "checkpoint": report.get("checkpoint_manifest") or report.get("checkpoint"),
            "pair_r1": report["pair_recall"]["pair_r1"],
            "pair_all_r8": report["pair_recall"]["pair_all_r8"],
            "execute_ok": executed,
            "successor_exact": exact,
            "successor_exact_rate": exact / len(cases),
            "two_flow_top2_gold_set": top2_gold_set,
            "two_flow_top2_gold_set_rate": (
                top2_gold_set / len(two_flow_ids) if two_flow_ids else None
            ),
            "evaluation_elapsed_s": report["elapsed_s"],
            "artifact_hashes": hashes,
        }

    paired = {}
    for competitor in ("marker_v1", "pr71_8b"):
        outcomes = defaultdict(int)
        clusters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for event_id in reference_cases:
            typed = bool(results["typed_v2"][event_id]["successor_exact"])
            other = bool(results[competitor][event_id]["successor_exact"])
            outcomes[("both_exact" if typed and other else
                      "typed_only_exact" if typed else
                      "competitor_only_exact" if other else "neither_exact")] += 1
            reaction_id = event_id.split("::", 1)[0]
            clusters[reaction_id][0] += 1
            clusters[reaction_id][1] += int(typed)
            clusters[reaction_id][2] += int(other)
        ci = cluster_bootstrap([tuple(value) for value in clusters.values()])
        paired[f"typed_v2_minus_{competitor}"] = {
            "successor_exact_rate_difference": (
                models["typed_v2"]["successor_exact"] - models[competitor]["successor_exact"]
            ) / len(reference_cases),
            "reaction_cluster_bootstrap_95pct_ci": list(ci),
            "bootstrap_seed": 17,
            "bootstrap_repetitions": 5000,
            "outcomes": dict(outcomes),
        }

    return {
        "split": split,
        "scope": "reference_current_state_not_product_start",
        "source_sha256": source["sha256"],
        "events": len(reference_cases),
        "reactions_with_events": len({event_id.split("::", 1)[0] for event_id in reference_cases}),
        "two_flow_events": len(two_flow_ids),
        "policy": "two_flows_then_one_on_executor_failure_no_reference_count",
        "policy_selected_after_validation": True,
        "models": models,
        "paired": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ARTIFACT_TYPES:
        for split in ("valid", "test"):
            parser.add_argument(f"--{label.replace('_', '-')}-{split}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "artifact_type": "pr81_typed_v2_three_model_paired_local_successor_comparison",
        "valid": compare_split("valid", {
            name: getattr(args, f"{name}_valid") for name in ARTIFACT_TYPES
        }),
        "test": compare_split("test", {
            name: getattr(args, f"{name}_test") for name in ARTIFACT_TYPES
        }),
        "limitations": [
            "reference current states, not autonomous product-start trajectories",
            "mech-USPTO-31k strict executable trace view, not the full 3120-reaction endpoint test",
            "typed v2, marker v1 and PR71 differ in architecture and/or checkpoint lineage; paired quality is not an architecture-only causal estimate",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "valid": report["valid"]["paired"],
                      "test": report["test"]["paired"]}), flush=True)


if __name__ == "__main__":
    main()
