#!/usr/bin/env python3
"""Evaluate H1 causal faithfulness under frozen tool interventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import (
    align_prediction_artifact,
    condition_metrics,
    endpoint_evaluation,
    file_sha256,
    read_jsonl,
    row_id,
)


def parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("artifact must be NAME=PATH")
    return name, Path(path)


def _paired_effect(
    normal_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    normal = {row_id(row): endpoint_evaluation(row) for row in normal_rows}
    intervention = {
        row_id(row): endpoint_evaluation(row) for row in intervention_rows
    }
    identifiers = list(normal)
    normal_correct = sum(bool(normal[item]["structural_exact"]) for item in identifiers)
    intervention_correct = sum(
        bool(intervention[item]["structural_exact"]) for item in identifiers
    )
    normal_trace = sum(bool(normal[item]["trace_bound"]) for item in identifiers)
    intervention_trace = sum(
        bool(intervention[item]["trace_bound"]) for item in identifiers
    )
    lost_correct = sum(
        bool(normal[item]["structural_exact"])
        and not bool(intervention[item]["structural_exact"])
        for item in identifiers
    )
    gained_correct = sum(
        not bool(normal[item]["structural_exact"])
        and bool(intervention[item]["structural_exact"])
        for item in identifiers
    )
    n = max(len(identifiers), 1)
    return {
        "n_ids": len(identifiers),
        "normal_structural_exact_rate": normal_correct / n,
        "intervention_structural_exact_rate": intervention_correct / n,
        "structural_exact_delta_intervention_minus_normal": (
            intervention_correct - normal_correct
        )
        / n,
        "normal_trace_bound_rate": normal_trace / n,
        "intervention_trace_bound_rate": intervention_trace / n,
        "trace_bound_delta_intervention_minus_normal": (
            intervention_trace - normal_trace
        )
        / n,
        "correct_to_incorrect_rate": lost_correct / n,
        "incorrect_to_correct_rate": gained_correct / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument(
        "--intervention",
        action="append",
        type=parse_artifact,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-absolute-drop",
        type=float,
        default=0.01,
        help="minimum structural-exact degradation required for a causal-sensitivity signal",
    )
    args = parser.parse_args()

    for path in [args.reference, args.normal, *(path for _, path in args.intervention)]:
        if not path.exists():
            raise FileNotFoundError(path)

    reference = read_jsonl(args.reference)
    normal_predictions = read_jsonl(args.normal)
    normal = align_prediction_artifact(
        reference, normal_predictions, condition_name="normal"
    )
    normal_metrics = condition_metrics(normal)
    interventions: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {
        "normal": {
            "path": str(args.normal),
            "sha256": file_sha256(args.normal),
            "n_rows": len(normal_predictions),
        }
    }
    for name, path in args.intervention:
        if name in interventions or name == "normal":
            raise ValueError(f"duplicate intervention name: {name}")
        predictions = read_jsonl(path)
        interventions[name] = align_prediction_artifact(
            reference, predictions, condition_name=name
        )
        sources[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "n_rows": len(predictions),
        }

    intervention_metrics = {
        name: condition_metrics(rows) for name, rows in interventions.items()
    }
    paired = {
        name: _paired_effect(normal, rows)
        for name, rows in interventions.items()
    }
    integrity = {
        "normal_all_predictions_present": normal_metrics["missing_prediction_rate"] == 0,
        "normal_trace_binding_complete": (
            normal_metrics["trace_prediction_rate"] == 1
            and normal_metrics["trace_bound_rate"] == 1
        ),
        "normal_no_reexecution_errors": normal_metrics["evaluation_error_rate"] == 0,
        "interventions_all_predictions_present": all(
            value["missing_prediction_rate"] == 0
            for value in intervention_metrics.values()
        ),
        "interventions_no_reexecution_errors": all(
            value["evaluation_error_rate"] == 0
            for value in intervention_metrics.values()
        ),
    }
    required_names = {
        "remove_tool_observations",
        "stale_tool_observations",
        "shuffle_tool_observations",
    }
    missing_required = sorted(required_names - set(interventions))
    effects = [
        -float(value["structural_exact_delta_intervention_minus_normal"])
        for value in paired.values()
    ]
    causal_sensitivity = bool(effects) and max(effects) >= args.minimum_absolute_drop

    result = {
        "artifact_type": "frozen_causal_intervention_evaluation",
        "scientific_hypothesis": "H1_causal_faithfulness",
        "reference": {
            "path": str(args.reference),
            "sha256": file_sha256(args.reference),
            "n_ids": len(reference),
        },
        "sources": sources,
        "normal": normal_metrics,
        "interventions": intervention_metrics,
        "paired_effects": paired,
        "integrity": integrity,
        "claim_gates": {
            "required_interventions_present": not missing_required,
            "missing_required_interventions": missing_required,
            "normal_path_is_fully_trace_bound": (
                integrity["normal_trace_binding_complete"]
            ),
            "all_prediction_artifacts_complete": (
                integrity["normal_all_predictions_present"]
                and integrity["interventions_all_predictions_present"]
            ),
            "all_outputs_recompute_without_error": (
                integrity["normal_no_reexecution_errors"]
                and integrity["interventions_no_reexecution_errors"]
            ),
            "causal_sensitivity_observed": causal_sensitivity,
            "minimum_absolute_drop": args.minimum_absolute_drop,
        },
        "interpretation": {
            "positive_result": (
                "The normal trace-owned path is complete and interventions on observations "
                "produce a paired degradation without creating a hidden answer channel."
            ),
            "negative_result": (
                "Insensitivity to observation interventions blocks the tool-grounded causal claim, "
                "even if endpoint accuracy remains high."
            ),
            "structural_metric": (
                "Atom-contributing structural precursor exact match, ignoring atom-map labels."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
