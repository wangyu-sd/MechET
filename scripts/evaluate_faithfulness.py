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
    file_sha256,
    read_jsonl,
    row_id,
)
from mechet.prediction_metrics import prediction_runtime_contract, prediction_set_metrics
from mechet.statistical_evaluation import holm_adjust, paired_binary_contrast
from mechet.strict_prediction_evaluation import condition_metrics, endpoint_evaluation


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
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    normal = {row_id(row): endpoint_evaluation(row) for row in normal_rows}
    intervention = {
        row_id(row): endpoint_evaluation(row) for row in intervention_rows
    }
    identifiers = list(normal)
    if set(identifiers) != set(intervention):
        raise ValueError("paired H1 artifacts do not share the same identifier universe")

    normal_correct_values = [
        bool(normal[item]["structural_exact"]) for item in identifiers
    ]
    intervention_correct_values = [
        bool(intervention[item]["structural_exact"]) for item in identifiers
    ]
    normal_trace_values = [bool(normal[item]["trace_bound"]) for item in identifiers]
    intervention_trace_values = [
        bool(intervention[item]["trace_bound"]) for item in identifiers
    ]
    structural = paired_binary_contrast(
        normal_correct_values,
        intervention_correct_values,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    trace = paired_binary_contrast(
        normal_trace_values,
        intervention_trace_values,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 1,
        confidence=confidence,
    )

    n = len(identifiers)
    return {
        "n_ids": n,
        "normal_structural_exact_rate": structural["left_rate"],
        "intervention_structural_exact_rate": structural["right_rate"],
        "structural_exact_delta_intervention_minus_normal": -float(
            structural["delta_left_minus_right"]
        ),
        "structural_exact_delta_normal_minus_intervention": structural[
            "delta_left_minus_right"
        ],
        "normal_trace_bound_rate": trace["left_rate"],
        "intervention_trace_bound_rate": trace["right_rate"],
        "trace_bound_delta_intervention_minus_normal": -float(
            trace["delta_left_minus_right"]
        ),
        "trace_bound_delta_normal_minus_intervention": trace[
            "delta_left_minus_right"
        ],
        "correct_to_incorrect_rate": structural[
            "left_correct_right_incorrect"
        ]
        / n,
        "incorrect_to_correct_rate": structural[
            "left_incorrect_right_correct"
        ]
        / n,
        "structural_exact_paired_inference": structural,
        "trace_bound_paired_inference": trace,
    }


def _audit_intervention(
    name: str, predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    audits = [
        dict((row.get("rollout_state") or {}).get("intervention_audit") or {})
        for row in predictions
    ]
    present = len(audits) == len(predictions) and all(bool(item) for item in audits)
    result: dict[str, Any] = {
        "n_rows": len(predictions),
        "audit_present_for_all_rows": present,
    }
    if name == "remove_tool_observations":
        result["length_preserved_for_all_rows"] = present and all(
            item.get("observation_length_preserved") is True for item in audits
        )
    if name == "shuffle_tool_observations":
        self_donor_pairs = []
        unavailable = {}
        for row, audit in zip(predictions, audits):
            target = str(audit.get("target_smiles") or row.get("target_smiles") or "")
            donors = dict(audit.get("shuffle_donors") or {})
            for tool_name, donor in donors.items():
                donor_target = str((donor or {}).get("donor_target_smiles") or "")
                if donor_target == target:
                    self_donor_pairs.append(
                        {
                            "id": row_id(row),
                            "tool": tool_name,
                            "target_smiles": target,
                        }
                    )
            missing = list(audit.get("shuffle_unavailable_tools") or [])
            if missing:
                unavailable[row_id(row)] = missing
        result.update(
            {
                "shuffle_contract_valid_for_all_rows": present
                and all(item.get("shuffle_contract_valid") is True for item in audits),
                "self_donor_pairs": self_donor_pairs,
                "self_donor_free": not self_donor_pairs,
                "unavailable_tools_by_id": unavailable,
                "all_called_tool_types_have_donors": not unavailable,
                "donor_manifest_sha256": sorted(
                    {
                        str(
                            (item.get("shuffle_plan_contract") or {}).get(
                                "donor_manifest_sha256"
                            )
                            or ""
                        )
                        for item in audits
                    }
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--intervention", action="append", type=parse_artifact, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-absolute-drop", type=float, default=0.01)
    parser.add_argument(
        "--primary-intervention", default="remove_tool_observations"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    for path in [
        args.reference,
        args.normal,
        *(path for _, path in args.intervention),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    reference = read_jsonl(args.reference)
    normal_predictions = read_jsonl(args.normal)
    normal = align_prediction_artifact(
        reference, normal_predictions, condition_name="normal"
    )
    normal_metrics = {
        **condition_metrics(normal),
        **prediction_set_metrics(normal),
    }
    normal_runtime = prediction_runtime_contract(
        normal_predictions, include_adapter=True
    )

    interventions: dict[str, list[dict[str, Any]]] = {}
    intervention_predictions: dict[str, list[dict[str, Any]]] = {}
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
        intervention_predictions[name] = predictions
        interventions[name] = align_prediction_artifact(
            reference, predictions, condition_name=name
        )
        sources[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "n_rows": len(predictions),
        }

    intervention_metrics = {
        name: {
            **condition_metrics(rows),
            **prediction_set_metrics(rows),
        }
        for name, rows in interventions.items()
    }
    intervention_audits = {
        name: _audit_intervention(name, rows)
        for name, rows in intervention_predictions.items()
    }
    runtime_contracts = {
        "normal": normal_runtime,
        **{
            name: prediction_runtime_contract(rows, include_adapter=True)
            for name, rows in intervention_predictions.items()
        },
    }
    runtime_digests = {
        name: value["runtime_contract_sha256"]
        for name, value in runtime_contracts.items()
    }
    runtime_consistent_within = all(
        value["runtime_contract_consistent"]
        for value in runtime_contracts.values()
    )
    runtime_complete = all(
        value["runtime_contract_complete"] for value in runtime_contracts.values()
    )
    runtime_matched_across = len(set(runtime_digests.values())) == 1

    paired = {
        name: _paired_effect(
            normal,
            rows,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index * 11,
            confidence=args.confidence,
        )
        for index, (name, rows) in enumerate(interventions.items())
    }
    adjusted_p_values = holm_adjust(
        {
            name: float(
                value["structural_exact_paired_inference"][
                    "mcnemar_exact_p_value"
                ]
            )
            for name, value in paired.items()
        }
    )
    for name, adjusted in adjusted_p_values.items():
        paired[name]["structural_exact_paired_inference"][
            "holm_adjusted_mcnemar_p_value"
        ] = adjusted

    remove_audit = intervention_audits.get("remove_tool_observations", {})
    shuffle_audit = intervention_audits.get("shuffle_tool_observations", {})
    integrity = {
        "normal_all_predictions_present": normal_metrics[
            "missing_prediction_rate"
        ]
        == 0,
        "normal_trace_binding_complete": normal_metrics[
            "trace_prediction_rate"
        ]
        == 1
        and normal_metrics["trace_bound_rate"] == 1
        and normal_metrics["unfinished_or_invalid_prediction_rate"] == 0,
        "normal_no_reexecution_errors": normal_metrics[
            "evaluation_error_rate"
        ]
        == 0,
        "interventions_all_predictions_present": all(
            value["missing_prediction_rate"] == 0
            for value in intervention_metrics.values()
        ),
        "interventions_no_reexecution_errors": all(
            value["evaluation_error_rate"] == 0
            for value in intervention_metrics.values()
        ),
        "interventions_all_explicitly_finished": all(
            value["unfinished_or_invalid_prediction_rate"] == 0
            for value in intervention_metrics.values()
        ),
        "runtime_contract_consistent_within_artifacts": runtime_consistent_within,
        "runtime_contract_complete": runtime_complete,
        "same_model_adapter_and_generation_budget": runtime_matched_across
        and runtime_complete,
        "removed_observation_lengths_preserved": remove_audit.get(
            "length_preserved_for_all_rows"
        )
        is True,
        "shuffle_has_no_self_donors": shuffle_audit.get("self_donor_free")
        is True,
        "shuffle_donors_available": shuffle_audit.get(
            "all_called_tool_types_have_donors"
        )
        is True,
        "shuffle_contract_valid": shuffle_audit.get(
            "shuffle_contract_valid_for_all_rows"
        )
        is True,
    }
    required_names = {
        "remove_tool_observations",
        "stale_tool_observations",
        "shuffle_tool_observations",
    }
    missing_required = sorted(required_names - set(interventions))
    intervention_contract_valid = (
        integrity["removed_observation_lengths_preserved"]
        and integrity["shuffle_has_no_self_donors"]
        and integrity["shuffle_donors_available"]
        and integrity["shuffle_contract_valid"]
    )

    primary = paired.get(args.primary_intervention)
    primary_ci = (
        list(
            primary["structural_exact_paired_inference"][
                "paired_bootstrap_ci"
            ]
        )
        if primary
        else None
    )
    primary_adjusted_p = (
        primary["structural_exact_paired_inference"].get(
            "holm_adjusted_mcnemar_p_value"
        )
        if primary
        else None
    )
    primary_effect_gate = bool(
        primary_ci
        and float(primary_ci[0]) >= args.minimum_absolute_drop
        and primary_adjusted_p is not None
        and float(primary_adjusted_p) <= args.alpha
    )

    result = {
        "artifact_type": "frozen_causal_intervention_evaluation",
        "scientific_hypothesis": "H1_causal_faithfulness",
        "reference": {
            "path": str(args.reference),
            "sha256": file_sha256(args.reference),
            "n_ids": len(reference),
        },
        "sources": sources,
        "runtime_contracts": runtime_contracts,
        "runtime_contract_digests": runtime_digests,
        "normal": normal_metrics,
        "interventions": intervention_metrics,
        "intervention_audits": intervention_audits,
        "paired_effects": paired,
        "multiple_comparison_correction": {
            "method": "Holm family-wise error correction",
            "family": sorted(paired),
            "adjusted_mcnemar_p_values": adjusted_p_values,
            "alpha": args.alpha,
        },
        "integrity": integrity,
        "claim_gates": {
            "required_interventions_present": not missing_required,
            "missing_required_interventions": missing_required,
            "normal_path_is_fully_trace_bound": integrity[
                "normal_trace_binding_complete"
            ],
            "all_prediction_artifacts_complete": integrity[
                "normal_all_predictions_present"
            ]
            and integrity["interventions_all_predictions_present"],
            "all_trace_predictions_explicitly_finished": integrity[
                "normal_trace_binding_complete"
            ]
            and integrity["interventions_all_explicitly_finished"],
            "all_outputs_recompute_without_error": integrity[
                "normal_no_reexecution_errors"
            ]
            and integrity["interventions_no_reexecution_errors"],
            "same_runtime_contract": runtime_consistent_within
            and runtime_matched_across
            and runtime_complete,
            "runtime_contract_complete": runtime_complete,
            "intervention_contract_valid": intervention_contract_valid,
            "primary_intervention": args.primary_intervention,
            "primary_intervention_present": primary is not None,
            "primary_effect_ci_lower_at_least_minimum_drop": bool(
                primary_ci
                and float(primary_ci[0]) >= args.minimum_absolute_drop
            ),
            "primary_holm_mcnemar_significant": bool(
                primary_adjusted_p is not None
                and float(primary_adjusted_p) <= args.alpha
            ),
            "causal_sensitivity_observed": primary_effect_gate,
            "minimum_absolute_drop": args.minimum_absolute_drop,
            "confidence": args.confidence,
            "alpha": args.alpha,
        },
        "interpretation": {
            "positive_result": "The primary audited intervention causes a paired structural-accuracy drop whose bootstrap lower bound exceeds the declared minimum and whose exact McNemar test survives Holm correction under an identical runtime contract.",
            "negative_result": "Runtime mismatch, incomplete metadata, invalid intervention construction, missing finish_trace, a confidence interval crossing the minimum effect, or a non-significant corrected paired test blocks the causal tool-grounding claim.",
            "structural_metric": "Atom-contributing structural precursor exact match, ignoring atom-map labels.",
            "statistical_unit": "Frozen target identifier; model-training seed aggregation is performed separately with aggregate_evaluation_seeds.py.",
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
