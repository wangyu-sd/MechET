#!/usr/bin/env python3
"""Evaluate H3 predictions against one frozen reference universe."""
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
from mechet.prediction_metrics import (
    prediction_runtime_contract,
    prediction_set_metrics,
)
from mechet.statistical_evaluation import holm_adjust, paired_binary_contrast
from mechet.strict_prediction_evaluation import condition_metrics, endpoint_evaluation

ALIASES = {
    "trace_no_knowledge": ("trace_no_knowledge", "none", "trace_none"),
    "trace_length_matched_irrelevant": ("trace_length_matched_irrelevant", "irrelevant"),
    "trace_textbook_rag": ("trace_textbook_rag", "textbook"),
    "trace_structured_anchors": ("trace_structured_anchors", "anchors"),
    "trace_text_plus_anchors": ("trace_text_plus_anchors", "combined"),
    "direct_textbook_rag": ("direct_textbook_rag", "direct"),
}


def parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must be NAME=PATH")
    return name, Path(path)


def find_condition(metrics: dict[str, Any], canonical: str) -> str | None:
    for candidate in ALIASES[canonical]:
        if candidate in metrics:
            return candidate
    return None


def delta(metrics, left, right, field):
    left_name = find_condition(metrics, left)
    right_name = find_condition(metrics, right)
    if left_name is None or right_name is None:
        return None
    return float(metrics[left_name].get(field, 0.0)) - float(
        metrics[right_name].get(field, 0.0)
    )


def is_direct_condition(name: str) -> bool:
    return name in set(ALIASES["direct_textbook_rag"])


def _paired_condition_contrast(
    aligned: dict[str, list[dict[str, Any]]],
    left: str,
    right: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any] | None:
    left_name = find_condition(aligned, left)
    right_name = find_condition(aligned, right)
    if left_name is None or right_name is None:
        return None
    left_rows = {row_id(row): endpoint_evaluation(row) for row in aligned[left_name]}
    right_rows = {
        row_id(row): endpoint_evaluation(row) for row in aligned[right_name]
    }
    if set(left_rows) != set(right_rows):
        raise ValueError(f"paired H3 conditions differ in IDs: {left} vs {right}")
    identifiers = list(left_rows)
    inference = paired_binary_contrast(
        [bool(left_rows[item]["structural_exact"]) for item in identifiers],
        [bool(right_rows[item]["structural_exact"]) for item in identifiers],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    return {
        "left_condition": left_name,
        "right_condition": right_name,
        "metric": "structural_exact",
        **inference,
    }


def _contrast_gate(
    contrast: dict[str, Any] | None,
    *,
    alpha: float,
    minimum_effect: float,
) -> bool | None:
    if contrast is None:
        return None
    ci = list(contrast["paired_bootstrap_ci"])
    adjusted = contrast.get("holm_adjusted_mcnemar_p_value")
    return bool(
        float(ci[0]) >= minimum_effect
        and adjusted is not None
        and float(adjusted) <= alpha
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=31)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--minimum-effect", type=float, default=0.0)
    args = parser.parse_args()

    if not args.reference.exists():
        raise FileNotFoundError(args.reference)
    reference_rows = read_jsonl(args.reference)
    aligned: dict[str, list[dict[str, Any]]] = {}
    raw_predictions: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for name, path in args.condition:
        if name in aligned:
            raise ValueError(f"duplicate condition name: {name}")
        if not path.exists():
            raise FileNotFoundError(path)
        prediction_rows = read_jsonl(path)
        raw_predictions[name] = prediction_rows
        aligned[name] = align_prediction_artifact(
            reference_rows, prediction_rows, condition_name=name
        )
        sources[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "n_prediction_rows": len(prediction_rows),
        }

    metrics = {
        name: {**condition_metrics(rows), **prediction_set_metrics(rows)}
        for name, rows in aligned.items()
    }
    runtime_contracts = {
        name: prediction_runtime_contract(rows, include_adapter=False)
        for name, rows in raw_predictions.items()
    }
    runtime_consistent_within = all(
        value["runtime_contract_consistent"] for value in runtime_contracts.values()
    )
    runtime_complete = all(
        value["runtime_contract_complete"] for value in runtime_contracts.values()
    )
    runtime_digests = {
        name: value["runtime_contract_sha256"]
        for name, value in runtime_contracts.items()
    }
    generation_contract_matched = len(set(runtime_digests.values())) == 1
    adapter_lineage = {
        name: sorted(
            {
                str((row.get("model") or {}).get("adapter_sha256") or "")
                for row in rows
            }
        )
        for name, rows in raw_predictions.items()
    }

    primary_field = "structural_exact_rate"
    textbook_vs_none = delta(
        metrics, "trace_textbook_rag", "trace_no_knowledge", primary_field
    )
    textbook_vs_irrelevant = delta(
        metrics,
        "trace_textbook_rag",
        "trace_length_matched_irrelevant",
        primary_field,
    )
    combined_vs_textbook = delta(
        metrics, "trace_text_plus_anchors", "trace_textbook_rag", primary_field
    )
    combined_vs_anchors = delta(
        metrics,
        "trace_text_plus_anchors",
        "trace_structured_anchors",
        primary_field,
    )
    trace_textbook_vs_direct = delta(
        metrics, "trace_textbook_rag", "direct_textbook_rag", primary_field
    )

    contrast_specs = {
        "textbook_minus_trace_only": (
            "trace_textbook_rag",
            "trace_no_knowledge",
        ),
        "textbook_minus_irrelevant": (
            "trace_textbook_rag",
            "trace_length_matched_irrelevant",
        ),
        "combined_minus_textbook": (
            "trace_text_plus_anchors",
            "trace_textbook_rag",
        ),
        "combined_minus_anchors": (
            "trace_text_plus_anchors",
            "trace_structured_anchors",
        ),
        "trace_textbook_minus_direct": (
            "trace_textbook_rag",
            "direct_textbook_rag",
        ),
    }
    paired_contrasts = {
        name: _paired_condition_contrast(
            aligned,
            left,
            right,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index * 13,
            confidence=args.confidence,
        )
        for index, (name, (left, right)) in enumerate(contrast_specs.items())
    }
    primary_family = {
        name: value
        for name, value in paired_contrasts.items()
        if name
        in {
            "textbook_minus_trace_only",
            "textbook_minus_irrelevant",
            "combined_minus_textbook",
            "combined_minus_anchors",
        }
    }
    adjusted = holm_adjust(
        {
            name: (
                None
                if value is None
                else float(value["mcnemar_exact_p_value"])
            )
            for name, value in primary_family.items()
        }
    )
    for name, adjusted_value in adjusted.items():
        if paired_contrasts[name] is not None:
            paired_contrasts[name][
                "holm_adjusted_mcnemar_p_value"
            ] = adjusted_value

    reward_violations = sum(
        int(value.get("knowledge_direct_reward_violations", 0))
        for value in metrics.values()
    )
    trace_names = [name for name in metrics if not is_direct_condition(name)]
    trace_binding_ok = bool(trace_names) and all(
        float(metrics[name].get("trace_prediction_rate", 0.0)) == 1.0
        and float(metrics[name].get("trace_bound_rate", 0.0)) == 1.0
        and float(metrics[name].get("missing_prediction_rate", 1.0)) == 0.0
        and float(
            metrics[name].get("unfinished_or_invalid_prediction_rate", 1.0)
        )
        == 0.0
        for name in trace_names
    )
    all_predictions_present = all(
        float(value.get("missing_prediction_rate", 1.0)) == 0.0
        for value in metrics.values()
    )
    no_evaluation_errors = all(
        float(value.get("evaluation_error_rate", 1.0)) == 0.0
        for value in metrics.values()
    )

    textbook_gate_none = _contrast_gate(
        paired_contrasts["textbook_minus_trace_only"],
        alpha=args.alpha,
        minimum_effect=args.minimum_effect,
    )
    textbook_gate_irrelevant = _contrast_gate(
        paired_contrasts["textbook_minus_irrelevant"],
        alpha=args.alpha,
        minimum_effect=args.minimum_effect,
    )
    combined_gate_textbook = _contrast_gate(
        paired_contrasts["combined_minus_textbook"],
        alpha=args.alpha,
        minimum_effect=args.minimum_effect,
    )
    combined_gate_anchors = _contrast_gate(
        paired_contrasts["combined_minus_anchors"],
        alpha=args.alpha,
        minimum_effect=args.minimum_effect,
    )

    claim_gates = {
        "all_frozen_ids_evaluated": all_predictions_present,
        "no_reexecution_errors": no_evaluation_errors,
        "runtime_contract_consistent_within_artifacts": runtime_consistent_within,
        "runtime_contract_complete": runtime_complete,
        "same_base_model_revision_and_generation_budget": (
            generation_contract_matched and runtime_complete
        ),
        "textbook_exceeds_trace_only": textbook_gate_none,
        "textbook_exceeds_irrelevant_context": textbook_gate_irrelevant,
        "textbook_evidence_claim": (
            None
            if textbook_gate_none is None or textbook_gate_irrelevant is None
            else textbook_gate_none and textbook_gate_irrelevant
        ),
        "combined_exceeds_textbook": combined_gate_textbook,
        "combined_exceeds_anchors": combined_gate_anchors,
        "combined_exceeds_each_individual": (
            None
            if combined_gate_textbook is None or combined_gate_anchors is None
            else combined_gate_textbook and combined_gate_anchors
        ),
        "trace_binding_preserved": trace_binding_ok,
        "zero_direct_evidence_reward_violations": reward_violations == 0,
        "paired_bootstrap_confidence": args.confidence,
        "holm_corrected_alpha": args.alpha,
        "minimum_effect": args.minimum_effect,
        "multi_seed_aggregation_required_for_final_claim": True,
        "causal_interventions_required_for_final_claim": True,
    }

    result = {
        "artifact_type": "frozen_prediction_evaluation",
        "scientific_hypothesis": "H3_empirical_evidence_separation",
        "reference": {
            "path": str(args.reference),
            "sha256": file_sha256(args.reference),
            "n_ids": len(reference_rows),
        },
        "prediction_sources": sources,
        "runtime_contracts": runtime_contracts,
        "runtime_contract_digests": runtime_digests,
        "adapter_lineage_by_condition": adapter_lineage,
        "n_reference_ids": len(reference_rows),
        "conditions": metrics,
        "contrasts": {
            "textbook_minus_trace_only_structural_exact": textbook_vs_none,
            "textbook_minus_irrelevant_structural_exact": textbook_vs_irrelevant,
            "combined_minus_textbook_structural_exact": combined_vs_textbook,
            "combined_minus_anchors_structural_exact": combined_vs_anchors,
            "trace_textbook_minus_direct_textbook_structural_exact": trace_textbook_vs_direct,
        },
        "paired_contrasts": paired_contrasts,
        "multiple_comparison_correction": {
            "method": "Holm family-wise error correction",
            "family": sorted(primary_family),
            "adjusted_mcnemar_p_values": adjusted,
            "alpha": args.alpha,
        },
        "claim_gates": claim_gates,
        "integrity_passed": (
            all_predictions_present
            and no_evaluation_errors
            and trace_binding_ok
            and reward_violations == 0
            and runtime_consistent_within
            and runtime_complete
            and generation_contract_matched
        ),
        "metric_contract": {
            "primary_endpoint": "atom-contributing structural precursor exact match with atom maps ignored",
            "secondary_endpoint": "mapped structural precursor exact match",
            "candidate_sets": "generation-order Pass@K without gold-based reranking",
            "paired_statistics": "paired bootstrap confidence intervals and exact McNemar tests over frozen target IDs",
            "multiple_comparison_control": "Holm correction over the four primary H3 contrasts",
            "multi_seed_statistics": "aggregate independent training seeds with aggregate_evaluation_seeds.py before a final scientific claim",
            "selective_risk": "error among non-abstained selected predictions",
            "runtime_matching": "base model, revision, tokenizer revision, seed, selector, temperature, top_p, max tokens, iterations, and K must be present and matched; adapter lineage is condition-specific",
            "trace_metrics": "require an explicit successful finish_trace and are then recomputed from the rollout flow_trace",
            "unfinished_traces": "counted as prediction failures; the evaluator never completes them",
            "trace_to_direct_fallback": "forbidden",
            "missing_predictions": "retained in the denominator as failures",
            "extra_or_duplicate_ids": "hard error",
            "training_rows_as_predictions": "hard error",
            "unavailable_metrics": "reaction-center and synthon metrics remain null until frozen labels exist",
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
