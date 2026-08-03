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
    condition_metrics,
    file_sha256,
    read_jsonl,
)
from mechet.prediction_metrics import prediction_set_metrics

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


def find_condition(metrics: dict[str, dict[str, Any]], canonical: str) -> str | None:
    for candidate in ALIASES[canonical]:
        if candidate in metrics:
            return candidate
    return None


def delta(metrics, left, right, field):
    left_name = find_condition(metrics, left)
    right_name = find_condition(metrics, right)
    if left_name is None or right_name is None:
        return None
    return float(metrics[left_name].get(field, 0.0)) - float(metrics[right_name].get(field, 0.0))


def is_direct_condition(name: str) -> bool:
    return name in set(ALIASES["direct_textbook_rag"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.reference.exists():
        raise FileNotFoundError(args.reference)
    reference_rows = read_jsonl(args.reference)
    aligned: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for name, path in args.condition:
        if name in aligned:
            raise ValueError(f"duplicate condition name: {name}")
        if not path.exists():
            raise FileNotFoundError(path)
        prediction_rows = read_jsonl(path)
        aligned[name] = align_prediction_artifact(reference_rows, prediction_rows, condition_name=name)
        sources[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "n_prediction_rows": len(prediction_rows),
        }

    metrics = {
        name: {
            **condition_metrics(rows),
            **prediction_set_metrics(rows),
        }
        for name, rows in aligned.items()
    }
    primary_field = "structural_exact_rate"
    textbook_vs_none = delta(metrics, "trace_textbook_rag", "trace_no_knowledge", primary_field)
    textbook_vs_irrelevant = delta(metrics, "trace_textbook_rag", "trace_length_matched_irrelevant", primary_field)
    combined_vs_textbook = delta(metrics, "trace_text_plus_anchors", "trace_textbook_rag", primary_field)
    combined_vs_anchors = delta(metrics, "trace_text_plus_anchors", "trace_structured_anchors", primary_field)
    trace_textbook_vs_direct = delta(metrics, "trace_textbook_rag", "direct_textbook_rag", primary_field)

    reward_violations = sum(int(value.get("knowledge_direct_reward_violations", 0)) for value in metrics.values())
    trace_names = [name for name in metrics if not is_direct_condition(name)]
    trace_binding_ok = bool(trace_names) and all(
        float(metrics[name].get("trace_prediction_rate", 0.0)) == 1.0
        and float(metrics[name].get("trace_bound_rate", 0.0)) == 1.0
        and float(metrics[name].get("missing_prediction_rate", 1.0)) == 0.0
        for name in trace_names
    )
    all_predictions_present = all(float(value.get("missing_prediction_rate", 1.0)) == 0.0 for value in metrics.values())
    no_evaluation_errors = all(float(value.get("evaluation_error_rate", 1.0)) == 0.0 for value in metrics.values())

    claim_gates = {
        "all_frozen_ids_evaluated": all_predictions_present,
        "no_reexecution_errors": no_evaluation_errors,
        "textbook_exceeds_trace_only": None if textbook_vs_none is None else textbook_vs_none > 0,
        "textbook_exceeds_irrelevant_context": None if textbook_vs_irrelevant is None else textbook_vs_irrelevant > 0,
        "combined_exceeds_each_individual": (
            None
            if combined_vs_textbook is None or combined_vs_anchors is None
            else combined_vs_textbook > 0 and combined_vs_anchors > 0
        ),
        "trace_binding_preserved": trace_binding_ok,
        "zero_direct_evidence_reward_violations": reward_violations == 0,
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
        "n_reference_ids": len(reference_rows),
        "conditions": metrics,
        "contrasts": {
            "textbook_minus_trace_only_structural_exact": textbook_vs_none,
            "textbook_minus_irrelevant_structural_exact": textbook_vs_irrelevant,
            "combined_minus_textbook_structural_exact": combined_vs_textbook,
            "combined_minus_anchors_structural_exact": combined_vs_anchors,
            "trace_textbook_minus_direct_textbook_structural_exact": trace_textbook_vs_direct,
        },
        "claim_gates": claim_gates,
        "integrity_passed": (
            all_predictions_present
            and no_evaluation_errors
            and trace_binding_ok
            and reward_violations == 0
        ),
        "metric_contract": {
            "primary_endpoint": "atom-contributing structural precursor exact match with atom maps ignored",
            "secondary_endpoint": "mapped structural precursor exact match",
            "top_k": "computed from retained candidate rollouts without gold-based reranking",
            "selective_risk": "error among non-abstained selected predictions",
            "trace_metrics": "recomputed from rollout flow_trace or re-executed compiled proof",
            "missing_predictions": "retained in the denominator as failures",
            "extra_or_duplicate_ids": "hard error",
            "training_rows_as_predictions": "hard error",
            "unavailable_metrics": "reaction-center and synthon metrics remain null until frozen labels exist",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
