#!/usr/bin/env python3
"""Evaluate endpoint, trace and evidence-use metrics across matched conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import (
    condition_metrics,
    matched_intersection,
    read_jsonl,
)

ALIASES = {
    "trace_no_knowledge": ("trace_no_knowledge", "none", "trace_none"),
    "trace_length_matched_irrelevant": (
        "trace_length_matched_irrelevant",
        "irrelevant",
    ),
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


def delta(
    metrics: dict[str, dict[str, Any]],
    left: str,
    right: str,
    field: str,
) -> float | None:
    left_name = find_condition(metrics, left)
    right_name = find_condition(metrics, right)
    if left_name is None or right_name is None:
        return None
    return float(metrics[left_name].get(field, 0.0)) - float(
        metrics[right_name].get(field, 0.0)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = {name: read_jsonl(path) for name, path in args.condition}
    identifiers, matched = matched_intersection(loaded)
    metrics = {name: condition_metrics(rows) for name, rows in matched.items()}

    textbook_vs_none = delta(
        metrics,
        "trace_textbook_rag",
        "trace_no_knowledge",
        "endpoint_exact_rate",
    )
    textbook_vs_irrelevant = delta(
        metrics,
        "trace_textbook_rag",
        "trace_length_matched_irrelevant",
        "endpoint_exact_rate",
    )
    combined_vs_textbook = delta(
        metrics,
        "trace_text_plus_anchors",
        "trace_textbook_rag",
        "endpoint_exact_rate",
    )
    combined_vs_anchors = delta(
        metrics,
        "trace_text_plus_anchors",
        "trace_structured_anchors",
        "endpoint_exact_rate",
    )
    trace_textbook_vs_direct = delta(
        metrics,
        "trace_textbook_rag",
        "direct_textbook_rag",
        "endpoint_exact_rate",
    )

    reward_violations = sum(
        int(value.get("knowledge_direct_reward_violations", 0))
        for value in metrics.values()
    )
    trace_names = [
        name
        for name, value in metrics.items()
        if float(value.get("trace_prediction_rate", 0.0)) > 0
    ]
    trace_binding_ok = all(
        float(metrics[name].get("trace_bound_rate", 0.0)) == 1.0
        for name in trace_names
    )

    claim_gates = {
        "textbook_exceeds_trace_only": (
            None if textbook_vs_none is None else textbook_vs_none > 0
        ),
        "textbook_exceeds_irrelevant_context": (
            None if textbook_vs_irrelevant is None else textbook_vs_irrelevant > 0
        ),
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
        "scientific_hypothesis": "H3_empirical_evidence_separation",
        "n_matched_ids": len(identifiers),
        "conditions": metrics,
        "contrasts": {
            "textbook_minus_trace_only_endpoint_exact": textbook_vs_none,
            "textbook_minus_irrelevant_endpoint_exact": textbook_vs_irrelevant,
            "combined_minus_textbook_endpoint_exact": combined_vs_textbook,
            "combined_minus_anchors_endpoint_exact": combined_vs_anchors,
            "trace_textbook_minus_direct_textbook_endpoint_exact": trace_textbook_vs_direct,
        },
        "claim_gates": claim_gates,
        "required_interpretation": {
            "prediction_artifacts": (
                "Evaluate model prediction files, not gold training rows; direct "
                "conditions are parsed from PRECURSOR:/ANSWER: or explicit prediction fields."
            ),
            "textbook_gain": "textbook RAG minus trace-only under matched IDs",
            "irrelevant_context_control": (
                "textbook RAG minus exact length-matched irrelevant context"
            ),
            "structured_gain": "structured anchors minus trace-only",
            "combined_gain": "combined evidence minus each individual condition",
            "architecture_contrast": (
                "trace-owned and direct models receive the same bounded evidence card"
            ),
            "faithfulness_gate": (
                "trace_bound_rate remains 1.0 for trace-owned prediction conditions"
            ),
            "evidence_reward_gate": (
                "knowledge_direct_reward_violations remains zero"
            ),
            "causal_gate": (
                "passage and tool-observation interventions are reported separately"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
