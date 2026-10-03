#!/usr/bin/env python3
"""Evaluate one-shot electron-flow candidates after complete-program generation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.data_audit import sha256_file
from mechet.endpoints import (
    mapped_exact, reference_structural_precursor, split_precursor_endpoints,
    structural_exact,
)
from mechet.open_flow_program import execute_open_flow
from mechet.prediction_metrics import prediction_runtime_contract
from mechet.proof_program import sides_equal
from scripts.evaluate_proof_candidates import (
    _formal_nll_order, _full_precursor, _hit, _index, _load_rankings,
    _read_jsonl, _target,
)


def score_candidate(candidate: dict[str, Any], reference: dict[str, Any], index: int,
                    *, max_tool_calls: int = 40) -> dict[str, Any]:
    target = _target(reference)
    if not target:
        raise ValueError(f"missing target for {reference.get('id')}")
    execution = execute_open_flow(
        str(candidate.get("prediction") or ""), target,
        max_tool_calls=max_tool_calls,
    )
    execute_ok = bool(execution["execute_ok"])
    derived_full = str(execution.get("derived_precursor") or "")
    derived_structural = ""
    endpoint_error = ""
    if execute_ok:
        try:
            derived_structural = split_precursor_endpoints(derived_full, target).structural
        except Exception as exc:
            endpoint_error = str(exc)
    expected_structural = reference_structural_precursor(reference)
    expected_full = _full_precursor(reference)
    return {
        "candidate_index": index,
        "sample_index": int(candidate.get("sample_index") or index),
        "execute_ok": execute_ok,
        "derived_full_precursor": derived_full,
        "derived_structural_precursor": derived_structural,
        "structural_exact": bool(derived_structural and expected_structural and
                                 structural_exact(derived_structural, expected_structural)),
        "mapped_exact": bool(derived_structural and expected_structural and
                             mapped_exact(derived_structural, expected_structural)),
        "full_precursor_exact": bool(derived_full and expected_full and
                                     sides_equal(derived_full, expected_full, ignore_maps=True)),
        "failure_code": str(execution.get("failure_code") or ""),
        "endpoint_error": endpoint_error,
        "generation_error": str(candidate.get("generation_error") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ranking-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--max-tool-calls", type=int, default=40)
    args = parser.parse_args()

    references = _index(_read_jsonl(args.reference), "references")
    predictions = _index(_read_jsonl(args.predictions), "predictions")
    rankings = _load_rankings(args.ranking_dir)
    if args.expected_rows and len(references) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} references, got {len(references)}")
    if set(predictions) != set(references):
        raise ValueError("prediction/reference ID mismatch")
    if rankings and set(rankings) != set(references):
        raise ValueError("ranking/reference ID mismatch")

    ks = (1, 5, 10)
    metrics = ("structural_exact", "mapped_exact", "full_precursor_exact", "execute_ok")
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    row_path = args.output.with_suffix(".rows.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with row_path.open("w", encoding="utf-8") as handle:
        for identifier, reference in references.items():
            candidates = list(predictions[identifier].get("candidates") or [])
            if len(candidates) != args.expected_candidates:
                raise ValueError(f"{identifier} has {len(candidates)} candidates, expected {args.expected_candidates}")
            evaluated = [score_candidate(candidate, reference, index,
                                         max_tool_calls=args.max_tool_calls)
                         for index, candidate in enumerate(candidates)]
            generation_order = list(range(len(evaluated)))
            ranked_order: list[int] = []
            if rankings:
                ranked_order = _formal_nll_order(
                    evaluated, list(rankings[identifier].get("candidate_scores") or [])
                )
            for item in evaluated:
                if not item["execute_ok"]:
                    failures[item["failure_code"] or "UNCLASSIFIED"] += 1
            for k in ks:
                for metric in metrics:
                    counts[f"generation_{metric}_{k}"] += int(_hit(evaluated, generation_order, metric, k))
                    if ranked_order:
                        counts[f"ranked_{metric}_{k}"] += int(_hit(evaluated, ranked_order, metric, k))
            handle.write(json.dumps({
                "id": identifier,
                "source_id": str(reference.get("source_id") or ""),
                "generation_order": generation_order,
                "formal_nll_ranked_order": ranked_order,
                "candidates": evaluated,
            }, ensure_ascii=False) + "\n")
    denominator = len(references)

    def block(prefix: str) -> dict[str, float]:
        return {f"{metric}_at_{k}": counts[f"{prefix}_{metric}_{k}"] / max(denominator, 1)
                for k in ks for metric in metrics}

    report: dict[str, Any] = {
        "artifact_type": "open_flow_candidate_evaluation",
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256_file(args.reference),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "n_reference_rows": denominator,
        "candidates_per_target": args.expected_candidates,
        "runtime_contract": prediction_runtime_contract(
            list(predictions.values()), include_adapter=True,
        ),
        "max_tool_calls": args.max_tool_calls,
        "generation_order": {"semantics": "independent-sample Pass@K; not ranked Top-K", **block("generation")},
        "formal_nll_ranked": None,
        "execution_failure_codes": dict(failures),
        "row_evaluation": str(row_path.resolve()),
        "row_evaluation_sha256": sha256_file(row_path),
    }
    if rankings:
        report["formal_nll_ranked"] = {
            "semantics": "formal-execution gate, then gold-independent assistant mean-NLL",
            "ranker": "open_flow_execute_gate__assistant_mean_nll_v1",
            "selection_uses_ground_truth": False,
            **block("ranked"),
        }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
