#!/usr/bin/env python3
"""Evaluate a sampled prediction artifact against a frozen reference split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechet.knowledge_ablation import (
    align_prediction_artifact,
    condition_metrics,
    file_sha256,
    read_jsonl,
)
from mechet.prediction_metrics import prediction_runtime_contract, prediction_set_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--expected-candidates", type=int, default=0)
    args = parser.parse_args()

    references = read_jsonl(args.reference)
    predictions = read_jsonl(args.predictions)
    if args.expected_rows and len(references) != args.expected_rows:
        raise ValueError(
            f"reference row count mismatch: expected {args.expected_rows}, got {len(references)}"
        )
    if args.expected_rows and len(predictions) != args.expected_rows:
        raise ValueError(
            f"prediction row count mismatch: expected {args.expected_rows}, got {len(predictions)}"
        )

    candidate_counts = [len(row.get("candidates") or [row]) for row in predictions]
    if args.expected_candidates and any(
        count != args.expected_candidates for count in candidate_counts
    ):
        bad = sum(count != args.expected_candidates for count in candidate_counts)
        raise ValueError(
            f"{bad} prediction rows do not contain exactly "
            f"{args.expected_candidates} candidates"
        )

    aligned = align_prediction_artifact(
        references, predictions, condition_name=args.condition_name
    )
    report = {
        "artifact_type": "sampled_prediction_evaluation",
        "condition_name": args.condition_name,
        "reference": str(args.reference.resolve()),
        "reference_sha256": file_sha256(args.reference),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": file_sha256(args.predictions),
        "n_reference_rows": len(references),
        "n_prediction_rows": len(predictions),
        "candidate_count_min": min(candidate_counts, default=0),
        "candidate_count_max": max(candidate_counts, default=0),
        "headline": {
            **condition_metrics(aligned),
            **prediction_set_metrics(aligned),
        },
        "runtime_contract": prediction_runtime_contract(
            predictions, include_adapter=True
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
