#!/usr/bin/env python3
"""Validate, merge and full-denominator-complete RetroBridge worker outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identifier(row: dict) -> str:
    return str(row.get("id") or row.get("stable_id") or "")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    references = read_jsonl(Path(plan["reference"]))
    reference_by_id = {identifier(row): row for row in references}
    reference_ids = [identifier(row) for row in references]
    native_rows: list[dict] = []
    trace_rows: list[dict] = []
    for worker in plan["workers"]:
        expected_indices = json.loads(Path(worker["indices"]).read_text(encoding="utf-8"))
        predictions = read_jsonl(Path(worker["predictions"]))
        traces = read_jsonl(Path(worker["traces"]))
        if len(predictions) != len(expected_indices) or len(traces) != len(expected_indices):
            raise ValueError(
                f"Worker {worker['worker_index']} is incomplete: "
                f"expected {len(expected_indices)}, got {len(predictions)}/{len(traces)}"
            )
        if [row["source_index"] for row in predictions] != expected_indices:
            raise ValueError(f"Worker {worker['worker_index']} source indices mismatch")
        if [row["stable_id"] for row in predictions] != [
            row["stable_id"] for row in traces
        ]:
            raise ValueError(f"Worker {worker['worker_index']} trace IDs mismatch")
        native_rows.extend(predictions)
        trace_rows.extend(traces)

    native_by_id = {row["stable_id"]: row for row in native_rows}
    if len(native_by_id) != len(native_rows):
        raise ValueError("Merged native predictions contain duplicate IDs")
    excluded_ids = list(plan["excluded_stable_ids"])
    expected_native_ids = [value for value in reference_ids if value not in set(excluded_ids)]
    if set(native_by_id) != set(expected_native_ids):
        raise ValueError("Merged native prediction IDs do not match the planned compatible set")
    native_rows = [native_by_id[value] for value in expected_native_ids]
    trace_by_id = {row["stable_id"]: row for row in trace_rows}
    trace_rows = [trace_by_id[value] for value in expected_native_ids]

    candidate_count = int(plan["candidates_per_target"])
    checkpoint = plan["checkpoint"]
    full_rows = []
    for source_index, stable_id in enumerate(reference_ids):
        if stable_id in native_by_id:
            full_rows.append(native_by_id[stable_id])
            continue
        reference = reference_by_id[stable_id]
        product = str(reference.get("product") or reference.get("product_unmapped") or "")
        candidates = [
            {
                "rank": sample_index + 1,
                "sample_index": sample_index,
                "precursors": "",
                "prediction": "",
                "score": None,
                "generation_error": "retrobridge_incompatible_input",
            }
            for sample_index in range(candidate_count)
        ]
        full_rows.append(
            {
                "id": stable_id,
                "stable_id": stable_id,
                "source_index": source_index,
                "product": product,
                "reference_precursors": "",
                "candidates": candidates,
                "runtime_ms": 0.0,
                "source_method": "RetroBridge",
                "checkpoint": checkpoint,
                "candidate_budget": candidate_count,
                "generated_candidate_count": 0,
                "valid_candidate_count": 0,
                "candidate_semantics": "independent_stochastic_samples",
                "ranking": "generation_order",
                "score_type": "retrobridge_native_score",
                "generation_error": "retrobridge_incompatible_input",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    native_reference = [reference_by_id[value] for value in expected_native_ids]
    write_jsonl(args.output_dir / "reference.native_compatible.jsonl", native_reference)
    write_jsonl(args.output_dir / "predictions.native_compatible.jsonl", native_rows)
    write_jsonl(args.output_dir / "predictions.jsonl", full_rows)
    write_jsonl(args.output_dir / "inference_traces.jsonl", trace_rows)
    valid_candidates = sum(int(row["valid_candidate_count"]) for row in native_rows)
    report = {
        "artifact_type": "retrobridge_inference_merge_v1",
        "plan": str(args.plan.resolve()),
        "reference_rows": len(reference_ids),
        "native_compatible_rows": len(native_rows),
        "full_denominator_rows": len(full_rows),
        "excluded_incompatible_rows_scored_as_failures": len(excluded_ids),
        "excluded_stable_ids": excluded_ids,
        "candidates_per_target": candidate_count,
        "native_generated_candidates": len(native_rows) * candidate_count,
        "native_valid_candidates": valid_candidates,
        "native_candidate_validity_rate": valid_candidates
        / max(len(native_rows) * candidate_count, 1),
        "checkpoint": checkpoint,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "checkpoint_epoch": plan["checkpoint_epoch"],
        "checkpoint_global_step": plan["checkpoint_global_step"],
        "sampling_steps": plan["sampling_steps"],
        "sampling_seed_per_worker": plan["sampling_seed_per_worker"],
    }
    (args.output_dir / "merge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
