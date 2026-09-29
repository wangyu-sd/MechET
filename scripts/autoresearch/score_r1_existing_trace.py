#!/usr/bin/env python3
"""Diagnostic multi-reference R1 rescoring of an existing MechET trace run.

Only executor-owned, formally executed finish_trace precursors count as
mechanistic support. A precursor absent from recorded references is *not*
automatically chemically unsupported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


def candidate_precursor(candidate: dict[str, Any]) -> str | None:
    if candidate.get("termination_reason") != "terminal_tool":
        return None
    state = candidate.get("rollout_state") or {}
    final = state.get("final_result") or {}
    if not (final.get("ok") is True and final.get("formal_execute") is True
            and final.get("trace_bound") is True
            and final.get("endpoint_source") == "environment_owned_trace"):
        return None
    raw = final.get("structural_precursor")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return product_key(raw)
    except ValueError:
        return None


def build(r1_cohort: Path, r1_manifest: Path, predictions: Path,
          prediction_report: Path, output: Path, *,
          expected_reference_rows: int = 28967) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R1 existing-trace diagnostic already frozen: {output}")
    r1_meta = json.loads(r1_manifest.read_text())
    if digest(r1_cohort) != r1_meta["cohort_sha256"]:
        raise ValueError("R1 multi-reference source hash drifted")
    prediction_meta = json.loads(prediction_report.read_text())
    if digest(predictions) != prediction_meta["predictions_sha256"]:
        raise ValueError("R1 trace prediction artifact hash drifted")
    if prediction_meta["n_reference_rows"] != expected_reference_rows:
        raise ValueError("R1 trace prediction run is not the expected strict test")
    references: dict[str, set[str]] = {}
    with r1_cohort.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            if product_key(product) != product or product in references:
                raise ValueError("R1 has noncanonical or duplicate products")
            references[product] = {str(item["precursor_smiles"])
                                   for item in row["references"]}
            if len(references[product]) < 2:
                raise ValueError("R1 row is not multi-reference")

    representative: dict[str, dict[str, Any]] = {}
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(str(row["target_smiles"]))
            if product not in references:
                continue
            if row.get("prediction_status") != "completed":
                raise ValueError("R1 diagnostic requires completed trace predictions")
            row_id = str(row["id"])
            if product not in representative or row_id < str(representative[product]["id"]):
                representative[product] = row

    records = []
    for product, refs in sorted(references.items()):
        prediction = representative.get(product)
        if prediction is None:
            records.append({"artifact_type": "r1_existing_trace_diagnostic_row_v1",
                            "product_smiles": product, "prediction_id": None,
                            "candidate_count": 0, "formally_executed_candidates": 0,
                            "formal_at_1": False, "formal_at_k": False,
                            "recorded_hit_at_1": False, "recorded_hit_at_k": False,
                            "nonreference_formal_candidates": 0})
            continue
        candidates = sorted(prediction["candidates"],
                            key=lambda item: int(item["sample_index"]))
        indices = [item["sample_index"] for item in candidates]
        if indices != list(range(len(candidates))):
            raise ValueError("R1 trace candidate indices are incomplete")
        precursors = [candidate_precursor(item) for item in candidates]
        first = precursors[0] if precursors else None
        records.append({
            "artifact_type": "r1_existing_trace_diagnostic_row_v1",
            "product_smiles": product, "prediction_id": str(prediction["id"]),
            "candidate_count": len(precursors),
            "formally_executed_candidates": sum(item is not None for item in precursors),
            "formal_at_1": first is not None,
            "formal_at_k": any(item is not None for item in precursors),
            "recorded_hit_at_1": first in refs if first else False,
            "recorded_hit_at_k": any(item in refs for item in precursors if item),
            "nonreference_formal_candidates": sum(item is not None and item not in refs
                                                  for item in precursors),
        })
    output.mkdir(parents=True)
    rows_path = output / "r1_existing_trace_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    n = len(records)
    totals = {name: sum(bool(row[name]) for row in records)
              for name in ("formal_at_1", "formal_at_k", "recorded_hit_at_1",
                           "recorded_hit_at_k")}
    report = {
        "artifact_type": "r1_existing_trace_diagnostic_v1",
        "status": "diagnostic_only_not_scientific_smoke",
        "r1_cohort_sha256": digest(r1_cohort),
        "existing_predictions_sha256": digest(predictions),
        "rows_sha256": digest(rows_path),
        "products": n,
        "products_with_prediction_row": sum(row["prediction_id"] is not None for row in records),
        "candidate_counts": {str(k): sum(row["candidate_count"] == k for row in records)
                             for k in sorted({row["candidate_count"] for row in records})},
        "formally_executed_candidates": sum(row["formally_executed_candidates"]
                                             for row in records),
        "nonreference_formal_candidates": sum(row["nonreference_formal_candidates"]
                                                for row in records),
        "totals": totals,
        "rates": {name: value / n for name, value in totals.items()},
        "claim_boundary": "Existing compact-full-state MechET checkpoint on frozen R1 products; not a paired Base-vs-Mech smoke, and formal execution or non-reference output alone does not establish chemical truth.",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-cohort", type=Path, required=True)
    parser.add_argument("--r1-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.r1_cohort, args.r1_manifest,
                           args.predictions, args.prediction_report,
                           args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
