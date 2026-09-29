#!/usr/bin/env python3
"""Diagnostic R1 multi-reference rescoring of an existing direct-model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mechet.metrics import extract_answer_from_prediction
from scripts.autoresearch.stratified_manifest import digest, product_key


def canonical_answer(text: str) -> str | None:
    answer = extract_answer_from_prediction(text)
    if not answer:
        return None
    try:
        return product_key(answer)
    except ValueError:
        return None


def score_one(single_reference: str, all_references: set[str],
              candidate_texts: list[str]) -> dict[str, Any]:
    if not candidate_texts:
        raise ValueError("R1 diagnostic requires at least one candidate")
    predictions = [canonical_answer(text) for text in candidate_texts]
    first = predictions[0]
    return {
        "candidate_count": len(predictions),
        "parseable_candidates": sum(item is not None for item in predictions),
        "single_reference_at_1": first == single_reference,
        "multi_reference_at_1": first in all_references,
        "single_reference_at_k": single_reference in predictions,
        "multi_reference_at_k": any(item in all_references for item in predictions),
    }


def build(r1_cohort: Path, r1_manifest: Path, official_test: Path,
          official_manifest: Path, predictions: Path, prediction_report: Path,
          output: Path, *, expected_reactions: int = 28971) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R1 diagnostic already frozen: {output}")
    r1_meta = json.loads(r1_manifest.read_text())
    if digest(r1_cohort) != r1_meta["cohort_sha256"]:
        raise ValueError("R1 cohort hash drifted")
    official_meta = json.loads(official_manifest.read_text())
    if digest(official_test) != official_meta["splits"]["test"]["output_sha256"]:
        raise ValueError("R1 official test source hash drifted")
    prediction_meta = json.loads(prediction_report.read_text())
    if digest(predictions) != prediction_meta["predictions_sha256"]:
        raise ValueError("R1 prediction artifact hash drifted")
    if prediction_meta["n_reference_rows"] != expected_reactions:
        raise ValueError("R1 existing direct run is not complete official FlowER test")

    r1: dict[str, set[str]] = {}
    with r1_cohort.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            if product in r1:
                raise ValueError("duplicate R1 product")
            r1[product] = {str(item["precursor_smiles"]) for item in row["references"]}
            if len(r1[product]) < 2:
                raise ValueError("R1 row lacks alternative reference")

    single_gold: dict[str, str] = {}
    with official_test.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(row["target_smiles"])
            if product in r1:
                single_gold[str(row["id"])] = product_key(row["structural_precursor"])

    representative: dict[str, dict[str, Any]] = {}
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(str(row["target_smiles"]))
            if product not in r1:
                continue
            row_id = str(row["id"])
            if row_id not in single_gold:
                raise ValueError("R1 prediction ID is not in official test source")
            if row.get("prediction_status") != "completed":
                raise ValueError("R1 diagnostic requires complete predictions")
            if product not in representative or row_id < representative[product]["id"]:
                representative[product] = row
    if set(representative) != set(r1):
        raise ValueError("R1 cohort has missing existing-model predictions")

    records = []
    for product, references in sorted(r1.items()):
        row = representative[product]
        row_id = str(row["id"])
        single = single_gold[row_id]
        if single not in references:
            raise ValueError("selected single reference is not in R1 reference set")
        candidates = sorted(row["candidates"], key=lambda item: int(item["sample_index"]))
        if [item["sample_index"] for item in candidates] != list(range(len(candidates))):
            raise ValueError("R1 candidate sample indices are incomplete")
        metrics = score_one(single, references,
                            [str(item.get("prediction") or "") for item in candidates])
        records.append({
            "artifact_type": "r1_existing_direct_diagnostic_row_v1",
            "product_smiles": product, "prediction_id": row_id,
            "single_reference_precursors": single,
            "recorded_reference_count": len(references),
            "metrics": metrics,
        })
    output.mkdir(parents=True)
    rows_path = output / "r1_existing_direct_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    n = len(records)
    totals = {name: sum(bool(row["metrics"][name]) for row in records)
              for name in ("single_reference_at_1", "multi_reference_at_1",
                           "single_reference_at_k", "multi_reference_at_k")}
    report = {
        "artifact_type": "r1_existing_direct_diagnostic_v1",
        "status": "diagnostic_only_not_scientific_smoke",
        "r1_cohort_sha256": digest(r1_cohort),
        "official_test_sha256": digest(official_test),
        "existing_predictions_sha256": digest(predictions),
        "rows_sha256": digest(rows_path),
        "products": n,
        "candidates_per_product": {str(count): sum(row["metrics"]["candidate_count"] == count
                                                  for row in records)
                                   for count in sorted({row["metrics"]["candidate_count"] for row in records})},
        "totals": totals,
        "rates": {name: value / n for name, value in totals.items()},
        "single_reference_false_negatives_recovered_at_1": sum(
            row["metrics"]["multi_reference_at_1"] and
            not row["metrics"]["single_reference_at_1"] for row in records),
        "single_reference_false_negatives_recovered_at_k": sum(
            row["metrics"]["multi_reference_at_k"] and
            not row["metrics"]["single_reference_at_k"] for row in records),
        "claim_boundary": "Existing FlowER-trained direct model on 220 multi-recorded official-test products; no MechET Base-vs-Mech contrast, forward round-trip or chemical validation.",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-cohort", type=Path, required=True)
    parser.add_argument("--r1-manifest", type=Path, required=True)
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.r1_cohort, args.r1_manifest,
                           args.official_test, args.official_manifest,
                           args.predictions, args.prediction_report, args.output),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
