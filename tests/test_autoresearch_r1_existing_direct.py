"""Existing direct predictions can be rescored against held-out alternatives."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.autoresearch.score_r1_existing_direct import build, score_one
from scripts.autoresearch.stratified_manifest import digest, product_key


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_alternative_recovers_single_reference_false_negative() -> None:
    outcome = score_one(product_key("C"), {product_key("C"), product_key("O")},
                        ["<answer>O</answer>", "<answer>not a smiles</answer>"])
    assert outcome["single_reference_at_1"] is False
    assert outcome["multi_reference_at_1"] is True
    assert outcome["single_reference_at_k"] is False
    assert outcome["multi_reference_at_k"] is True
    assert outcome["parseable_candidates"] == 1


def test_r1_existing_direct_diagnostic_preserves_prediction_provenance(tmp_path: Path) -> None:
    source = tmp_path / "official_test.jsonl"
    _jsonl(source, [
        {"id": "test:1", "target_smiles": "CC", "structural_precursor": "C"},
        {"id": "test:2", "target_smiles": "CC", "structural_precursor": "O"},
    ])
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"splits": {"test": {"output_sha256": digest(source)}}}))
    r1 = tmp_path / "r1.jsonl"
    _jsonl(r1, [{"product_smiles": product_key("CC"), "references": [
        {"precursor_smiles": product_key("C")},
        {"precursor_smiles": product_key("O")},
    ]}])
    r1_manifest = tmp_path / "r1_manifest.json"
    r1_manifest.write_text(json.dumps({"cohort_sha256": digest(r1)}))
    predictions = tmp_path / "predictions.jsonl"
    _jsonl(predictions, [
        {"id": "test:1", "target_smiles": "CC", "prediction_status": "completed",
         "candidates": [{"sample_index": 0, "prediction": "<answer>O</answer>"},
                        {"sample_index": 1, "prediction": "<answer>N</answer>"}]},
        {"id": "test:2", "target_smiles": "CC", "prediction_status": "completed",
         "candidates": [{"sample_index": 0, "prediction": "<answer>C</answer>"},
                        {"sample_index": 1, "prediction": "<answer>N</answer>"}]},
    ])
    prediction_report = tmp_path / "evaluation.json"
    prediction_report.write_text(json.dumps({"predictions_sha256": digest(predictions),
                                             "n_reference_rows": 2}))
    result = build(r1, r1_manifest, source, official, predictions,
                   prediction_report, tmp_path / "out", expected_reactions=2)
    assert result["status"] == "diagnostic_only_not_scientific_smoke"
    assert result["products"] == 1
    assert result["single_reference_false_negatives_recovered_at_1"] == 1
    assert result["totals"]["single_reference_at_1"] == 0
    assert result["totals"]["multi_reference_at_1"] == 1
