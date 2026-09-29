"""Only an executor-owned terminal precursor supports a recorded R1 route."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.autoresearch.score_r1_existing_trace import build, candidate_precursor
from scripts.autoresearch.stratified_manifest import digest, product_key


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _candidate(index: int, precursor: str | None, *, bound: bool = True) -> dict:
    final = ({} if precursor is None else {
        "ok": True, "formal_execute": True, "trace_bound": bound,
        "endpoint_source": "environment_owned_trace",
        "structural_precursor": precursor,
    })
    return {"sample_index": index,
            "termination_reason": "terminal_tool" if precursor else "no_tool_call",
            "rollout_state": {"final_result": final,
                              "expected_precursor": "O"}}


def test_trace_score_ignores_gold_and_unbound_outputs() -> None:
    assert candidate_precursor(_candidate(0, None)) is None
    assert candidate_precursor(_candidate(0, "O", bound=False)) is None
    assert candidate_precursor(_candidate(0, "O")) == product_key("O")


def test_existing_trace_diagnostic_keeps_missing_products_in_denominator(tmp_path: Path) -> None:
    r1 = tmp_path / "r1.jsonl"
    _jsonl(r1, [
        {"product_smiles": "CC", "references": [
            {"precursor_smiles": "C"}, {"precursor_smiles": "O"}]},
        {"product_smiles": "CO", "references": [
            {"precursor_smiles": "C"}, {"precursor_smiles": "N"}]},
    ])
    r1_manifest = tmp_path / "r1_manifest.json"
    r1_manifest.write_text(json.dumps({"cohort_sha256": digest(r1)}))
    predictions = tmp_path / "predictions.jsonl"
    _jsonl(predictions, [{
        "id": "strict:1", "target_smiles": "CC", "prediction_status": "completed",
        "candidates": [_candidate(0, "N"), _candidate(1, "O")],
    }])
    prediction_report = tmp_path / "evaluation.json"
    prediction_report.write_text(json.dumps({
        "predictions_sha256": digest(predictions), "n_reference_rows": 1}))
    result = build(r1, r1_manifest, predictions, prediction_report,
                   tmp_path / "out", expected_reference_rows=1)
    assert result["products"] == 2
    assert result["products_with_prediction_row"] == 1
    assert result["totals"]["formal_at_1"] == 1
    assert result["totals"]["recorded_hit_at_1"] == 0
    assert result["totals"]["recorded_hit_at_k"] == 1
    assert result["nonreference_formal_candidates"] == 1
