"""R5 candidate-conditioned trace-support scoring contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch import stratified_manifest
from scripts.autoresearch.score_r5_external import score
from scripts.autoresearch.stratified_manifest import digest


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    cohort = source / "r5_external_predictions.jsonl"
    rows = []
    for product in ("CC", "CO"):
        candidates = []
        for rank in range(1, 6):
            precursor = "O" if rank == 1 else "C" if rank == 2 else None
            candidates.append({"rank": rank, "canonical_precursors": precursor,
                               "smiles_status": "valid_smiles" if precursor else "missing",
                               "recorded_reference_status": (
                                   "recorded_reference" if rank == 2 else
                                   "not_recorded_not_proven_invalid" if rank == 1 else "unassessable")})
        rows.append({"product_smiles": product, "model_input": {"product_smiles": product},
                     "candidates": candidates})
    _jsonl(cohort, rows)
    (source / "manifest.json").write_text(json.dumps({
        "cohort_sha256": digest(cohort), "target_semantics": "retrosynthetic_precursor_set",
        "products": 2, "ranks_per_product": 5}))
    (source / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "cohort_sha256": digest(cohort), "evaluation_allowed": True}))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"engineering_only": False,
                                  "mech_comparison_identifiable": True,
                                  "evaluation_hashes": {"r5_external_predictions": digest(cohort)}}))
    for condition in ("base", "mech"):
        path = tmp_path / f"{condition}.jsonl"
        verification = []
        for product in ("CC", "CO"):
            for rank in range(1, 6):
                valid = rank <= 2
                attempts = [{"termination_reason": "generation_failed"}] if valid else []
                if condition == "mech" and rank == 2:
                    attempts = [{"termination_reason": "terminal_tool", "final_result": {
                        "ok": True, "formal_execute": True, "trace_bound": True,
                        "endpoint_source": "environment_owned_trace",
                        "structural_precursor": "C"}}]
                verification.append({"product_smiles": product, "rank": rank,
                                     "model_input": ({"product_smiles": product,
                                                      "proposed_precursors": "O" if rank == 1 else "C"}
                                                     if valid else None),
                                     "verification_status": "completed" if valid else "skipped_invalid_or_missing",
                                     "attempts": attempts})
        _jsonl(path, verification)
        path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
            "verification_sha256": digest(path), "cohort_sha256": digest(cohort),
            "condition": condition, "checkpoint_identifier": f"{condition}-ckpt",
            "checkpoint_sha256": ("a" if condition == "base" else "b") * 64,
            "input_fields": ["product_smiles", "proposed_precursors"],
            "verification_semantics": "candidate_conditioned_executor_trace_v1",
            "max_attempts": 2}))
    return cohort, tmp_path / "base.jsonl", tmp_path / "mech.jsonl", freeze


def test_r5_scoring_promotes_only_matched_executable_trace(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(stratified_manifest.EVALUATION_ROW_BOUNDS,
                        "r5_external_predictions", (2, 2))
    cohort, base, mech, freeze = _fixture(tmp_path)
    result = score(cohort, base, mech, freeze, tmp_path / "out", expected_products=2)
    assert result["denominators"] == {"products": 2, "external_candidate_slots": 10}
    assert result["metrics"]["base"]["known_recorded_at_1_after"] == 0
    assert result["metrics"]["mech"]["known_recorded_at_1_after"] == 1
    assert result["metrics"]["mech"]["trace_supported_slots"] == 2
    assert result["metrics"]["unsupported_at_1"] is None
    rows = [json.loads(line) for line in (tmp_path / "out/mech_rows.jsonl").read_text().splitlines()]
    assert all(row["reranked_original_ranks"] == [2, 1, 3, 4, 5] for row in rows)


def test_r5_scoring_rejects_forbidden_source_and_incomplete_slots(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(stratified_manifest.EVALUATION_ROW_BOUNDS,
                        "r5_external_predictions", (2, 2))
    cohort, base, mech, freeze = _fixture(tmp_path)
    status = cohort.parent / "ARTIFACT_STATUS.json"
    status.write_text(json.dumps({"cohort_sha256": digest(cohort), "evaluation_allowed": False}))
    with pytest.raises(ValueError, match="forbidden"):
        score(cohort, base, mech, freeze, tmp_path / "forbidden", expected_products=2)
    status.write_text(json.dumps({"cohort_sha256": digest(cohort), "evaluation_allowed": True}))
    rows = [json.loads(line) for line in mech.read_text().splitlines()]
    _jsonl(mech, rows[:-1])
    sidecar = mech.with_suffix(mech.suffix + ".manifest.json")
    manifest = json.loads(sidecar.read_text())
    manifest["verification_sha256"] = digest(mech)
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every frozen product-rank slot"):
        score(cohort, base, mech, freeze, tmp_path / "incomplete", expected_products=2)


def test_r5_scoring_does_not_match_a_different_executed_endpoint(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(stratified_manifest.EVALUATION_ROW_BOUNDS,
                        "r5_external_predictions", (2, 2))
    cohort, base, mech, freeze = _fixture(tmp_path)
    rows = [json.loads(line) for line in mech.read_text().splitlines()]
    for row in rows:
        if row["rank"] == 2:
            row["attempts"][0]["final_result"]["structural_precursor"] = "O"
    _jsonl(mech, rows)
    sidecar = mech.with_suffix(mech.suffix + ".manifest.json")
    manifest = json.loads(sidecar.read_text())
    manifest["verification_sha256"] = digest(mech)
    sidecar.write_text(json.dumps(manifest))
    result = score(cohort, base, mech, freeze, tmp_path / "wrong_endpoint", expected_products=2)
    assert result["metrics"]["mech"]["executed_trace_attempts"] == 2
    assert result["metrics"]["mech"]["trace_supported_slots"] == 0
    assert result["metrics"]["mech"]["known_recorded_at_1_after"] == 0
