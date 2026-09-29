"""R5 external-source diagnostics keep failures and overlap strata visible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.analyze_r5_external_source import analyze
from scripts.autoresearch.stratified_manifest import digest


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "frozen"
    root.mkdir()
    cohort = root / "r5_external_predictions.jsonl"
    rows = []
    for product, overlap, status, first in (
            ("CC", True, "completed", "recorded_reference"),
            ("CO", False, "failed", "unassessable")):
        candidates = []
        for rank in range(1, 6):
            valid = status == "completed" and rank <= 2
            candidates.append({
                "rank": rank,
                "smiles_status": "valid_smiles" if valid else "missing",
                "recorded_reference_status": (
                    first if rank == 1 else "not_recorded_not_proven_invalid"
                    if valid else "unassessable"),
            })
        rows.append({"product_smiles": product, "model_input": {"product_smiles": product},
                     "inference_status": status,
                     "external_training_exact_product_overlap": overlap,
                     "strata": {"reference_support_cohort": "multi_recorded",
                                "heavy_atom_quartile": "Q1"},
                     "candidates": candidates})
    cohort.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "manifest.json").write_text(json.dumps({
        "cohort_sha256": digest(cohort), "products": 2, "ranks_per_product": 5,
        "target_semantics": "retrosynthetic_precursor_set", "model_name": "toy",
        "checkpoint_identifier": "toy-checkpoint", "ranking_semantics": "frequency",
        "training_exact_product_overlap_count": 1,
    }))
    (root / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "cohort_sha256": digest(cohort), "evaluation_allowed": True,
        "training_allowed": False, "headline_allowed": False,
    }))
    return cohort


def test_r5_diagnostic_preserves_failed_products_and_overlap(tmp_path: Path) -> None:
    cohort = _source(tmp_path)
    report = analyze(cohort, tmp_path / "report", expected_products=2)
    assert report["overall"]["products"] == 2
    assert report["overall"]["candidate_slots"] == 10
    assert report["overall"]["completed_products"] == 1
    assert report["overall"]["failed_products"] == 1
    assert report["overall"]["valid_candidate_slots"] == 2
    assert report["overall"]["recorded_at_1"] == 1
    assert report["overall"]["recorded_at_5"] == 1
    assert report["strata"]["training_exact_product_overlap"]["products"] == 1
    assert report["strata"]["training_exact_product_disjoint"]["products"] == 1
    assert report["mechet_verification_performed"] is False
    assert report["headline_allowed"] is False
    with pytest.raises(FileExistsError):
        analyze(cohort, tmp_path / "report", expected_products=2)


def test_r5_diagnostic_rejects_manifest_and_slot_drift(tmp_path: Path) -> None:
    cohort = _source(tmp_path)
    manifest_path = cohort.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cohort_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash-checked"):
        analyze(cohort, tmp_path / "wrong_hash", expected_products=2)
    manifest["cohort_sha256"] = digest(cohort)
    manifest_path.write_text(json.dumps(manifest))
    rows = [json.loads(line) for line in cohort.read_text().splitlines()]
    rows[0]["candidates"][0]["rank"] = 2
    cohort.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["cohort_sha256"] = digest(cohort)
    manifest_path.write_text(json.dumps(manifest))
    status_path = cohort.parent / "ARTIFACT_STATUS.json"
    status = json.loads(status_path.read_text())
    status["cohort_sha256"] = digest(cohort)
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="Top-5 slots"):
        analyze(cohort, tmp_path / "wrong_slots", expected_products=2)
