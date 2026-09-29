"""R1 stratification preserves source hashes and diagnostic claim boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch import stratified_manifest
from scripts.autoresearch.analyze_r1_existing_diagnostics import analyze
from scripts.autoresearch.stratified_manifest import digest


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    cohort = source / "r1_multi_reference.jsonl"
    products = ("CC", "CO", "CN", "CCC")
    _jsonl(cohort, [{"product_smiles": product, "reference_count": 3 if i == 2 else 2,
                     "strata": {"reference_count": "three_or_more" if i == 2 else "exactly_two",
                                "disconnection": "different" if i < 2 else "similar",
                                "structural_overlap": "high" if i % 2 == 0 else "low"}}
                    for i, product in enumerate(products)])
    (source / "manifest.json").write_text(json.dumps({"cohort_sha256": digest(cohort)}))
    (source / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "cohort_sha256": digest(cohort), "evaluation_allowed": True}))
    direct = tmp_path / "direct"
    trace = tmp_path / "trace"
    direct.mkdir()
    trace.mkdir()
    direct_metrics = [
        (False, True, False, True),
        (True, True, True, True),
        (False, False, False, True),
        (False, False, False, False),
    ]
    _jsonl(direct / "r1_existing_direct_rows.jsonl", [
        {"product_smiles": product, "recorded_reference_count": 3 if i == 2 else 2,
         "metrics": {"candidate_count": 10,
                     "single_reference_at_1": values[0], "multi_reference_at_1": values[1],
                     "single_reference_at_k": values[2], "multi_reference_at_k": values[3]}}
        for i, (product, values) in enumerate(zip(products, direct_metrics))])
    _jsonl(trace / "r1_existing_trace_rows.jsonl", [
        {"product_smiles": product, "candidate_count": 10,
         "formal_at_1": i == 0, "formal_at_k": i < 2,
         "recorded_hit_at_1": False, "recorded_hit_at_k": i == 1}
        for i, product in enumerate(products)])
    for kind, directory in (("direct", direct), ("trace", trace)):
        rows = directory / f"r1_existing_{kind}_rows.jsonl"
        (directory / "result.json").write_text(json.dumps({
            "artifact_type": f"r1_existing_{kind}_diagnostic_v1",
            "status": "diagnostic_only_not_scientific_smoke",
            "r1_cohort_sha256": digest(cohort), "rows_sha256": digest(rows),
            "existing_predictions_sha256": kind[0] * 64, "products": 4}))
    return cohort, direct, trace


def test_r1_stratified_diagnostic_keeps_selected_cohort_denominator(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(stratified_manifest.EVALUATION_ROW_BOUNDS,
                        "r1_multi_reference", (4, 4))
    cohort, direct, trace = _fixture(tmp_path)
    result = analyze(cohort, direct, trace, tmp_path / "out", expected_products=4,
                     bootstrap_replicates=100)
    assert result["status"] == "diagnostic_only_not_scientific_smoke"
    assert result["overall"]["counts"]["alternative_recovered_at_1"] == 1
    assert result["overall"]["counts"]["alternative_recovered_at_k"] == 2
    assert result["by_stratum"]["disconnection"]["different"]["products"] == 2
    assert result["by_stratum"]["reference_count"]["three_or_more"]["products"] == 1
    assert result["overall"]["counts"]["trace_formal_at_k"] == 2
    assert result["product_bootstrap_95pct_interval_alternative_recovered_fraction"]
    with pytest.raises(FileExistsError):
        analyze(cohort, direct, trace, tmp_path / "out", expected_products=4,
                bootstrap_replicates=100)


def test_r1_stratified_diagnostic_rejects_drifted_rows(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(stratified_manifest.EVALUATION_ROW_BOUNDS,
                        "r1_multi_reference", (4, 4))
    cohort, direct, trace = _fixture(tmp_path)
    path = direct / "r1_existing_direct_rows.jsonl"
    path.write_text(path.read_text().replace('"candidate_count": 10', '"candidate_count": 9', 1))
    with pytest.raises(ValueError, match="provenance/hash mismatch"):
        analyze(cohort, direct, trace, tmp_path / "drift", expected_products=4,
                bootstrap_replicates=100)
