"""Archived external G2S proposals can feed R5 without reference selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.freeze_r5_external_predictions import freeze
from scripts.autoresearch.import_r5_orbit_g2s import convert
from scripts.autoresearch.stratified_manifest import digest


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    query = tmp_path / "query.jsonl"
    query.write_text("".join(json.dumps({
        "product_smiles": product,
        "model_input": {"product_smiles": product},
        "source_dataset": "FlowER", "source_split": "test", "strata": {},
        "private_reference": {"recorded_precursor_sets": [
            {"precursor_smiles": "C", "support_record_ids": ["r"]}]},
    }) + "\n" for product in ("CC", "CO")))
    query_manifest = tmp_path / "query_manifest.json"
    query_manifest.write_text(json.dumps({"cohort_sha256": digest(query), "products": 2}))
    source = tmp_path / "g2s.json"
    source.write_text(json.dumps([
        {"product_smiles": "CC", "predicted_precursors": ["O", "C"]},
        {"product_smiles": "C-C", "predicted_precursors": ["N"]},
        {"product_smiles": "CO", "predicted_precursors": ["C", "O"]},
    ]))
    provenance = tmp_path / "PROVENANCE.md"
    provenance.write_text("# Graph2SMILES collaborator archive\n")
    return query, query_manifest, source, provenance


def test_import_preserves_top5_and_first_archived_duplicate(tmp_path: Path) -> None:
    query, manifest, source, provenance = _fixture(tmp_path)
    output = tmp_path / "converted"
    report = convert(query, manifest, source, provenance, output,
                     expected_source_sha256=digest(source),
                     expected_provenance_sha256=digest(provenance),
                     expected_products=2)
    assert report["products"] == 2
    assert report["duplicate_query_products"] == 1
    assert report["conflicting_top5_products"] == 1
    rows = [json.loads(line) for line in (output / "predictions.jsonl").open()]
    assert rows[0]["source_row_index"] == 0
    assert rows[0]["candidates"][0] == {"rank": 1, "precursors": "O"}
    assert rows[0]["candidates"][1] == {"rank": 2, "precursors": "C"}
    assert rows[0]["canonical_product_match_count"] == 2
    meta = json.loads((output / "provenance.json").read_text())
    assert meta["training_overlap_audited"] is False
    assert meta["target_semantics"] == "full_reaction_world"
    assert "unavailable" in meta["checkpoint_identifier"]
    with pytest.raises(ValueError, match="precursor-set predictions"):
        freeze(query, manifest, output / "predictions.jsonl",
               output / "provenance.json", tmp_path / "frozen",
               expected_products=2)


def test_import_fails_closed_on_hash_or_coverage_drift(tmp_path: Path) -> None:
    query, manifest, source, provenance = _fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 changed"):
        convert(query, manifest, source, provenance, tmp_path / "bad_hash",
                expected_source_sha256="0" * 64,
                expected_provenance_sha256=digest(provenance),
                expected_products=2)
    source.write_text(json.dumps([{"product_smiles": "CC",
                                   "predicted_precursors": ["C"]}]))
    with pytest.raises(ValueError, match="miss 1 R5 products"):
        convert(query, manifest, source, provenance, tmp_path / "bad_coverage",
                expected_source_sha256=digest(source),
                expected_provenance_sha256=digest(provenance),
                expected_products=2)
