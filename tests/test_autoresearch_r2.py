"""R2 freezes supported positives without treating GT divergence as falsity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.build_r2_positives import build
from scripts.autoresearch.build_r2_missing_fragment_negatives import (
    build as build_missing_fragment, omission_candidates,
)
from scripts.autoresearch.stratified_manifest import digest, product_key


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _source(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    multi = ["CCO", "CCN", "CCC", "CCOC"]
    single = ["CO", "CN", "CC", "CCCl"]
    source_rows = []
    r1_rows = []
    for index, product in enumerate(multi):
        references = []
        for suffix, precursor in enumerate(("C", "O")):
            record_id = f"m{index}:{suffix}"
            source_rows.append({"id": record_id, "source_id": record_id,
                                "target_smiles": product,
                                "structural_precursor": precursor})
            references.append({"precursor_smiles": product_key(precursor),
                               "support_record_ids": [record_id]})
        r1_rows.append({"product_smiles": product_key(product),
                        "references": references})
    for index, product in enumerate(single):
        source_rows.append({"id": f"s{index}", "source_id": f"s{index}",
                            "target_smiles": product, "structural_precursor": "C"})
    source = tmp_path / "official_test.jsonl"
    _jsonl(source, source_rows)
    official = tmp_path / "official_manifest.json"
    official.write_text(json.dumps({"splits": {"test": {"output_sha256": digest(source)}}}))
    r1 = tmp_path / "r1.jsonl"
    _jsonl(r1, r1_rows)
    r1_manifest = tmp_path / "r1_manifest.json"
    r1_manifest.write_text(json.dumps({"cohort_sha256": digest(r1)}))
    return source, official, r1, r1_manifest


def test_r2_positives_are_balanced_and_not_evaluation_ready(tmp_path: Path) -> None:
    source, official, r1, r1_manifest = _source(tmp_path)
    output = tmp_path / "r2"
    report = build(source, official, r1, r1_manifest, output, per_stratum=4)
    rows = [json.loads(line) for line in (output / "r2_positives.jsonl").open()]
    assert report["positive_proposals"] == 8
    assert report["documented_alternative"] == report["recorded_precursor"] == 4
    assert report["negative_proposals"] == 0
    assert report["quartiles"] == {f"Q{index}": 2 for index in range(1, 5)}
    assert len({row["product_smiles"] for row in rows}) == 8
    assert all(row["private_label"]["known_recorded_positive"] for row in rows)
    assert all("private_label" not in row["model_input"] for row in rows)
    assert all(row["model_input"]["product_smiles"] == row["product_smiles"] for row in rows)
    assert all(row["model_input"]["proposed_precursors"] for row in rows)
    status = json.loads((output / "ARTIFACT_STATUS.json").read_text())
    assert status["positive_source_allowed"] is True
    assert status["evaluation_allowed"] is False
    assert status["training_allowed"] is False
    with pytest.raises(FileExistsError):
        build(source, official, r1, r1_manifest, output, per_stratum=4)


def test_r2_refuses_drifted_independent_source(tmp_path: Path) -> None:
    source, official, r1, r1_manifest = _source(tmp_path)
    r1.write_text(r1.read_text().replace('"O"', '"N"'))
    with pytest.raises(ValueError, match="hash drifted"):
        build(source, official, r1, r1_manifest, tmp_path / "bad", per_stratum=4)


def test_r2_missing_fragment_has_atom_conservation_witness(tmp_path: Path) -> None:
    examples = [
        ("[CH3:1][OH:2]", "[CH3:1][Br:3].[OH-:2]"),
        ("[CH3:1][CH2:2][OH:3]", "[CH3:1][CH2:2][Br:4].[OH-:3]"),
        ("[CH3:1][CH2:2][CH2:3][OH:4]", "[CH3:1][CH2:2][CH2:3][Br:5].[OH-:4]"),
        ("[CH3:1][CH2:2][CH2:3][CH2:4][OH:5]",
         "[CH3:1][CH2:2][CH2:3][CH2:4][Br:6].[OH-:5]"),
    ]
    assert any(2 in item["missing_product_atom_maps"]
               for item in omission_candidates(*examples[0]))
    # Removing the mapped CH3:1 would leave a *different* unmapped methyl
    # group; map-label absence alone must not manufacture a negative.
    remappings = omission_candidates(
        "[CH3:1][O:2][CH3:3]", "[CH3:1][Br:4].[O:2]([CH3:3])[CH3:5]")
    assert all(item["missing_element_counts"] for item in remappings)
    assert all("[CH3:1][Br:4]" not in item["removed_fragment_mapped"]
               for item in remappings)
    source_rows = []
    positives = []
    for index, (product, precursor) in enumerate(examples):
        source_rows.append({"id": f"r{index}", "source_id": f"r{index}",
                            "target_smiles": product, "structural_precursor": precursor})
        positives.append({
            "product_smiles": product_key(product),
            "model_input": {"product_smiles": product_key(product),
                            "proposed_precursors": product_key(precursor)},
            "private_label": {"proposal_record_ids": [f"r{index}"]},
            "strata": {"positive_type": "recorded_precursor" if index < 2
                       else "documented_alternative", "heavy_atom_quartile": "Q1"},
        })
    source = tmp_path / "source.jsonl"
    _jsonl(source, source_rows)
    official = tmp_path / "official.json"
    official.write_text(json.dumps({"splits": {"test": {"output_sha256": digest(source)}}}))
    positive_path = tmp_path / "positives.jsonl"
    _jsonl(positive_path, positives)
    positive_manifest = tmp_path / "positives_manifest.json"
    positive_manifest.write_text(json.dumps({
        "cohort_sha256": digest(positive_path), "positive_proposals": 4}))
    output = tmp_path / "negatives"
    report = build_missing_fragment(source, official, positive_path,
                                    positive_manifest, output, count=4)
    rows = [json.loads(line) for line in
            (output / "r2_missing_fragment_negatives.jsonl").open()]
    assert report["negative_proposals"] == 4
    assert all(row["private_label"]["missing_product_atom_maps"] for row in rows)
    assert all(row["private_label"]["missing_element_counts"] for row in rows)
    assert all(row["private_label"]["scope"] ==
               "closed_stated_precursor_inventory_atom_conservation" for row in rows)
    assert all("private_label" not in row["model_input"] for row in rows)
    assert json.loads((output / "ARTIFACT_STATUS.json").read_text())["evaluation_allowed"] is False
