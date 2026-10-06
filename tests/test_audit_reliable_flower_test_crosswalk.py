import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_reliable_flower_test_crosswalk import compare


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    full_file, strict_file = tmp_path / "full.jsonl", tmp_path / "strict.jsonl"
    full_manifest, strict_manifest = tmp_path / "full_manifest.json", tmp_path / "strict_manifest.json"
    full_sha = _write(full_file, [
        {"source_id": "1", "target_smiles": "[CH4:5]", "structural_precursor": "CO"},
        {"source_id": "2", "target_smiles": "[CH4:5]", "structural_precursor": "CO"},
        {"source_id": "3", "target_smiles": "[CH4:5]", "structural_precursor": "CO"},
    ])
    strict_sha = _write(strict_file, [
        {"source_id": "flower_mech_proof_test_1", "target_smiles": "C",
         "structural_precursor": "[CH3:2][OH:3]"},
        {"source_id": "flower_mech_proof_test_2", "target_smiles": "N",
         "structural_precursor": "CO.O"},
    ])
    full_manifest.write_text(json.dumps({"splits": {"test": {
        "rows": 3, "output_sha256": full_sha,
    }}}), encoding="utf-8")
    strict_manifest.write_text(json.dumps({"splits": {"test": {
        "rows": 2, "sha256": strict_sha,
    }}}), encoding="utf-8")
    return full_file, full_manifest, strict_file, strict_manifest


def test_crosswalk_separates_input_and_endpoint_label_differences(tmp_path: Path):
    full_file, full_manifest, strict_file, strict_manifest = _fixture(tmp_path)
    report = compare(
        full_file=full_file, full_manifest=full_manifest,
        strict_file=strict_file, strict_manifest=strict_manifest,
        expected_full=3, expected_strict=2,
    )
    assert report["full_only_reaction_ids"] == ["3"]
    assert report["counts"]["same_structural_precursor"] == 1
    assert report["counts"]["strict_has_extra_structural_fragments"] == 1
    assert report["counts"]["product_mismatch"] == 1
    assert (report["counts"].get("same_product_only_model_input", 0)
            + report["counts"].get("chemically_equal_but_different_model_input", 0)) == 1
    assert report["examples"]["strict_has_extra_structural_fragments"][0]["strict_only"] == ["O"]


def test_crosswalk_refuses_changed_test_bytes(tmp_path: Path):
    full_file, full_manifest, strict_file, strict_manifest = _fixture(tmp_path)
    strict_file.write_text(strict_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test bytes disagree"):
        compare(
            full_file=full_file, full_manifest=full_manifest,
            strict_file=strict_file, strict_manifest=strict_manifest,
            expected_full=3, expected_strict=2,
        )
