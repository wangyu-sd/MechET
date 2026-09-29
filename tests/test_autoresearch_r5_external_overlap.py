"""External-model train overlap is measured on its actual product inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.audit_r5_external_train_overlap import audit
from scripts.autoresearch.stratified_manifest import digest


def test_r5_external_overlap_preserves_full_denominator(tmp_path: Path) -> None:
    query = tmp_path / "query.jsonl"
    query.write_text(''.join(json.dumps({"product_smiles": item}) + '\n'
                             for item in ("CC", "CO")))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cohort_sha256": digest(query), "products": 2}))
    train = tmp_path / "train.txt"
    train.write_text("[CH3:1][CH3:2]>>O\nN>>C\n")
    output = tmp_path / "audit.json"
    report = audit(query, manifest, train, output,
                   expected_train_sha256=digest(train), expected_train_rows=2)
    assert report["external_train_rows"] == 2
    assert report["overlapping_query_products"] == 1
    assert report["overlapping_products_and_train_line_numbers"] == {"CC": [1]}
    assert report["leakage_clean_for_exact_product_overlap"] is False
    with pytest.raises(FileExistsError):
        audit(query, manifest, train, output,
              expected_train_sha256=digest(train), expected_train_rows=2)


def test_r5_external_overlap_rejects_source_drift(tmp_path: Path) -> None:
    query = tmp_path / "query.jsonl"
    query.write_text(json.dumps({"product_smiles": "CC"}) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cohort_sha256": digest(query), "products": 1}))
    train = tmp_path / "train.txt"
    train.write_text("CC>>C\n")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        audit(query, manifest, train, tmp_path / "audit.json",
              expected_train_sha256="0" * 64, expected_train_rows=1)
