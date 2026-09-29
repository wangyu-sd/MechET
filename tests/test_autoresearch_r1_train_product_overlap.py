import json

import pytest

from scripts.autoresearch.audit_r1_train_product_overlap import audit
from scripts.autoresearch.analyze_r1_overlap_diagnostic import analyze
from scripts.autoresearch.stratified_manifest import digest


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_r1_overlap_audit_tracks_exact_products_and_strata(tmp_path):
    cohort = tmp_path / "r1.jsonl"
    train = tmp_path / "train.jsonl"
    cohort_meta = tmp_path / "r1_manifest.json"
    official_meta = tmp_path / "official_manifest.json"
    output = tmp_path / "result.json"
    write_jsonl(cohort, [
        {"product_smiles": "CO", "strata": {"reference_count": "exactly_two"}},
        {"product_smiles": "CCO", "strata": {"reference_count": "three_or_more"}},
    ])
    write_jsonl(train, [
        {"id": "train:1", "target_smiles": "[CH3:1][OH:2]"},
        {"id": "train:2", "target_smiles": "CCC"},
        {"id": "train:3", "target_smiles": "CO"},
    ])
    cohort_meta.write_text(json.dumps({"cohort_sha256": digest(cohort), "products": 2}))
    official_meta.write_text(json.dumps({"splits": {"train": {
        "output_sha256": digest(train), "rows": 3}}}))

    result = audit(cohort, cohort_meta, train, official_meta, output)
    assert result["r1_products"] == 2
    assert result["overlapping_products"] == 1
    assert result["overlapping_train_ids"] == {"CO": ["train:1", "train:3"]}
    assert result["by_stratum"]["reference_count"] == {
        "exactly_two:total": 1, "exactly_two:overlap": 1,
        "three_or_more:total": 1, "three_or_more:overlap": 0,
    }
    with pytest.raises(FileExistsError):
        audit(cohort, cohort_meta, train, official_meta, output)


def test_r1_overlap_audit_rejects_hash_drift(tmp_path):
    cohort = tmp_path / "r1.jsonl"
    train = tmp_path / "train.jsonl"
    cohort_meta = tmp_path / "r1_manifest.json"
    official_meta = tmp_path / "official_manifest.json"
    write_jsonl(cohort, [{"product_smiles": "CO", "strata": {}}])
    write_jsonl(train, [{"id": "train:1", "target_smiles": "CO"}])
    cohort_meta.write_text(json.dumps({"cohort_sha256": "wrong", "products": 1}))
    official_meta.write_text(json.dumps({"splits": {"train": {
        "output_sha256": digest(train), "rows": 1}}}))
    with pytest.raises(ValueError, match="cohort hash"):
        audit(cohort, cohort_meta, train, official_meta, tmp_path / "result.json")


def test_r1_existing_diagnostic_is_partitioned_without_changing_denominator(tmp_path):
    overlap = tmp_path / "overlap.json"
    rows = tmp_path / "rows.jsonl"
    diagnostic = tmp_path / "diagnostic.json"
    write_jsonl(rows, [
        {"product_smiles": "CO", "metrics": {
            "multi_reference_at_1": True, "single_reference_at_1": False,
            "multi_reference_at_k": True, "single_reference_at_k": False}},
        {"product_smiles": "CCO", "metrics": {
            "multi_reference_at_1": True, "single_reference_at_1": True,
            "multi_reference_at_k": True, "single_reference_at_k": True}},
    ])
    overlap.write_text(json.dumps({
        "artifact_type": "r1_existing_model_train_product_overlap_audit_v1",
        "r1_cohort_sha256": "source", "r1_products": 2,
        "overlapping_train_ids": {"CO": ["train:1"]}}))
    diagnostic.write_text(json.dumps({
        "artifact_type": "r1_existing_direct_diagnostic_v1",
        "r1_cohort_sha256": "source", "products": 2,
        "rows_sha256": digest(rows),
        "single_reference_false_negatives_recovered_at_1": 1,
        "single_reference_false_negatives_recovered_at_k": 1}))
    result = analyze(overlap, diagnostic, rows, tmp_path / "partition.json")
    assert result["by_overlap"]["exact_product_overlap"]["recovered_at_1"] == 1
    assert result["by_overlap"]["exact_product_disjoint"]["products"] == 1
