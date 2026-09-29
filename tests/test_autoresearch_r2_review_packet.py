import csv
import json

import pytest

from scripts.autoresearch.prepare_r2_review_packet import build
from scripts.autoresearch.stratified_manifest import digest


def test_review_packet_is_unlabeled_and_hash_bound(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "review_candidates.jsonl"
    row = {
        "candidate_id": "a" * 64,
        "chemical_negative_label": None,
        "review_status": "pending_independent_chemistry_review",
        "model_input": {"product_smiles": "CCO", "proposed_precursors": "CC.O"},
        "source_trace_reaction_id": "reaction-1",
        "review_context": {
            "known_recorded_precursors": ["CCO"],
            "event_mutation": {"field": "destination", "from": "A01", "to": "A02"},
        },
    }
    source.write_text(json.dumps(row) + "\n")
    (source_dir / "manifest.json").write_text(json.dumps({
        "cohort_sha256": digest(source), "candidate_count": 1}))
    (source_dir / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "training_allowed": False, "evaluation_allowed": False}))
    output = tmp_path / "packet"
    report = build(source, output, expected_candidates=1)
    assert report["cases"] == 1
    assert report["chemical_labels_provided"] is False
    assert report["review_packet_sha256"] == digest(output / "review_packet.html")
    html = (output / "review_packet.html").read_text()
    assert "<svg" in html
    assert "No labels are supplied" in html
    with (output / "review_template.csv").open(newline="") as stream:
        worksheet = list(csv.DictReader(stream))
    assert len(worksheet) == 1
    assert worksheet[0]["candidate_id"] == "a" * 64
    assert worksheet[0]["review_finding"] == ""
    with pytest.raises(FileExistsError):
        build(source, output, expected_candidates=1)


def test_review_packet_rejects_prelabelled_source(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "review_candidates.jsonl"
    source.write_text(json.dumps({"candidate_id": "x", "chemical_negative_label": True,
                                  "review_status": "pending_independent_chemistry_review"}) + "\n")
    (source_dir / "manifest.json").write_text(json.dumps({
        "cohort_sha256": digest(source), "candidate_count": 1}))
    (source_dir / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "training_allowed": False, "evaluation_allowed": False}))
    with pytest.raises(ValueError, match="already labeled"):
        build(source, tmp_path / "packet", expected_candidates=1)
