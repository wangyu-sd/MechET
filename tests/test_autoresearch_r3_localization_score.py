"""Unmarked R3 localization never uses the exposed-failure repair query."""

import hashlib
import json

import pytest

from scripts.autoresearch import score_r3_localization as module
from scripts.autoresearch.stratified_manifest import digest


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _fixture(tmp_path, *, index, status="completed"):
    source_dir = tmp_path / "source"
    query_dir = tmp_path / "query"
    source_dir.mkdir()
    query_dir.mkdir()
    imported = {"name": "import_fragments", "arguments": {"fragments": []}}
    early = {"name": "apply_electron_flow", "arguments": {"direction": "retrosynthetic",
                                                      "electron_flow": []}}
    changed = {"name": "apply_electron_flow", "arguments": {"direction": "retrosynthetic",
                                                        "electron_flow": [{"source": "A01"}]}}
    finish = {"name": "finish_trace", "arguments": {}}
    source_row = {
        "source_split": "test",
        "model_visible": {"target_smiles": "CC", "prefix_actions": [imported, early],
                          "corrupted_action": changed,
                          "executor_result": {"ok": True, "code": "PASS"}},
        "private_reference": {"first_failure_index": 2, "suffix_actions": [finish]},
        "strata": {"failure_depth": "middle", "event_coordination": "two"},
    }
    source = source_dir / "r3_corruptions.jsonl"
    _write(source, source_row)
    case_id = hashlib.sha256(source.read_text().rstrip("\n").encode()).hexdigest()
    _write(source_dir / "manifest.json", {"cohort_sha256": digest(source)})
    query = query_dir / "r3_unmarked_queries.jsonl"
    _write(query, {"case_id": case_id,
                   "model_input": {"target_smiles": "CC",
                                   "candidate_actions": [imported, early, changed, finish]}})
    _write(query_dir / "manifest.json", {
        "source_sha256": digest(source), "query_sha256": digest(query),
        "cases": 1, "model_input_fields": ["target_smiles", "candidate_actions"]})
    _write(query_dir / "ARTIFACT_STATUS.json", {
        "query_sha256": digest(query), "localization_inference_allowed": True,
        "evaluation_allowed": False})
    predictions = tmp_path / "predictions.jsonl"
    _write(predictions, {"case_id": case_id, "generation_status": status,
                         "predicted_failure_index": index})
    _write(predictions.with_suffix(".jsonl.manifest.json"), {
        "predictions_sha256": digest(predictions), "query_sha256": digest(query),
        "input_fields": ["model_input"],
        "prediction_semantics": "zero_based_first_reference_divergence_index_v1",
        "checkpoint_identifier": "unit-test", "checkpoint_sha256": "a" * 64})
    return source, query, predictions


def test_r3_unmarked_localization_scores_model_and_predeclared_shortcut(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_evaluation_source", lambda source, name: digest(source))
    source, query, predictions = _fixture(tmp_path, index=2)
    report = module.score(source, query, predictions, tmp_path / "result", expected_cases=1)
    assert report["top1_exact"] == 1
    assert report["shortcut_baseline"]["top1_exact"] == 0
    assert report["mean_absolute_distance_all_missing_penalized"] == 0


def test_r3_unmarked_localization_missing_generation_stays_in_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_evaluation_source", lambda source, name: digest(source))
    source, query, predictions = _fixture(tmp_path, index=None, status="failed")
    report = module.score(source, query, predictions, tmp_path / "result", expected_cases=1)
    assert report["top1_exact"] == 0
    assert report["coverage"] == 0
    assert report["mean_absolute_distance_all_missing_penalized"] == 4


def test_r3_unmarked_localization_rejects_out_of_range_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_evaluation_source", lambda source, name: digest(source))
    source, query, predictions = _fixture(tmp_path, index=4)
    with pytest.raises(ValueError, match="invalid localization prediction"):
        module.score(source, query, predictions, tmp_path / "result", expected_cases=1)


def test_r3_unmarked_localization_rejects_rehashed_query_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_evaluation_source", lambda source, name: digest(source))
    source, query, predictions = _fixture(tmp_path, index=2)
    leaked = json.loads(query.read_text())
    leaked["model_input"]["first_failure_index"] = 2
    _write(query, leaked)
    query_manifest = query.parent / "manifest.json"
    query_status = query.parent / "ARTIFACT_STATUS.json"
    prediction_manifest = predictions.with_suffix(".jsonl.manifest.json")
    for path in (query_manifest, query_status, prediction_manifest):
        value = json.loads(path.read_text())
        value["query_sha256"] = digest(query)
        _write(path, value)
    with pytest.raises(ValueError, match="differs from its frozen private source"):
        module.score(source, query, predictions, tmp_path / "result", expected_cases=1)
