"""R3 model repair is scored separately from private oracle replay."""

import hashlib
import json

import pytest

from scripts.autoresearch import score_r3_repair as module
from scripts.autoresearch.stratified_manifest import digest


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _electron_action():
    return {"name": "apply_electron_flow", "arguments": {
        "direction": "retrosynthetic",
        "electron_flow": [{"source": "the bond between atoms A01 and A02",
                           "destination": "atom A01",
                           "instruction": "Move 1: transfer the electron pair from the bond between atoms A01 and A02 to atom A01."}],
        "bond_order_changes": [], "charge_changes": []}}


def _fixture(tmp_path, *, predicted_action):
    source_dir = tmp_path / "source"
    query_dir = tmp_path / "query"
    source_dir.mkdir()
    query_dir.mkdir()
    trace = tmp_path / "trace.jsonl"
    _write_json(trace, {"source_id": "r1", "target_smiles": "[CH3:1][CH3:2]"})
    public = {"target_smiles": "CC", "prefix_actions": [],
              "corrupted_action": {"name": "apply_electron_flow", "arguments": {}},
              "executor_result": {"ok": False, "code": "REJECTED"}}
    row = {"source_split": "test", "reaction_id": "r1", "model_visible": public,
           "strata": {"failure_depth": "early", "event_coordination": "one"},
           "private_reference": {"first_failure_index": 0,
                                 "correct_action": _electron_action(),
                                 "suffix_actions": [{"name": "finish_trace", "arguments": {}}],
                                 "expected_successor": "[CH3+].[CH3-]",
                                 "expected_precursor": "[CH3+].[CH3-]"}}
    source = source_dir / "r3_corruptions.jsonl"
    _write_json(source, row)
    source_line = source.read_text().rstrip("\n")
    case_id = hashlib.sha256(source_line.encode()).hexdigest()
    _write_json(source_dir / "manifest.json", {
        "cohort_sha256": digest(source), "trace_source": str(trace.resolve()),
        "trace_sha256": digest(trace)})
    query = query_dir / "r3_queries.jsonl"
    _write_json(query, {"case_id": case_id, "model_input": public})
    _write_json(query_dir / "manifest.json", {
        "source_sha256": digest(source), "query_sha256": digest(query), "cases": 1,
        "model_input_fields": sorted(public)})
    _write_json(query_dir / "ARTIFACT_STATUS.json", {
        "query_sha256": digest(query), "repair_inference_allowed": True,
        "localization_evaluation_allowed": False})
    predictions = tmp_path / "predictions.jsonl"
    _write_json(predictions, {"case_id": case_id,
                              "generation_status": "completed" if predicted_action else "failed",
                              "repair_action": predicted_action})
    _write_json(predictions.with_suffix(".jsonl.manifest.json"), {
        "predictions_sha256": digest(predictions), "query_sha256": digest(query),
        "input_fields": ["model_input"],
        "prediction_semantics": "one_replacement_action_at_exposed_failure_v1",
        "checkpoint_identifier": "unit-test", "checkpoint_sha256": "a" * 64})
    return source, query, trace, predictions


def test_r3_model_repair_score_keeps_full_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_evaluation_source", lambda source, name: digest(source))
    source, query, trace, predictions = _fixture(
        tmp_path, predicted_action=_electron_action())
    report = module.score(source, query, trace, predictions, tmp_path / "score",
                          expected_cases=1)
    assert report["cases"] == 1
    assert report["endpoint_exact"] == 1
    assert report["first_failure_localization_top1"] is None


def test_r3_missing_prediction_counts_as_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_evaluation_source", lambda source, name: digest(source))
    source, query, trace, predictions = _fixture(tmp_path, predicted_action=None)
    report = module.score(source, query, trace, predictions, tmp_path / "score",
                          expected_cases=1)
    assert report["cases"] == 1
    assert report["endpoint_exact"] == 0
    assert report["repair_action_accepted"] == 0


def test_r3_repair_cannot_replace_electron_event_with_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_evaluation_source", lambda source, name: digest(source))
    source, query, trace, predictions = _fixture(
        tmp_path, predicted_action={"name": "finish_trace", "arguments": {}})
    report = module.score(source, query, trace, predictions, tmp_path / "score",
                          expected_cases=1)
    assert report["endpoint_exact"] == 0
    assert report["repair_action_accepted"] == 0


def test_r3_score_rejects_prediction_query_hash_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_evaluation_source", lambda source, name: digest(source))
    source, query, trace, predictions = _fixture(
        tmp_path, predicted_action=_electron_action())
    sidecar = predictions.with_suffix(".jsonl.manifest.json")
    meta = json.loads(sidecar.read_text())
    meta["query_sha256"] = "0" * 64
    _write_json(sidecar, meta)
    with pytest.raises(ValueError, match="provenance/hash mismatch"):
        module.score(source, query, trace, predictions, tmp_path / "score",
                     expected_cases=1)
