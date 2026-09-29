"""R3 repair prompts use only the frozen answer-free query view."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autoresearch.prepare_r3_repair_prompts import build, prompt_from_query
from scripts.autoresearch.stratified_manifest import digest


def _query() -> dict:
    return {
        "target_smiles": "CO",
        "prefix_actions": [{
            "name": "import_fragments",
            "arguments": {"fragments": [{"smiles": "O", "count": 1,
                                      "purpose": "electron_participant"}]},
            "result": {"ok": True, "code": "PASS", "current_state": "CO.O"},
        }],
        "corrupted_action": {
            "name": "apply_electron_flow",
            "arguments": {"direction": "retrosynthetic", "electron_flow": [],
                          "bond_order_changes": [], "charge_changes": []},
        },
        "executor_result": {"ok": False, "code": "REJECTED",
                            "current_state": None, "error": "invalid destination"},
    }


def test_repair_prompt_has_pre_action_state_and_public_feedback_only() -> None:
    prompt = prompt_from_query(_query())
    assert "CURRENT STATE SMILES: CO.O" in prompt
    assert "ANNOTATED CURRENT STATE:" in prompt
    assert "accepted_actions: 1" in prompt
    assert "submitted_action:" in prompt
    assert "invalid destination" in prompt
    assert "expected_precursor" not in prompt
    assert "private_reference" not in prompt
    assert "correct_action" not in prompt


def test_repair_prompt_rejects_private_or_malformed_query_fields() -> None:
    with pytest.raises(ValueError, match="extra or missing"):
        prompt_from_query({**_query(), "private_reference": {"correct_action": {}}})
    malformed = _query()
    malformed["prefix_actions"][0]["result"]["ok"] = False
    with pytest.raises(ValueError, match="non-accepted"):
        prompt_from_query(malformed)


def test_repair_prompt_freeze_requires_hash_and_never_enables_evaluation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    queries = source / "r3_queries.jsonl"
    queries.write_text(json.dumps({"artifact_type": "r3_answer_free_query_v1",
                                   "case_id": "a" * 64,
                                   "model_input": _query()}) + "\n")
    (source / "manifest.json").write_text(json.dumps({
        "query_sha256": digest(queries), "cases": 1,
        "model_input_fields": sorted(_query()),
    }))
    (source / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "query_sha256": digest(queries), "repair_inference_allowed": True,
        "localization_evaluation_allowed": False,
    }))
    output = tmp_path / "prompts"
    report = build(queries, output, expected_cases=1)
    status = json.loads((output / "ARTIFACT_STATUS.json").read_text())
    assert report["cases"] == 1
    assert report["prompts_sha256"] == digest(output / "r3_repair_prompts.jsonl")
    assert status["inference_allowed"] is True
    assert status["evaluation_allowed"] is False
    assert status["localization_evaluation_allowed"] is False
    with pytest.raises(FileExistsError):
        build(queries, output, expected_cases=1)
    (source / "manifest.json").write_text(json.dumps({
        "query_sha256": "0" * 64, "cases": 1,
        "model_input_fields": sorted(_query()),
    }))
    with pytest.raises(ValueError, match="provenance"):
        build(queries, tmp_path / "drifted", expected_cases=1)
