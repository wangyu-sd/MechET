"""Frozen unmarked localization prompts and full-denominator predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.agent_model_init import path_sha256
from scripts.autoresearch.prepare_r3_localization_prompts import (
    PROMPT_VERSION, build, prompt_from_query,
)
from scripts.autoresearch.run_r3_localization_inference import (
    load_inputs, prediction_row,
)
from scripts.autoresearch.stratified_manifest import digest


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    queries_dir = tmp_path / "queries"
    queries_dir.mkdir()
    queries = queries_dir / "r3_unmarked_queries.jsonl"
    queries.write_text(json.dumps({
        "artifact_type": "r3_unmarked_localization_query_v1",
        "case_id": "a" * 64,
        "model_input": {
            "target_smiles": "CCO",
            "candidate_actions": [
                {"name": "import_fragments", "arguments": {"fragments": []}},
                {"name": "apply_electron_flow", "arguments": {
                    "direction": "retrosynthetic", "electron_flow": []}},
            ],
        },
    }) + "\n")
    (queries_dir / "manifest.json").write_text(json.dumps({
        "query_sha256": digest(queries), "cases": 1,
        "model_input_fields": ["target_smiles", "candidate_actions"],
    }))
    (queries_dir / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "query_sha256": digest(queries), "localization_inference_allowed": True,
        "training_allowed": False, "evaluation_allowed": False,
    }))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic")
    (adapter / "adapter_manifest.json").write_text(json.dumps({
        "adapter_sha256": path_sha256(adapter),
        "base_model": "Qwen/Qwen3-8B",
        "base_model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "condition_name": "flower_natural_language_event_compact_history_v2_qwen3_8b",
    }))
    return queries, adapter, queries_dir


def test_r3_localization_prompt_frozen_public_only(tmp_path: Path) -> None:
    queries, adapter, _ = _fixture(tmp_path)
    output = tmp_path / "prompts"
    report = build(queries, output, expected_cases=1)
    prompts = output / "r3_localization_prompts.jsonl"
    row = json.loads(prompts.read_text())
    assert report["prompts_sha256"] == digest(prompts)
    assert row["prompt_version"] == PROMPT_VERSION
    assert row["candidate_action_count"] == 2
    assert "0:" in row["user_prompt"] and "1:" in row["user_prompt"]
    assert "private_reference" not in row["user_prompt"]
    loaded, provenance = load_inputs(prompts, queries, adapter, expected_cases=1)
    assert loaded == [row]
    assert provenance["query_sha256"] == digest(queries)
    with pytest.raises(FileExistsError):
        build(queries, output, expected_cases=1)
    queries.write_text(queries.read_text().replace("CCO", "CCN"))
    with pytest.raises(ValueError, match="contract drifted"):
        load_inputs(prompts, queries, adapter, expected_cases=1)


def test_r3_localization_prompt_rejects_private_action_fields() -> None:
    with pytest.raises(ValueError, match="leaks reference"):
        prompt_from_query({
            "target_smiles": "CCO",
            "candidate_actions": [
                {"name": "import_fragments", "arguments": {"private_reference": 1}},
                {"name": "finish_trace", "arguments": {}},
            ],
        })


@pytest.mark.parametrize("completion,expected", [
    ('{"first_error_index":1}', 1),
    ('{"first_error_index":0}', 0),
    ('{"first_error_index":2}', None),
    ('{"first_error_index":true}', None),
    ('{"first_error_index":"1"}', None),
    ('{"first_error_index":1,"reason":"guess"}', None),
    ('Some reasoning {"first_error_index":1}', None),
])
def test_r3_localization_prediction_preserves_failed_slot(completion: str,
                                                           expected: int | None) -> None:
    prediction = prediction_row("a" * 64, completion, 2)
    assert prediction == {
        "case_id": "a" * 64,
        "generation_status": "completed" if expected is not None else "failed",
        "predicted_failure_index": expected,
    }
