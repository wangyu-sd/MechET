"""R3 inference lineage and full-denominator prediction formatting."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.agent_model_init import path_sha256
from scripts.autoresearch.prepare_r3_repair_prompts import PROMPT_VERSION
from scripts.autoresearch.run_r3_repair_inference import (
    load_inputs, prediction_row, verify_local_base,
)
from scripts.autoresearch.stratified_manifest import digest
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    queries_dir = tmp_path / "queries"
    queries_dir.mkdir()
    queries = queries_dir / "r3_queries.jsonl"
    queries.write_text(json.dumps({"case_id": "a" * 64, "model_input": {}}) + "\n")
    (queries_dir / "manifest.json").write_text(json.dumps({
        "query_sha256": digest(queries), "cases": 1,
    }))
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompts = prompts_dir / "r3_repair_prompts.jsonl"
    prompts.write_text(json.dumps({"case_id": "a" * 64,
                                   "prompt_version": PROMPT_VERSION,
                                   "user_prompt": "public current state"}) + "\n")
    (prompts_dir / "manifest.json").write_text(json.dumps({
        "prompts_sha256": digest(prompts),
        "queries_sha256": digest(queries), "cases": 1,
        "queries_manifest_sha256": digest(queries_dir / "manifest.json"),
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "tools_sha256": hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "prompt_version": PROMPT_VERSION,
    }))
    (prompts_dir / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "prompts_sha256": digest(prompts), "inference_allowed": True,
        "evaluation_allowed": False,
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
    return prompts, queries, adapter


def test_r3_inference_loads_only_hash_bound_public_inputs(tmp_path: Path) -> None:
    prompts, queries, adapter = _fixture(tmp_path)
    rows, provenance = load_inputs(prompts, queries, adapter, expected_cases=1)
    assert rows[0]["user_prompt"] == "public current state"
    assert provenance["query_sha256"] == digest(queries)
    assert provenance["checkpoint_sha256"] == path_sha256(adapter)
    queries.write_text(queries.read_text().replace("a" * 64, "b" * 64))
    with pytest.raises(ValueError, match="contract drifted"):
        load_inputs(prompts, queries, adapter, expected_cases=1)


def test_r3_inference_retains_failed_candidate_slot() -> None:
    completed = prediction_row("a" * 64, "apply_electron_flow",
                               {"direction": "retrosynthetic"}, "")
    failed = prediction_row("a" * 64, "import_fragments", {}, "")
    assert set(completed) == set(failed) == {
        "case_id", "generation_status", "repair_action"}
    assert completed["generation_status"] == "completed"
    assert failed == {"case_id": "a" * 64,
                      "generation_status": "failed", "repair_action": None}


def test_r3_offline_base_checks_revision_and_weight_hash(tmp_path: Path) -> None:
    revision = "b" * 40
    weight = b"pinned synthetic weight"
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(weight)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer": "model-00001-of-00001.safetensors"}
    }))
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json", "model-00001-of-00001.safetensors"):
        etag = (hashlib.sha256(weight).hexdigest()
                if name.endswith(".safetensors") else "etag")
        (metadata / f"{name}.metadata").write_text(f"{revision}\n{etag}\n0\n")
    verify_local_base(tmp_path, revision)
    with pytest.raises(ValueError, match="revision drifted"):
        verify_local_base(tmp_path, "c" * 40)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="weight hash drifted"):
        verify_local_base(tmp_path, revision)
