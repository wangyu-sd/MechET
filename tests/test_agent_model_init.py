import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "agent_model_init.py"
    spec = importlib.util.spec_from_file_location("agent_model_init", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent_model_init = load_module()


def valid_adapter(
    tmp_path,
    *,
    environment="trace-v2",
    executor="proof-v2",
    model_revision="revision-abc",
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    contract = adapter / "data_contract.json"
    contract.write_text(json.dumps({"ok": True}))
    digest = agent_model_init.path_sha256(adapter)
    manifest = {
        "artifact_type": "trainable_peft_adapter",
        "adapter_path": str(adapter),
        "adapter_sha256": digest,
        "base_model": "Qwen/Qwen3-0.6B",
        "base_model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "condition_name": "trace_no_knowledge",
        "data_contract": str(contract),
        "environment_revision": environment,
        "executor_revision": executor,
    }
    (adapter / "adapter_manifest.json").write_text(json.dumps(manifest))
    return adapter, digest


def test_lineage_report_records_and_validates_adapter_manifest(tmp_path):
    adapter, digest = valid_adapter(tmp_path)
    report = agent_model_init.validate_lineage(
        {
            "model_name_or_path": "Qwen/Qwen3-0.6B",
            "initial_adapter_path": str(adapter),
            "initial_adapter_sha256": "auto",
            "require_initial_adapter": True,
            "environment_revision": "trace-v2",
            "executor_revision": "proof-v2",
        }
    )
    assert report["initial_adapter_exists"] is True
    assert report["initial_adapter_hash_matches"] is True
    assert report["initial_adapter_sha256_actual"] == digest
    assert report["adapter_artifact_type"] == "trainable_peft_adapter"
    assert report["resolved_model_revision"] == "revision-abc"


def test_required_adapter_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="initial_adapter_path"):
        agent_model_init.validate_lineage(
            {
                "model_name_or_path": "Qwen/Qwen3-0.6B",
                "initial_adapter_path": str(tmp_path / "missing"),
                "require_initial_adapter": True,
            }
        )


def test_required_adapter_manifest_must_exist(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "weights.bin").write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="manifest"):
        agent_model_init.validate_lineage(
            {
                "model_name_or_path": "Qwen/Qwen3-0.6B",
                "initial_adapter_path": str(adapter),
                "initial_adapter_sha256": agent_model_init.path_sha256(adapter),
                "require_initial_adapter": True,
            }
        )


def test_declared_hash_must_match(tmp_path):
    adapter, _ = valid_adapter(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        agent_model_init.validate_lineage(
            {
                "model_name_or_path": "Qwen/Qwen3-0.6B",
                "initial_adapter_path": str(adapter),
                "initial_adapter_sha256": "wrong",
                "require_initial_adapter": True,
            }
        )


def test_environment_revision_must_match(tmp_path):
    adapter, _ = valid_adapter(tmp_path, environment="trace-v1")
    with pytest.raises(ValueError, match="environment_revision mismatch"):
        agent_model_init.validate_lineage(
            {
                "model_name_or_path": "Qwen/Qwen3-0.6B",
                "initial_adapter_path": str(adapter),
                "initial_adapter_sha256": "auto",
                "require_initial_adapter": True,
                "environment_revision": "trace-v2",
            }
        )


def test_model_revision_must_match(tmp_path):
    adapter, _ = valid_adapter(tmp_path, model_revision="revision-old")
    with pytest.raises(ValueError, match="base model revision mismatch"):
        agent_model_init.validate_lineage(
            {
                "model_name_or_path": "Qwen/Qwen3-0.6B",
                "initial_adapter_path": str(adapter),
                "initial_adapter_sha256": "auto",
                "require_initial_adapter": True,
                "training": {"model_revision": "revision-new"},
            }
        )


def test_required_adapter_needs_revision_or_explicit_config(tmp_path):
    adapter, _ = valid_adapter(tmp_path, model_revision="")
    with pytest.raises(ValueError, match="no frozen base-model revision"):
        agent_model_init.validate_lineage(
            {
                "model_name_or_path": "Qwen/Qwen3-0.6B",
                "initial_adapter_path": str(adapter),
                "initial_adapter_sha256": "auto",
                "require_initial_adapter": True,
            }
        )
    report = agent_model_init.validate_lineage(
        {
            "model_name_or_path": "Qwen/Qwen3-0.6B",
            "initial_adapter_path": str(adapter),
            "initial_adapter_sha256": "auto",
            "require_initial_adapter": True,
            "training": {"model_revision": "revision-explicit"},
        }
    )
    assert report["resolved_model_revision"] == "revision-explicit"


def test_optional_adapter_is_reported_without_failure(tmp_path):
    report = agent_model_init.validate_lineage(
        {
            "model_name_or_path": "Qwen/Qwen3-0.6B",
            "initial_adapter_path": str(tmp_path / "missing"),
            "require_initial_adapter": False,
        }
    )
    assert report["initial_adapter_exists"] is False
