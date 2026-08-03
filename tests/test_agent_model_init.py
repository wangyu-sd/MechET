import importlib.util
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


def test_lineage_report_records_adapter_and_manifest(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    digest = agent_model_init.path_sha256(adapter)
    report = agent_model_init.validate_lineage(
        {
            "initial_adapter_path": str(adapter),
            "initial_adapter_sha256": digest,
            "require_initial_adapter": True,
            "tool_sft_manifest": str(manifest),
            "environment_revision": "trace-v1",
            "executor_revision": "proof-v1",
        }
    )
    assert report["initial_adapter_exists"] is True
    assert report["initial_adapter_hash_matches"] is True
    assert report["environment_revision"] == "trace-v1"


def test_required_adapter_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="initial_adapter_path"):
        agent_model_init.validate_lineage(
            {
                "initial_adapter_path": str(tmp_path / "missing"),
                "require_initial_adapter": True,
            }
        )


def test_declared_hash_must_match(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "weights.bin").write_bytes(b"weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        agent_model_init.validate_lineage(
            {
                "initial_adapter_path": str(adapter),
                "initial_adapter_sha256": "wrong",
                "require_initial_adapter": True,
            }
        )


def test_optional_adapter_is_reported_without_failure(tmp_path):
    report = agent_model_init.validate_lineage(
        {
            "initial_adapter_path": str(tmp_path / "missing"),
            "require_initial_adapter": False,
        }
    )
    assert report["initial_adapter_exists"] is False
