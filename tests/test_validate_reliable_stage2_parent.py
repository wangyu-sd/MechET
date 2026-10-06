import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_reliable_stage2_parent import validate_parent


def _fixture(tmp_path: Path):
    parent = tmp_path / "final_state_adapter"
    parent.mkdir()
    weights = b"completed-stage-one-weights"
    (parent / "adapter_model.safetensors").write_bytes(weights)
    (parent / "adapter_config.json").write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(weights).hexdigest()
    config = {
        "condition_name": "reliable_state_sft",
        "model_name_or_path": "Qwen/Qwen3-0.6B",
        "output_dir": "outputs/agent/state",
        "training": {"model_revision": "frozen-revision"},
    }
    manifest = {"splits": {"train": {"output_sha256": "train-source-sha"}}}
    adapter = {
        "artifact_type": "trainable_peft_adapter",
        "condition_name": config["condition_name"],
        "base_model": config["model_name_or_path"],
        "requested_model_revision": "frozen-revision",
        "train_file_sha256": "train-source-sha",
        "data_contract": "outputs/agent/state/data_contract.json",
    }
    contract = {
        "artifact_type": "tool_sft_data_contract",
        "condition_name": config["condition_name"],
        "model_name_or_path": config["model_name_or_path"],
        "train_file_sha256": "train-source-sha",
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "assistant_only_loss": True,
        "packing": False,
        "base_model_revision": "frozen-revision",
    }
    (parent / "adapter_manifest.json").write_text(json.dumps(adapter), encoding="utf-8")
    (parent / "data_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return parent, digest, config, manifest


def test_final_stage_one_parent_must_match_weight_and_data_lineage(tmp_path: Path):
    parent, digest, config, manifest = _fixture(tmp_path)
    assert validate_parent(
        parent, expected_sha256=digest, stage1_config=config, stage1_manifest=manifest
    ) == digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_parent(
            parent, expected_sha256="0" * 64, stage1_config=config,
            stage1_manifest=manifest,
        )
    manifest["splits"]["train"]["output_sha256"] = "other-data"
    with pytest.raises(ValueError, match="train_file_sha256"):
        validate_parent(
            parent, expected_sha256=digest, stage1_config=config,
            stage1_manifest=manifest,
        )


def test_checkpoint_without_final_adapter_manifest_is_not_stage_two_parent(tmp_path: Path):
    parent, digest, config, manifest = _fixture(tmp_path)
    (parent / "adapter_manifest.json").unlink()
    with pytest.raises(ValueError, match="missing final Stage-I artifact"):
        validate_parent(
            parent, expected_sha256=digest, stage1_config=config,
            stage1_manifest=manifest,
        )


def test_stage_one_parent_cannot_be_a_warm_started_adapter(tmp_path: Path):
    parent, digest, config, manifest = _fixture(tmp_path)
    report_path = parent / "data_contract.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["initial_adapter_path"] = "a-different-parent"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="initialized from another adapter"):
        validate_parent(
            parent, expected_sha256=digest, stage1_config=config,
            stage1_manifest=manifest,
        )


def test_stage_two_task_template_is_fail_closed_and_streams_logs():
    root = Path(__file__).resolve().parents[1]
    task = json.loads((root / "configs/taiji" /
        "meteor_mechet_reliable_trajectory_06b_1ep_8a100_qy_TEMPLATE.json"
    ).read_text(encoding="utf-8"))
    assert task["task_flag"].startswith("meteor")
    assert task["readable_name"].startswith("meteor")
    assert task["GPUName"] == "A100" and task["host_gpu_num"] == 8
    assert "REPLACE_WITH_RUN_ID" in task["task_flag"]
    assert "REPLACE_WITH_PRIVATE_INIT_CMD" in task["init_cmd"]
    assert "MECHET_RELIABLE_STAGE=trajectory" in task["start_cmd"]
    assert "REPLACE_WITH_FROZEN_STAGE_I_WEIGHTS_SHA256" in task["start_cmd"]
    assert "taiji_run_with_heartbeat.sh" in task["start_cmd"]
    assert "TAIJI_MIRROR_PID1_STDOUT=1" in task["start_cmd"]
