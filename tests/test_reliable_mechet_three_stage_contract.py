from pathlib import Path
import json

import pytest
import yaml

from scripts.run_earho_v2 import (
    _critic_config, prepare, resolve_reliable_paths, validate_contract,
    validate_reliable_contract,
)
from scripts.run_natural_language_anchor_branch_rl import worker_command, run_workers, _sha256
from scripts.run_anchor_branch_rl import run_train


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_reliable_mechet_three_stage_lineage_is_explicit():
    umbrella = load_yaml("configs/experiments/reliable_mechet_three_stage_v1.yaml")
    stage1 = load_yaml("configs/agent/natural_language_event_v2_qwen3_0_6b.yaml")
    stage2 = load_yaml("configs/agent/natural_language_history_v2_qwen3_0_6b.yaml")
    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")

    expected_model = "Qwen/Qwen3-0.6B"
    expected_revision = "c1899de289a04d12100db370d81485cdf75e47ca"

    assert umbrella["model_policy"]["primary"]["model"] == expected_model
    assert umbrella["model_policy"]["primary"]["revision"] == expected_revision

    assert stage1["model_name_or_path"] == expected_model
    assert stage2["model_name_or_path"] == expected_model
    assert stage3["model_name_or_path"] == expected_model
    assert stage1["training"]["model_revision"] == expected_revision
    assert stage2["training"]["model_revision"] == expected_revision
    assert stage3["model_revision"] == expected_revision

    assert stage2["initial_adapter_path"] == stage1["output_dir"]
    assert stage3["initial_adapter_path"] == stage2["output_dir"]
    assert stage3["initial_adapter_model_sha256"] == "REPLACE_WITH_FROZEN_STAGE_II_SHA256"


def test_three_stages_keep_reverse_et_as_the_main_prediction_path():
    umbrella = load_yaml("configs/experiments/reliable_mechet_three_stage_v1.yaml")
    stage1 = load_yaml("configs/agent/natural_language_event_v2_qwen3_0_6b.yaml")
    stage2 = load_yaml("configs/agent/natural_language_history_v2_qwen3_0_6b.yaml")

    assert umbrella["stages"]["stage1_state_sft"]["output_decisions"] == [
        "import_fragment",
        "reverse_electron_flow",
        "finish",
    ]
    assert stage1["contract"]["electron_flow_model_generated"] is True
    assert stage2["contract"]["electron_flow_model_generated"] is True
    assert stage1["contract"]["terminal_tool"] == "finish_trace"
    assert stage2["contract"]["terminal_tool"] == "finish_trace"


def test_endpoint_and_process_denominators_stay_separate():
    umbrella = load_yaml("configs/experiments/reliable_mechet_three_stage_v1.yaml")
    assert umbrella["data"]["headline_endpoint_denominator"] == 28971
    assert umbrella["data"]["process_analysis_denominator"] == 28967


def test_reliable_earho_refuses_unfrozen_stage_ii_before_large_data_scan():
    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")
    with pytest.raises(ValueError, match="freeze the Stage-II adapter SHA-256"):
        validate_contract(stage3)


def test_reliable_earho_requires_structural_endpoint_reward():
    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")
    stage3["reward"].pop("endpoint_metric")
    with pytest.raises(ValueError, match="structural endpoint reward"):
        validate_reliable_contract(stage3)


def test_reliable_earho_requires_the_same_decision_and_import_budget_as_rollout():
    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")
    stage3["rollout"]["max_imports"] = 64
    with pytest.raises(ValueError, match="40-decision/32-import budget"):
        validate_reliable_contract(stage3)


def test_reliable_earho_paths_bind_pr_code_to_shared_artifacts(tmp_path: Path):
    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")
    resolved = resolve_reliable_paths(stage3, tmp_path)
    assert Path(resolved["train_file"]).is_relative_to(tmp_path)
    assert Path(resolved["output_dir"]).is_relative_to(tmp_path)
    assert stage3["train_file"].startswith("data/")


def test_reliable_earho_workers_keep_unified_v2_prompt_contract():
    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")
    command = worker_command(
        stage3, Path("source.jsonl"), Path("adapter"), Path("rank0.jsonl"),
        0, frontier=2, round_index=0, evaluation=False,
    )
    assert "--protocol-v2" in command
    assert "--reject-target-retained-finish" in command
    assert command[command.index("--endpoint-metric") + 1] == "structural"
    assert command[command.index("--model") + 1] == stage3["model_snapshot"]

    legacy = dict(stage3, protocol_version="trajectory_history_v2")
    assert "--endpoint-metric" not in worker_command(
        legacy, Path("source.jsonl"), Path("adapter"), Path("rank0.jsonl"),
        0, frontier=2, round_index=0, evaluation=False,
    )


def test_reliable_earho_critic_uses_same_small_base(tmp_path: Path):
    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")
    dataset = tmp_path / "critic_data"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({
        "splits": {"train": {"rows": 12}, "valid": {"rows": 3}},
    }))
    critic_path = _critic_config(
        stage3, dataset, tmp_path / "critic_output", tmp_path / "parent",
    )
    critic = yaml.safe_load(critic_path.read_text())
    assert critic["model_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert critic["training"]["model_revision"] == stage3["model_revision"]
    assert critic["training"]["qlora"] is False
    assert critic["contract"]["reaction_denominator"] == stage3["reaction_denominator"]


def test_reliable_earho_prepare_streams_large_source(monkeypatch, tmp_path: Path):
    import scripts.run_earho_v2 as driver

    stage3 = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")
    train, valid = tmp_path / "train.jsonl", tmp_path / "valid.jsonl"
    train.write_text("".join(json.dumps({"id": str(i), "source_id": str(i)}) + "\n" for i in range(3)))
    valid.write_text("".join(json.dumps({"id": str(i), "source_id": str(i)}) + "\n" for i in range(2)))
    stage3.update(
        train_file=str(train), validation_file=str(valid),
        history_file=str(tmp_path / "history_train.jsonl"),
        history_validation_file=str(tmp_path / "history_valid.jsonl"),
        reaction_denominator={"train": 3, "valid": 2, "test": 0},
        rounds=1, products_per_round=2, validation_monitor_rows=1,
    )
    observed_budgets = []

    def fake_attach(rows, _path, *, max_imports):
        observed_budgets.append(max_imports)
        return rows

    monkeypatch.setattr(driver, "_attach_decisions", fake_attach)
    stage3["rollout"]["max_imports"] = 32
    monkeypatch.setattr(driver, "read_rows", lambda _path: (_ for _ in ()).throw(
        AssertionError("reliable preparation must not load the full source")
    ))
    output = tmp_path / "prepared"
    prepare(stage3, output)
    plan = json.loads((output / "plan.json").read_text())
    assert plan["source_reactions"] == 3
    assert plan["selected_train_reactions"] == 2
    assert plan["protocol_version"] == "reliable_mechet_three_stage_v1"
    assert plan["private_product_mapping_basis"] == "source_original_mapped_product"
    assert plan["product_only_private_remap"] is False
    assert observed_budgets == [32, 32]

    plan.pop("private_product_mapping_basis")
    (output / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="private-map provenance"):
        prepare(stage3, output)


def test_reliable_earho_actor_update_uses_natural_language_stage(monkeypatch, tmp_path: Path):
    import scripts.run_anchor_branch_rl as driver

    data = tmp_path / "training.jsonl"
    data.write_text(json.dumps({"kind": "rl", "advantage": 1.0}) + "\n")
    output = tmp_path / "actor_training"
    commands = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(command)
        (output / "adapter").mkdir(parents=True)
        (output / "adapter" / "adapter_model.safetensors").write_bytes(b"smoke")

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    result = run_train(
        {"model_snapshot": "/model", "initial_adapter_path": "/parent"},
        data, tmp_path / "parent", output, 17,
        stage_script="scripts/natural_language_anchor_branch_stage.py",
    )
    assert result == output / "adapter"
    assert "scripts/natural_language_anchor_branch_stage.py" in commands[0]
    assert "scripts/anchor_branch_stage.py" not in commands[0]


def test_reliable_earho_resume_rejects_stale_collection_lineage(monkeypatch, tmp_path: Path):
    import scripts.run_natural_language_anchor_branch_rl as collector

    cfg = load_yaml("configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"actor")
    output = tmp_path / "collection"
    output.mkdir()
    for rank in range(8):
        (output / f"rank{rank}.jsonl").write_text("{}\n")
    marker = output / "collection_done.json"
    lineage = {
        "protocol_version": cfg["protocol_version"],
        "model_revision": cfg["model_revision"],
        "adapter_model_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "value_adapter": "", "value_adapter_model_sha256": "",
        "source_sha256": _sha256(source),
        "endpoint_metric": "structural",
        "adapter": str(adapter), "frontier": 2, "round": 0,
        "evaluation": False,
    }
    marker.write_text(json.dumps(lineage))
    monkeypatch.setattr(collector, "summarize", lambda _shards: {
        "collector_error_rate": 0.0, "group_summaries": [],
    })
    shards, _ = run_workers(
        cfg, source, adapter, output, frontier=2, round_index=0,
        evaluation=False,
    )
    assert len(shards) == 8
    marker.write_text(json.dumps({**lineage, "source_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="collection lineage changed: source_sha256"):
        run_workers(
            cfg, source, adapter, output, frontier=2, round_index=0,
            evaluation=False,
        )
    marker.write_text(json.dumps({**lineage, "endpoint_metric": "full"}))
    with pytest.raises(ValueError, match="collection lineage changed: endpoint_metric"):
        run_workers(
            cfg, source, adapter, output, frontier=2, round_index=0,
            evaluation=False,
        )
    critic = tmp_path / "critic"
    critic.mkdir()
    (critic / "adapter_model.safetensors").write_bytes(b"first-critic")
    with_critic = {**cfg, "value_adapter_path": str(critic)}
    marker.write_text(json.dumps({
        **lineage,
        "value_adapter": str(critic),
        "value_adapter_model_sha256": _sha256(critic / "adapter_model.safetensors"),
    }))
    run_workers(
        with_critic, source, adapter, output, frontier=2, round_index=0,
        evaluation=False,
    )
    (critic / "adapter_model.safetensors").write_bytes(b"changed-critic")
    with pytest.raises(ValueError, match="collection lineage changed: value_adapter_model_sha256"):
        run_workers(
            with_critic, source, adapter, output, frontier=2, round_index=0,
            evaluation=False,
        )
