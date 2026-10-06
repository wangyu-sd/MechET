import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_natural_language_event_local as local_eval
from scripts.eval_natural_language_event_local import validate_adapter_lineage


def test_local_and_suffix_evaluation_require_matching_frozen_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "base_model": "Qwen/Qwen3-0.6B",
                "base_model_revision": "frozen-revision",
            }
        )
    )
    validate_adapter_lineage(adapter, "Qwen/Qwen3-0.6B", "frozen-revision")
    with pytest.raises(ValueError, match="base model"):
        validate_adapter_lineage(adapter, "Qwen/Qwen3-8B", "frozen-revision")
    with pytest.raises(ValueError, match="revision"):
        validate_adapter_lineage(adapter, "Qwen/Qwen3-0.6B", "wrong-revision")


def test_provisional_checkpoint_probe_requires_trainer_and_config_lineage(tmp_path: Path):
    output = tmp_path / "state_run"
    adapter = output / "checkpoint-2000"
    adapter.mkdir(parents=True)
    (adapter / "trainer_state.json").write_text(json.dumps({
        "global_step": 2000, "max_steps": 31366,
    }))
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
    }))
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    config = tmp_path / "training.yaml"
    config.write_text(
        "model_name_or_path: Qwen/Qwen3-0.6B\n"
        "output_dir: outputs/agent/state_run\n"
        "training:\n  model_revision: frozen-revision\n"
        "contract:\n  stage: state_sft\n"
    )
    with pytest.raises(FileNotFoundError, match="adapter lineage manifest missing"):
        validate_adapter_lineage(adapter, "Qwen/Qwen3-0.6B", "frozen-revision")
    report = validate_adapter_lineage(
        adapter, "Qwen/Qwen3-0.6B", "frozen-revision",
        provisional_training_config=config,
    )
    assert report["kind"] == "provisional_checkpoint"
    assert report["checkpoint_step"] == 2000
    with pytest.raises(ValueError, match="model/revision"):
        validate_adapter_lineage(
            adapter, "Qwen/Qwen3-0.6B", "wrong-revision",
            provisional_training_config=config,
        )
    (adapter / "trainer_state.json").write_text(json.dumps({
        "global_step": 1999, "max_steps": 31366,
    }))
    with pytest.raises(ValueError, match="step does not match"):
        validate_adapter_lineage(
            adapter, "Qwen/Qwen3-0.6B", "frozen-revision",
            provisional_training_config=config,
        )


def test_stage_two_local_eval_uses_frozen_history_prompts_in_decision_order(monkeypatch):
    source = {
        "id": "source-1",
        "source_id": "reaction-1",
        "metadata": {"trace_plan": {"steps": [{}]}},
    }
    monkeypatch.setattr(local_eval, "stratified_sample", lambda rows, size, seed: [source])
    monkeypatch.setattr(
        local_eval,
        "_private_states",
        lambda row: [
            {"decision_type": "import", "private_state": "C", "reference_successor": "", "event_depth": 0},
            {"decision_type": "event", "private_state": "C.O", "reference_successor": "CO", "event_depth": 1},
        ],
    )

    def decision(index, kind):
        return {
            "id": f"decision-{index}::history_v2",
            "source_id": "reaction-1",
            "metadata": {"decision_index": index, "decision_type": kind},
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"TRAJECTORY HISTORY: accepted={index}"},
                {"role": "assistant", "tool_calls": [{"function": {"name": kind, "arguments": {}}}]},
            ],
            "tools": [],
        }

    tasks, reaction_ids = local_eval.collect_tasks(
        [source], sample_reactions=1, seed=17,
        decision_rows=[decision(1, "event"), decision(0, "import")],
    )
    assert reaction_ids == ["reaction-1"]
    assert [task["key"] for task in tasks] == [
        "decision-0::history_v2", "decision-1::history_v2"
    ]
    assert [task["messages"][1]["content"] for task in tasks] == [
        "TRAJECTORY HISTORY: accepted=0", "TRAJECTORY HISTORY: accepted=1"
    ]


def test_replayed_local_scoring_uses_executor_states_not_trace_plan_states(monkeypatch):
    from scripts import earho_v2_protocol

    source = {
        "id": "source-1", "source_id": "reaction-1",
        "metadata": {"trace_plan": {"steps": [{}]}},
    }
    monkeypatch.setattr(local_eval, "stratified_sample", lambda rows, size, seed: [source])
    monkeypatch.setattr(local_eval, "_private_states", lambda row: [{
        "decision_type": "event", "private_state": "wrong-map-state",
        "reference_successor": "wrong-map-successor", "event_depth": 1,
    }])
    observed = []

    def fake_replay(row, decisions, *, compact_history):
        observed.append(compact_history)
        return SimpleNamespace(nodes=[
            SimpleNamespace(state="[CH3:1][Br:2]"),
            SimpleNamespace(state="[CH3:1].[Br:2]"),
        ])

    monkeypatch.setattr(earho_v2_protocol, "replay_reference", fake_replay)
    decision = {
        "id": "decision-0", "source_id": "reaction-1",
        "metadata": {
            "decision_index": 0, "decision_type": "event",
            "decision_contract": "unified_inventory_compressed_history_tool_decision_v2",
        },
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "history prompt"},
            {"role": "assistant", "tool_calls": [{
                "function": {"name": "apply_electron_flow", "arguments": {}},
            }]},
        ],
        "tools": [],
    }
    tasks, _ = local_eval.collect_tasks(
        [source], sample_reactions=1, seed=17, decision_rows=[decision],
        replay_decision_states=True,
    )
    assert observed == [True]
    assert tasks[0]["private_state"] == "[CH3:1][Br:2]"
    assert tasks[0]["reference_successor"] == "[CH3:1].[Br:2]"
    with pytest.raises(ValueError, match="requires frozen decision rows"):
        local_eval.collect_tasks(
            [source], sample_reactions=1, seed=17,
            replay_decision_states=True,
        )
