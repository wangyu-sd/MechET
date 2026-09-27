import json

import pytest

from scripts.run_earho_v2 import _dataset_contract, _sample_reactions
from scripts.run_natural_language_anchor_branch_rl import worker_command
from scripts.run_anchor_branch_rl import run_train


def test_streaming_sample_is_deterministic_and_unique(tmp_path):
    source = tmp_path / "reactions.jsonl"
    source.write_text("".join(json.dumps({"source_id": str(i)}) + "\n" for i in range(20)))
    first = _sample_reactions(source, 20, 7, 17)
    second = _sample_reactions(source, 20, 7, 17)
    assert first == second
    assert len({row["source_id"] for row in first}) == 7
    with pytest.raises(ValueError, match="row count changed"):
        _sample_reactions(source, 21, 7, 17)


def test_only_pinned_dataset_contracts_are_accepted():
    assert _dataset_contract({"dataset_id": "flower_strict_executable"})["reaction_denominator"]["train"] == 257167
    assert _dataset_contract({"dataset_id": "mech_uspto31k_current_compiler"})["reaction_denominator"]["train"] == 10152
    with pytest.raises(ValueError, match="unknown EARHO dataset_id"):
        _dataset_contract({"dataset_id": "unfiltered"})


def test_precision_matched_actor_flag_reaches_every_collector(tmp_path):
    cfg = {
        "model_snapshot": "/model",
        "actor_quantization": "bnb_nf4_double_quant_bf16",
        "candidates_per_product": 8,
        "seed": 17,
        "invalid_penalty": 0.1,
        "reward": {
            "wrong_terminal_penalty": 0.5,
            "endpoint_similarity_weight": 0.45,
            "first_successor_progress_weight": 0.25,
            "nonexact_reward_ceiling": 0.01,
        },
        "curriculum": {"full_episode_fraction": 0.25},
        "rollout": {
            "temperature": 1.0,
            "max_new_tokens": 384,
            "max_context": 4096,
            "max_decisions": 40,
            "max_imports": 32,
        },
    }
    command = worker_command(
        cfg, tmp_path / "data.jsonl", tmp_path / "adapter",
        tmp_path / "rollout.jsonl", 0,
        frontier=2, round_index=0, evaluation=False,
    )
    position = command.index("--actor-quantization")
    assert command[position + 1] == "bnb_nf4_double_quant_bf16"


def test_precision_matched_actor_flag_reaches_learner(tmp_path, monkeypatch):
    source = tmp_path / "updates.jsonl"
    source.write_text(json.dumps({"kind": "verified_replay"}) + "\n")
    output = tmp_path / "training"
    observed = []

    def fake_run(command, *, check):
        assert check
        observed.extend(command)
        (output / "adapter").mkdir(parents=True)
        (output / "adapter" / "adapter_model.safetensors").touch()

    monkeypatch.setattr("scripts.run_anchor_branch_rl.subprocess.run", fake_run)
    run_train(
        {"model_snapshot": "/model", "initial_adapter_path": "/parent",
         "actor_quantization": "bnb_nf4_double_quant_bf16"},
        source, tmp_path / "parent", output, 17,
    )
    assert "--qlora-nf4" in observed
