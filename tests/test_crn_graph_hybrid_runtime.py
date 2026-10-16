"""Runtime-level PR83 checks for environments with Torch, RDKit and Transformers."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mechet.crn_graph_bridge import route_tool_name_greedy
from scripts.run_crn_qwen_hybrid import completed_ids, selected_shard


def test_greedy_controller_keeps_import_in_one_generation() -> None:
    class CharacterTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[ord(char) for char in text]]),
                    "attention_mask": torch.ones((1, len(text)), dtype=torch.long)}

        def decode(self, ids, **kwargs):
            return "".join(chr(token) for token in ids)

    class GreedyModel:
        def __init__(self, response):
            self.response = response
            self.calls = 0

        def generate(self, input_ids, stopping_criteria, max_new_tokens, **kwargs):
            self.calls += 1
            sequence = input_ids.clone()
            scores = []
            for char in self.response[:max_new_tokens]:
                sequence = torch.cat((sequence, torch.tensor([[ord(char)]])), dim=1)
                scores.append(torch.zeros((1, 1)))
                if any(stop(sequence, scores) for stop in stopping_criteria):
                    break
            return type("Generation", (), {"sequences": sequence, "scores": scores})()

        def compute_transition_scores(self, sequences, scores, **kwargs):
            return torch.full((1, len(scores)), -0.1)

    tokenizer = CharacterTokenizer()
    import_text = '<tool_call>\n{"name": "import_fragments", "arguments": {"fragments": [{"smiles": "O", "count": 1}]}}</tool_call>'
    model = GreedyModel(import_text)
    decision = route_tool_name_greedy(
        model, tokenizer, "cpu", "prompt", max_route_tokens=256,
    )
    assert model.calls == 1
    assert decision.raw == import_text
    assert decision.import_arguments["fragments"][0]["smiles"] == "O"
    assert decision.name_tokens < len(decision.token_ids)

    model = GreedyModel('<tool_call>\n{"name": "apply_electron_flow", "arguments": {"bad": true}}</tool_call>')
    decision = route_tool_name_greedy(
        model, tokenizer, "cpu", "prompt", max_route_tokens=256,
    )
    assert model.calls == 1
    assert decision.name == "apply_electron_flow"
    assert "arguments" not in decision.raw
    assert decision.import_arguments is None


def test_rl_cli_has_single_source_sha_option() -> None:
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/train_crn_graph_endpoint_rl.py"), "--help"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split("options:", 1)[-1].count(
        "--source-sha256 SOURCE_SHA256"
    ) == 1


def test_distributed_shards_preserve_every_selected_id() -> None:
    rows = [{"id": f"reaction_{number}"} for number in range(29)]
    shards = [selected_shard(rows, rank, 8) for rank in range(8)]
    flattened = [row["id"] for shard in shards for row in shard]
    assert len(flattened) == len(rows)
    assert set(flattened) == {row["id"] for row in rows}
    with pytest.raises(ValueError):
        selected_shard(rows, 8, 8)


def test_resume_rejects_duplicate_or_out_of_shard_records(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"id": "reaction_1"}) + "\n")
    assert completed_ids(path, {"reaction_1"}) == {"reaction_1"}
    with pytest.raises(ValueError):
        completed_ids(path, {"reaction_2"})
    path.write_text((json.dumps({"id": "reaction_1"}) + "\n") * 2)
    with pytest.raises(ValueError):
        completed_ids(path, {"reaction_1"})
