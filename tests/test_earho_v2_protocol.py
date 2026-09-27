"""Contracts for paper Stage III: v2 history, real first divergence, private labels."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from mechet.in_place_grounded_flow import mapped_atom_numbers
from mechet.trajectory_history import TrajectoryHistory
from scripts.build_natural_language_event_sft import _decision_row
from scripts.build_natural_language_history_sft import add_history
from scripts.earho_v2_protocol import (
    anchor_task,
    locate_first_divergence,
    promote_productive_horizon,
    replay_reference,
)
from scripts.build_earho_v2_successor_value import build_rows
from scripts.natural_language_anchor_branch_stage import (
    _messages, _node, _v2_probe, _v2_successor_fingerprint,
    _verified_replay_records,
)
from scripts.run_natural_language_value_search import (
    Action, Node, execute, policy_prompt, visible,
)


def fixture(reaction_id: str = "toy"):
    mapped = "[CH3:1][Br:2]"
    source = {
        "id": f"toy:{reaction_id}",
        "source_id": reaction_id,
        "target_smiles": mapped,
        "expected_precursor": "[CH3:1][Br:2].[Na+:3]",
        "full_precursor_state": "[CH3:1][Br:2].[Na+:3]",
    }
    public = dict(source, target_smiles=visible(mapped),
                  expected_precursor="CBr.[Na+]")
    node = Node(
        target="CBr", state=mapped,
        next_map=max(mapped_atom_numbers(mapped)) + 1, visited={"CBr"},
    )
    imported = {"fragments": [{"smiles": "[Na+]", "count": 1,
                                 "purpose": "endpoint_context"}]}
    first, error = execute(
        node, Action("import_fragments", imported, "", 0.0, 1), max_imports=4
    )
    assert first is not None and not error
    final, error = execute(
        first, Action("finish_trace", {}, "", 0.0, 1), max_imports=4
    )
    assert final is not None and not error
    raw = [
        _decision_row(
            row=public, sequence_index=0, decision_type="import",
            mapped_state=node.state, name="import_fragments",
            arguments=imported, result=first.actions[-1]["result"],
        ),
        _decision_row(
            row=public, sequence_index=1, decision_type="finish",
            mapped_state=first.state, name="finish_trace",
            arguments={}, result=final.actions[-1]["result"],
        ),
    ]
    history = TrajectoryHistory()
    decisions = []
    for row in raw:
        decisions.append(add_history(row, history))
        call = row["messages"][2]["tool_calls"][0]["function"]
        result = first.actions[-1]["result"] if len(decisions) == 1 else final.actions[-1]["result"]
        history = history.accept(call["name"], call["arguments"], result)
    return source, decisions


def test_reference_replay_matches_exact_stage_ii_prompt_and_endpoint():
    source, decisions = fixture()
    reference = replay_reference(source, decisions)
    assert len(reference.nodes) == 3
    assert reference.nodes[-1].terminal
    task = anchor_task(reference, 1, divergence_reason="EXECUTED_SUCCESSOR_DIVERGED")
    assert task.reference_name == "finish_trace"
    assert task.anchor_actions[0]["name"] == "import_fragments"
    assert "accepted_actions: 1" in decisions[1]["messages"][1]["content"]
    assert task.reference_next_state == reference.nodes[-1].state
    actor_prompt = _messages(task, "unified")[1]["content"]
    assert actor_prompt == policy_prompt(
        task.target, task.anchor_state, include_inventory=True,
        actions=task.anchor_actions, compact_history=True,
    )
    assert "expected_precursor" not in actor_prompt
    assert "reference_successor" not in actor_prompt
    assert len(_node(task).actions) == 1


def test_verified_fallback_replays_whole_remaining_trajectory_with_matching_prefix():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                tools=None, enable_thinking=False):
            assert not tokenize and not enable_thinking
            rendered = ""
            for index, message in enumerate(messages):
                content = str(message.get("content") or "")
                if message.get("tool_calls"):
                    content += "<tool_call>" + str(message["tool_calls"]) + "</tool_call>"
                followed_by_tool = (
                    message["role"] == "assistant"
                    and index + 1 < len(messages)
                    and messages[index + 1]["role"] == "tool"
                )
                if message["role"] == "assistant" and not followed_by_tool:
                    content = "<think>\n\n</think>\n\n" + content
                rendered += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
            if add_generation_prompt:
                rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            return rendered

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(char) for char in text]}

    source, decisions = fixture()
    reference = replay_reference(source, decisions)
    first = anchor_task(reference, 0, divergence_reason="INVALID_ACTION")
    args = SimpleNamespace(legacy_dual_prompt=False)
    records = _verified_replay_records(
        Tokenizer(), first, source, first, 100000, args, reference=reference
    )
    assert [item["reference_decision_index"] for item in records] == [0, 1]
    assert [item["reference_action"] for item in records] == [
        "import_fragments", "finish_trace"
    ]
    for index, record in enumerate(records):
        ids = record["input_ids"]
        mask = record["loss_mask"]
        boundary = mask.index(1)
        prefix = "".join(chr(value) for value in ids[:boundary])
        completion = "".join(chr(value) for value in ids[boundary:])
        assert prefix.endswith("<|im_start|>assistant\n")
        assert "<think>\n\n</think>\n\n" not in prefix
        assert completion.startswith("<tool_call>")
        assert "<think>" not in completion
        assert "<|im_start|>tool" not in completion
        assert f"accepted_actions: {index}" in prefix
        assert all(value == 0 for value in mask[:boundary])
        assert all(value == 1 for value in mask[boundary:])
    second = anchor_task(reference, 1, divergence_reason="INVALID_ACTION")
    suffix = _verified_replay_records(
        Tokenizer(), second, source, second, 100000, args, reference=reference
    )
    assert [item["reference_decision_index"] for item in suffix] == [1]


def test_real_qwen_verified_replay_supervises_tool_call_without_think_when_available():
    from transformers import AutoTokenizer

    snapshot = Path(
        "/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache/"
        "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    )
    if not snapshot.is_dir():
        pytest.skip("local Qwen3-8B tokenizer snapshot is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    source, decisions = fixture()
    reference = replay_reference(source, decisions)
    task = anchor_task(reference, 0, divergence_reason="INVALID_ACTION")
    records = _verified_replay_records(
        tokenizer, task, source, task, 100000,
        SimpleNamespace(legacy_dual_prompt=False), reference=reference,
    )
    assert len(records) == 2
    for record in records:
        boundary = record["loss_mask"].index(1)
        prefix = tokenizer.decode(record["input_ids"][:boundary], skip_special_tokens=False)
        completion = tokenizer.decode(record["input_ids"][boundary:], skip_special_tokens=False)
        assert prefix.endswith("<|im_start|>assistant\n")
        assert completion.startswith("<tool_call>")
        assert "<think>" not in completion
        assert "<|im_start|>tool" not in completion


def test_product_probe_uses_first_executed_successor_mismatch():
    source, decisions = fixture()
    reference = replay_reference(source, decisions)
    aligned = locate_first_divergence(
        reference, lambda node: (reference.nodes[len(node.actions) + 1], "")
    )
    assert aligned.decision_index is None and aligned.exact_endpoint
    invalid = locate_first_divergence(
        reference, lambda node: (None, "INVALID_IMPORT")
    )
    assert invalid.decision_index == 0 and invalid.reason == "INVALID_IMPORT"
    late = locate_first_divergence(
        reference,
        lambda node: (reference.nodes[1], "")
        if not node.actions else (None, "BAD_FINISH"),
    )
    assert late.decision_index == 1 and late.accepted_actions == 1
    early_finish = replace(reference.nodes[0], actions=[{"name": "finish_trace"}], terminal=True)
    divergent = locate_first_divergence(
        reference, lambda node: (early_finish, "")
    )
    assert divergent.decision_index == 0
    assert divergent.reason == "EXECUTED_SUCCESSOR_DIVERGED"


def test_v2_collector_probe_conditions_on_accepted_history(monkeypatch):
    source, decisions = fixture()
    reference = replay_reference(source, decisions)
    import scripts.natural_language_anchor_branch_stage as stage

    seen_actions = []
    def prompt(tokenizer, task, state, mode, *, actions):
        seen_actions.append(len(actions))
        return [1]
    monkeypatch.setattr(stage, "_render_prompt", prompt)
    responses = iter([
        {"error": "", "name": "import_fragments",
         "arguments": decisions[0]["messages"][2]["tool_calls"][0]["function"]["arguments"],
         "text": "import", "ids": [1], "logps": [0.0]},
        {"error": "BAD_FINISH", "name": "", "arguments": {},
         "text": "bad", "ids": [1], "logps": [0.0]},
    ])
    monkeypatch.setattr(stage, "_decode_action", lambda *args: next(responses))
    llm = SimpleNamespace(generate=lambda *args, **kwargs: [
        SimpleNamespace(outputs=[object()])
    ])
    args = SimpleNamespace(max_new_tokens=8, max_context=32, max_imports=4,
                           reject_target_retained_finish=True)
    result = _v2_probe(llm, object(), object(), object(), [], reference, args)
    assert result.decision_index == 1
    assert seen_actions == [0, 1]


def test_reference_prompt_drift_and_private_leak_are_rejected():
    source, decisions = fixture()
    changed = [dict(item) for item in decisions]
    changed[1] = dict(changed[1], messages=[dict(item) for item in changed[1]["messages"]])
    changed[1]["messages"][1]["content"] += "\nexpected_precursor: CBr.[Na+]"
    with pytest.raises(ValueError, match="prompt/replay drift"):
        replay_reference(source, changed)


def test_v2_collection_rejects_stale_completed_marker(tmp_path):
    import json

    from scripts.run_natural_language_anchor_branch_rl import run_workers

    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    adapter = tmp_path / "actor"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"actor")
    output = tmp_path / "collection"
    output.mkdir()
    (output / "collection_done.json").write_text(json.dumps({"adapter": str(adapter)}))
    with pytest.raises(ValueError, match="different inputs"):
        run_workers(
            {"protocol_version": "trajectory_history_v2"},
            source, adapter, output, frontier=2, round_index=0,
            evaluation=False,
        )


def test_v2_runtime_lineage_survives_fresh_temporary_extraction(tmp_path):
    from scripts.run_natural_language_anchor_branch_rl import _runtime_lineage_identity

    source = tmp_path / "pinned_ceph_runtime"
    config = {"vllm_runtime": str(source)}
    markers = []
    for name in ("stage_a", "stage_b"):
        marker = tmp_path / name / ".mechet_vllm_runtime_complete"
        marker.parent.mkdir()
        marker.write_text("vllm 0.8.5\n")
        markers.append(marker)
    assert _runtime_lineage_identity(config, markers[0]) == _runtime_lineage_identity(
        config, markers[1]
    )
    markers[1].write_text("different runtime\n")
    assert _runtime_lineage_identity(config, markers[0]) != _runtime_lineage_identity(
        config, markers[1]
    )
    with pytest.raises(ValueError, match="pinned vLLM runtime source"):
        _runtime_lineage_identity({}, markers[0])


def test_chemical_successor_pooling_ignores_action_surface():
    left = _v2_successor_fingerprint("[Na+:3].[CH3:1][Br:2]", False, "alpha")
    right = _v2_successor_fingerprint("[CH3:9][Br:8].[Na+:7]", False, "beta")
    assert left == right
    assert left != _v2_successor_fingerprint("[CH3:9][Br:8].[Na+:7]", True, "finish")
    assert left != _v2_successor_fingerprint("[CH3:1][Br:2]", False, "alpha")


def test_horizon_promotes_on_verified_successor_not_wrong_endpoint():
    rows = [
        {"is_full_episode": False, "effective": True,
         "successor_success": True, "endpoint_success": False},
        {"is_full_episode": False, "effective": True,
         "successor_success": True, "endpoint_success": False},
        {"is_full_episode": True, "effective": True,
         "successor_success": False, "endpoint_success": True},
    ]
    horizon, report = promote_productive_horizon(
        2, rows, minimum_effective_groups=2,
        productive_pass_rate=0.75, maximum_horizon=4,
    )
    assert horizon == 3 and report["verified_productive_at_k"] == 1.0
    no_support = [dict(row, successor_success=False) for row in rows[:2]]
    horizon, _ = promote_productive_horizon(
        2, no_support, minimum_effective_groups=2,
        productive_pass_rate=0.75, maximum_horizon=4,
    )
    assert horizon == 2


def test_successor_value_labels_are_reaction_disjoint_and_private():
    valid_id = next(
        str(index) for index in range(100)
        if int.from_bytes(hashlib.sha256(f"17:{index}".encode()).digest()[:4], "big") % 10 == 0
    )
    train_id = next(
        str(index) for index in range(100)
        if int.from_bytes(hashlib.sha256(f"17:{index}".encode()).digest()[:4], "big") % 10 != 0
    )
    sources = []
    rollouts = []
    for reaction_id in (train_id, valid_id):
        source, decisions = fixture(reaction_id)
        source["earho_v2_reference_decisions"] = decisions
        sources.append(source)
        reference = replay_reference(source, decisions)
        task = anchor_task(reference, 0, divergence_reason="EXECUTED_SUCCESSOR_DIVERGED")
        rollouts.append({
            "kind": "rl", "id": reaction_id,
            "anchor": {"version": "earho_first_divergence_v2",
                       "state_hash": task.state_hash, "decision_index": 0,
                       "divergence_reason": task.divergence_reason},
            # Collector persists the public, unmapped state, not the private
            # executor atom maps. This is the deployed rollout contract.
            "score": {"first_successor_state": "CBr.[K+]",
                      "first_successor_terminal": False, "correct": False},
            "reward": -0.1,
        })
    rows, stats = build_rows(sources, rollouts, max_hard_negatives=2)
    assert stats["reference_positives"] == 2
    assert stats["hard_negatives"] == 2
    assert {row["source_id"] for row in rows["train"]}.isdisjoint(
        {row["source_id"] for row in rows["valid"]}
    )
    for row in rows["train"] + rows["valid"]:
        prompt = row["messages"][1]["content"]
        assert "expected_precursor" not in prompt
        assert "reference_successor" not in prompt
