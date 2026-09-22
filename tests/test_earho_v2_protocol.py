"""Contracts for paper Stage III: v2 history, real first divergence, private labels."""

from __future__ import annotations

from dataclasses import replace
import hashlib
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


def test_chemical_successor_pooling_ignores_action_surface():
    left = _v2_successor_fingerprint("[Na+:3].[CH3:1][Br:2]", False, "alpha")
    right = _v2_successor_fingerprint("[CH3:9][Br:8].[Na+:7]", False, "beta")
    assert left == right
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
            "score": {"first_successor_state": "[CH3:1][Br:2].[K+:3]",
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
