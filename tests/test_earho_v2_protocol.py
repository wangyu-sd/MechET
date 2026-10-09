"""Contracts for paper Stage III: v2 history, real first divergence, private labels."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
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
from scripts.run_earho_v2 import _attach_decisions
from scripts.audit_reliable_product_mapping_parity import audit
from scripts.natural_language_anchor_branch_stage import (
    AsyncVLLMBridge, _collect_initialized, _messages, _node, _render_prompt,
    _score_rollout, _v2_probe,
    _v2_successor_fingerprint,
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


def test_async_collector_matches_sync_on_executable_toy_reactions(tmp_path):
    class Tokenizer:
        eos_token_id = 0

        def convert_tokens_to_ids(self, token):
            return 0

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                tools=None, enable_thinking=False):
            rendered = "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                for message in messages
            )
            return rendered + ("<|im_start|>assistant\n" if add_generation_prompt else "")

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(token) for token in ids if token != 0)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": self.encode(text)}

    tokenizer = Tokenizer()

    def generated(prompt, params):
        rendered = tokenizer.decode(prompt["prompt_token_ids"])
        if "accepted_actions: 1" in rendered:
            action = '<tool_call>{"name":"finish_trace","arguments":{}}</tool_call>'
        else:
            action = ('<tool_call>{"name":"import_fragments","arguments":'
                      '{"fragments":[{"smiles":"[Na+]","count":1,'
                      '"purpose":"endpoint_context"}]}}</tool_call>')
        ids = tokenizer.encode(action) + [0]
        output = SimpleNamespace(
            token_ids=ids,
            logprobs=[{token: SimpleNamespace(logprob=-0.1)} for token in ids],
            finish_reason="stop",
        )
        return SimpleNamespace(finished=True, outputs=[output for _ in range(params.n)])

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.seed = None

    class LoRARequest:
        def __init__(self, *args):
            self.args = args

    class SyncEngine:
        def generate(self, prompts, params, **kwargs):
            return [generated(prompt, params) for prompt in prompts]

    class AsyncEngine:
        def __init__(self):
            self.active = 0
            self.peak = 0

        async def generate(self, prompt, params, *, request_id, lora_request):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.002)
            self.active -= 1
            yield generated(prompt, params)

    rows = []
    for number in range(4):
        source, decisions = fixture(f"r{number}")
        rows.append(dict(source, earho_v2_reference_decisions=decisions))
    data = tmp_path / "reactions.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def args(output, async_reactions):
        return SimpleNamespace(
            data=str(data), output=str(output), model="fake", adapter=str(tmp_path),
            rank=0, world_size=1, k=2, seed=17, round_index=0, frontier=2,
            full_episode_fraction=1.0, invalid_penalty=0.1,
            wrong_terminal_penalty=0.5, endpoint_similarity_weight=0.45,
            first_successor_progress_weight=0.25, nonexact_reward_ceiling=0.01,
            target_retained_penalty=0.5, reference_first_successor_weight=0.0,
            endpoint_metric="structural", value_adapter=None, value_kind="successor_pn",
            continuation_candidates_per_mode=1, continuation_temperature=0.7,
            value_score_weight=1.0, policy_score_weight=0.1,
            continuation_beam_width=1, success_gated_advantages=True,
            temperature=1.0, max_new_tokens=100, max_context=100000,
            max_decisions=4, max_imports=4, evaluation=True, full_only=True,
            reject_target_retained_finish=False, legacy_dual_prompt=False,
            protocol_v2=True, state_only_observation=False, vnext_credit=None,
            vnext_private_reference_credit=False, async_reactions=async_reactions,
        )

    sync_output = tmp_path / "sync.jsonl"
    _collect_initialized(args(sync_output, 1), SyncEngine(), tokenizer,
                         SamplingParams, LoRARequest)

    async def run_async():
        engine = AsyncEngine()
        bridge = AsyncVLLMBridge(engine, asyncio.get_running_loop(), seed=17, rank=0)
        async_output = tmp_path / "async.jsonl"
        await asyncio.to_thread(
            _collect_initialized, args(async_output, 4), bridge, tokenizer,
            SamplingParams, LoRARequest,
        )
        return engine, async_output

    engine, async_output = asyncio.run(run_async())
    sync_rows = [json.loads(line) for line in sync_output.read_text().splitlines()]
    async_rows = [json.loads(line) for line in async_output.read_text().splitlines()]
    assert engine.peak > 1
    assert len(sync_rows) == len(async_rows) == 8
    assert all(row["kind"] == "rl" and row["score"]["formal_execute"]
               and row["score"]["correct"] for row in async_rows)
    assert sorted(sync_rows, key=lambda row: (row["id"], row["candidate_index"])) == sorted(
        async_rows, key=lambda row: (row["id"], row["candidate_index"])
    )
    assert not (tmp_path / "sync.jsonl.errors.jsonl").read_text()
    assert not (tmp_path / "async.jsonl.errors.jsonl").read_text()


def test_product_only_mapping_audit_accepts_equivalent_toy_replay(tmp_path):
    import rdkit

    source, decisions = fixture()
    source_path = tmp_path / "source.jsonl"
    decision_path = tmp_path / "decisions.jsonl"
    source_path.write_text(json.dumps(source) + "\n")
    decision_path.write_text("".join(json.dumps(row) + "\n" for row in decisions))
    report = audit(source_path, decision_path, n=1, seed=17, max_imports=1)
    assert report["rdkit_version"] == rdkit.__version__
    assert report["max_imports"] == 1
    assert report["counts"]["root_prompt_exact"] == 1
    assert report["counts"]["original_private_map_replay_ok"] == 1
    assert report["counts"]["product_only_remap_replay_ok"] == 1
    assert report["failures"] == []
    with pytest.raises(ValueError, match="max_imports must be positive"):
        audit(source_path, decision_path, n=1, seed=17, max_imports=0)


def test_product_only_mapping_audit_supports_state_only_decisions(tmp_path):
    source, history_decisions = fixture()
    reference = replay_reference(source, history_decisions)
    state_decisions = []
    for index, row in enumerate(history_decisions):
        copy = dict(row, messages=[dict(message) for message in row["messages"]])
        copy["messages"][1]["content"] = policy_prompt(
            reference.target, reference.nodes[index].state,
            include_inventory=True, actions=reference.nodes[index].actions,
            compact_history=False,
        )
        state_decisions.append(copy)
    source_path = tmp_path / "source.jsonl"
    decision_path = tmp_path / "decisions.jsonl"
    source_path.write_text(json.dumps(source) + "\n")
    decision_path.write_text("".join(json.dumps(row) + "\n" for row in state_decisions))
    report = audit(
        source_path, decision_path, n=1, seed=17, compact_history=False,
    )
    assert report["observation_contract"] == "state_only"
    assert report["counts"]["product_only_remap_replay_ok"] == 1


def test_state_direct_earho_prompt_matches_stage_i_observation():
    source, history_decisions = fixture()
    reference = replay_reference(source, history_decisions)
    state_decisions = []
    for index, row in enumerate(history_decisions):
        copy = dict(row, messages=[dict(message) for message in row["messages"]])
        copy["messages"][1]["content"] = policy_prompt(
            reference.target, reference.nodes[index].state,
            include_inventory=True, actions=reference.nodes[index].actions,
            compact_history=False,
        )
        state_decisions.append(copy)
    state_reference = replay_reference(source, state_decisions, compact_history=False)
    task = anchor_task(state_reference, 1, divergence_reason="EXECUTED_SUCCESSOR_DIVERGED")
    actor_prompt = _messages(task, "unified", state_only=True)[1]["content"]
    assert actor_prompt == state_decisions[1]["messages"][1]["content"]
    assert "accepted_actions:" not in actor_prompt


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


def test_earho_preparation_replays_reference_under_the_rollout_import_budget(tmp_path):
    source, decisions = fixture()
    history_file = tmp_path / "history.jsonl"
    history_file.write_text("".join(json.dumps(row) + "\n" for row in decisions))
    assert len(_attach_decisions([source], history_file, max_imports=1)) == 1
    with pytest.raises(ValueError, match="IMPORT_BUDGET_EXCEEDED"):
        _attach_decisions([source], history_file, max_imports=0)


def test_reliable_endpoint_reward_uses_structural_precursor_not_context():
    source, decisions = fixture()
    reference = replay_reference(source, decisions)
    task = anchor_task(reference, 0, divergence_reason="PRODUCT_ONLY_EVALUATION")
    assert task.expected_structural_precursor == "[CH3:1][Br:2]"
    # This toy is for endpoint projection only; a production rollout separately
    # applies the executor's no-op/target-retained finish gate.
    alternate = replace(
        reference.nodes[-1], state="[CH3:1][Br:2].[K+:4]",
    )
    kwargs = dict(
        first_successor_state=alternate.state,
        invalid_penalty=0.1,
        wrong_terminal_penalty=0.5,
        endpoint_similarity_weight=0.45,
        first_successor_progress_weight=0.25,
        nonexact_reward_ceiling=0.01,
        target_retained_penalty=0.75,
        reference_first_successor_state=reference.nodes[1].state,
        reference_first_successor_weight=0.25,
    )
    structural = _score_rollout(
        task, alternate, "", 2, endpoint_metric="structural", **kwargs,
    )
    full = _score_rollout(task, alternate, "", 2, **kwargs)
    assert structural["correct"] is True
    assert structural["structural_endpoint_exact"] is True
    assert structural["full_endpoint_exact"] is False
    assert structural["reward"] == 1.0
    assert structural["structural_precursor_smiles"] == "CBr"
    assert full["correct"] is False
    assert full["reward"] < 0


def test_v2_collector_uses_completed_sft_tool_prefix_not_thinking_prompt():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                tools=None, enable_thinking=False):
            text = "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                for message in messages
            )
            if add_generation_prompt:
                text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            return text

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    source, decisions = fixture()
    reference = replay_reference(source, decisions)
    task = anchor_task(reference, 0, divergence_reason="PRODUCT_ONLY_EVALUATION")
    encoded = _render_prompt(Tokenizer(), task, task.anchor_state, "unified")
    text = "".join(map(chr, encoded))
    assert text.endswith("<|im_start|>assistant\n")
    assert "<think>" not in text


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
    def prompt(tokenizer, task, state, mode, *, actions, state_only=False):
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
