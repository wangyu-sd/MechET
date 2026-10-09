import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from mechet.natural_language_anchor_branch_rl import (
    assign_local_advantages,
    endpoint_potential,
    endpoint_shaped_reward,
    contains_unchanged_target,
    state_value_margin,
    successor_fingerprint,
    task_from_episode,
)
from scripts.natural_language_anchor_branch_stage import (
    AsyncVLLMBridge, _advance, _beam_continue, _beam_continue_many,
    _bounded_results, _first_action_sampling_plan, _node,
)
from scripts.run_natural_language_value_search import Action, Node, execute, visible
from scripts.run_natural_language_anchor_branch_rl import worker_command


def test_validation_k2_keeps_two_sampled_candidates():
    args = SimpleNamespace(k=2, temperature=1.0, evaluation=True,
                           legacy_dual_prompt=False)
    assert _first_action_sampling_plan(args) == (2, 1.0)


def test_training_k8_keeps_eight_sampled_candidates():
    args = SimpleNamespace(k=8, temperature=1.0, evaluation=False,
                           legacy_dual_prompt=False)
    assert _first_action_sampling_plan(args) == (8, 1.0)


def test_multiple_greedy_candidates_are_rejected_before_gpu_start():
    args = SimpleNamespace(k=2, temperature=0.0, evaluation=True,
                           legacy_dual_prompt=False)
    with pytest.raises(ValueError, match="non-greedy"):
        _first_action_sampling_plan(args)


def test_flower_stage3_validation_command_uses_legal_k2_sampling():
    config = Path(__file__).resolve().parents[1] / "configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml"
    cfg = yaml.safe_load(config.read_text())
    command = worker_command(
        cfg, "/data", "/adapter", "/out", 0,
        frontier=2, round_index=-1, evaluation=True,
    )
    assert "--evaluation" in command
    args = SimpleNamespace(
        k=int(command[command.index("--k") + 1]),
        temperature=float(command[command.index("--temperature") + 1]),
        legacy_dual_prompt="--legacy-dual-prompt" in command,
    )
    assert _first_action_sampling_plan(args) == (2, 1.0)


def test_worker_command_enables_async_reactions_only_when_configured():
    config = Path(__file__).resolve().parents[1] / "configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml"
    cfg = yaml.safe_load(config.read_text())
    command = worker_command(cfg, "/data", "/adapter", "/out", 0,
                             frontier=2, round_index=0, evaluation=False)
    assert "--async-reactions" not in command
    cfg["rollout"]["async_reactions"] = 4
    cfg["rollout"]["dtype"] = "float16"
    command = worker_command(cfg, "/data", "/adapter", "/out", 0,
                             frontier=2, round_index=0, evaluation=False)
    assert command[command.index("--async-reactions") + 1] == "4"
    assert command[command.index("--dtype") + 1] == "float16"


def test_worker_command_runtime_async_override_preserves_frozen_config(monkeypatch):
    config = Path(__file__).resolve().parents[1] / "configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml"
    cfg = yaml.safe_load(config.read_text())
    monkeypatch.setenv("MECHET_EARHO_ASYNC_REACTIONS", "4")
    command = worker_command(cfg, "/data", "/adapter", "/out", 0,
                             frontier=2, round_index=0, evaluation=False)
    assert command[command.index("--async-reactions") + 1] == "4"
    assert "async_reactions" not in cfg["rollout"]
    monkeypatch.setenv("MECHET_EARHO_ASYNC_REACTIONS", "33")
    with pytest.raises(ValueError, match="between 1 and 32"):
        worker_command(cfg, "/data", "/adapter", "/out", 0,
                       frontier=2, round_index=0, evaluation=False)


def test_async_bridge_interleaves_reactions_and_preserves_request_outputs():
    class FakeEngine:
        def __init__(self):
            self.active = 0
            self.peak = 0
            self.calls = []

        async def generate(self, prompt, parameters, *, request_id, lora_request):
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append((prompt, parameters.seed, request_id, lora_request))
            await asyncio.sleep(0.01)
            self.active -= 1
            yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(
                token_ids=[prompt["prompt_token_ids"][0]], logprobs=[{1: -0.25}],
            )])

    async def run():
        engine = FakeEngine()
        bridge = AsyncVLLMBridge(engine, asyncio.get_running_loop(), seed=17, rank=0)
        parameters = SimpleNamespace(seed=None, n=2)

        def reaction(reaction_id, prompts):
            bridge.begin_reaction(reaction_id)
            return bridge.generate(
                [{"prompt_token_ids": [prompt]} for prompt in prompts],
                parameters, lora_request="actor",
            )

        first, second = await asyncio.gather(
            asyncio.to_thread(reaction, "r1", [11, 12]),
            asyncio.to_thread(reaction, "r2", [21, 22]),
        )
        assert engine.peak > 1
        assert [row.outputs[0].token_ids for row in first] == [[11], [12]]
        assert [row.outputs[0].token_ids for row in second] == [[21], [22]]
        assert first[0].outputs[0].logprobs == [{1: -0.25}]
        assert len({call[2] for call in engine.calls}) == 4
        assert parameters.seed is None
        saved = {(call[0]["prompt_token_ids"][0], call[1], call[2]) for call in engine.calls}
        await asyncio.to_thread(reaction, "r1", [11, 12])
        assert {(call[0]["prompt_token_ids"][0], call[1], call[2]) for call in engine.calls[-2:]} <= saved

    asyncio.run(run())


def test_bounded_results_processes_each_reaction_once():
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(_bounded_results(pool, lambda value: value * 2, range(19), 3))
    assert sorted(results) == [value * 2 for value in range(19)]


def test_state_direct_worker_keeps_stage_i_prompt_and_structural_reward():
    cfg = {
        "protocol_version": "state_direct_v2", "model_snapshot": "/model",
        "candidates_per_product": 8, "seed": 17, "invalid_penalty": 0.1,
        "reward": {
            "endpoint_metric": "structural", "wrong_terminal_penalty": 0.5,
            "endpoint_similarity_weight": 0.45,
            "first_successor_progress_weight": 0.25,
            "reference_first_successor_weight": 0.25,
            "nonexact_reward_ceiling": 0.01, "target_retained_penalty": 0.75,
        },
        "rollout": {"temperature": 1.0, "max_new_tokens": 384,
                    "max_context": 4096, "max_decisions": 40, "max_imports": 32},
        "curriculum": {"full_episode_fraction": 0.25},
        "optimization": {"success_gated_advantages": True},
    }
    command = worker_command(
        cfg, "/data", "/adapter", "/out", 0,
        frontier=2, round_index=0, evaluation=False,
    )
    assert "--protocol-v2" in command
    assert "--state-only-observation" in command
    assert command[command.index("--endpoint-metric") + 1] == "structural"


def test_task_hides_reference_suffix_and_tracks_reset():
    task = task_from_episode(
        {
            "reaction_id": "train_1",
            "target": "C=O",
            "start_state": "[CH2:1]=[O:2]",
            "expected_precursor": "CO",
            "horizon": 2,
            "total_events": 4,
        }
    )
    assert task.prefix_events == 2
    assert not task.is_full_episode
    assert not hasattr(task, "gold_suffix")


def test_successor_pooling_is_chemical_state_based():
    left = successor_fingerprint(
        prompt_mode="event",
        action_name="apply_electron_flow",
        successor_state="[CH4:1].[OH2:2]",
        terminal=False,
    )
    right = successor_fingerprint(
        prompt_mode="event",
        action_name="apply_electron_flow",
        successor_state="[OH2:2].[CH4:1]",
        terminal=False,
    )
    assert left == right


def test_advantages_are_local_to_prompt_mode():
    records = []
    for mode, rewards in (("action", (1.0, 0.0)), ("event", (0.0, -0.1))):
        for index, reward in enumerate(rewards):
            records.append(
                {
                    "id": "train_1",
                    "anchor": {"state_hash": "same-state"},
                    "prompt_mode": mode,
                    "action_fingerprint": f"{mode}-{index}",
                    "reward": reward,
                    "score": {"correct": reward == 1.0},
                }
            )
    summary = assign_local_advantages(records)
    assert summary["effective"]
    assert records[0]["advantage"] > 0 > records[1]["advantage"]
    assert records[2]["advantage"] > 0 > records[3]["advantage"]


def test_successor_gated_advantages_never_promote_all_negative_groups():
    records = [
        {
            "id": "train_1",
            "anchor": {"state_hash": "same-state"},
            "prompt_mode": "event",
            "action_fingerprint": f"wrong-{index}",
            "reward": reward,
            "score": {
                "correct": False,
                "reference_first_successor_exact": False,
            },
        }
        for index, reward in enumerate((-0.01, -0.2, -0.8))
    ]
    summary = assign_local_advantages(records, success_gated=True)
    assert not summary["effective"]
    assert summary["modes_with_positive"] == 0
    assert all(row["advantage"] == 0.0 for row in records)
    assert not any(row["update_eligible"] for row in records)


def test_successor_gated_advantages_credit_verified_first_successor_only():
    records = []
    for index, exact in enumerate((True, False, False)):
        records.append(
            {
                "id": "train_1",
                "anchor": {"state_hash": "same-state"},
                "prompt_mode": "event",
                "action_fingerprint": f"state-{index}",
                "reward": -0.01 if exact else -0.2,
                "score": {
                    "correct": False,
                    "reference_first_successor_exact": exact,
                },
            }
        )
    summary = assign_local_advantages(records, success_gated=True)
    assert summary["effective"] and summary["successor_success"]
    assert records[0]["advantage"] > 0
    assert records[1]["advantage"] < 0 and records[2]["advantage"] < 0


def test_receding_horizon_beam_falls_back_when_greedy_branch_dies(monkeypatch):
    import scripts.natural_language_anchor_branch_stage as stage

    root_state = "[CH4:1]"
    dead_state = "[CH3:1][CH3:2]"
    fallback_state = "[CH3:1][OH:2]"
    terminal_state = "[CH3:1][NH2:2]"
    root = Node(target="C", state=root_state, next_map=2, visited={"C"})

    class FakeLLM:
        def generate(self, prompts, parameters, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace()]) for _ in prompts]

    def fake_decode(tokenizer, value, eos_ids, mode):
        return {
            "error": "",
            "name": mode,
            "arguments": {},
            "text": mode,
            "logps": [-1.0],
            "ids": [1],
        }

    def fake_advance(node, decoded, max_imports, **kwargs):
        if node.state == root_state:
            state = dead_state if decoded["name"] == "action" else fallback_state
            terminal = False
        elif node.state == dead_state:
            return None, "DEAD_END"
        elif node.state == fallback_state and decoded["name"] == "action":
            state, terminal = terminal_state, True
        else:
            return None, "DEAD_END"
        return Node(
            target=node.target,
            state=state,
            next_map=1,
            actions=node.actions
            + [{"state_before": node.state, "name": decoded["name"]}],
            visited=set(node.visited) | {state},
            logprob=node.logprob - 1.0,
            tokens=node.tokens + 1,
            terminal=terminal,
        ), ""

    def fake_values(llm, tokenizer, value_lora, parameters, task, nodes, **kwargs):
        # The dead CC branch is locally preferred; width two must retain CO.
        values = {dead_state: 2.0, fallback_state: 1.0, terminal_state: 3.0}
        return [values[node.state] for node in nodes]

    monkeypatch.setattr(stage, "_render_prompt", lambda *args, **kwargs: [1])
    monkeypatch.setattr(stage, "_decode_action", fake_decode)
    monkeypatch.setattr(stage, "_advance", fake_advance)
    monkeypatch.setattr(stage, "_critic_scores", fake_values)
    args = SimpleNamespace(
        continuation_beam_width=2,
        max_new_tokens=16,
        max_context=128,
        max_imports=2,
        reject_target_retained_finish=False,
        value_score_weight=1.0,
        policy_score_weight=0.0,
        value_kind="state_abc",
        legacy_dual_prompt=True,
    )
    task = SimpleNamespace(target="C", anchor_state="C")
    result, error = _beam_continue(
        FakeLLM(),
        None,
        None,
        None,
        None,
        None,
        [],
        task,
        root,
        args,
        remaining_decisions=2,
    )
    assert not error
    assert result.terminal and result.state == terminal_state
    assert len(result.actions) == 2


def test_independent_continuations_share_generation_batch(monkeypatch):
    import scripts.natural_language_anchor_branch_stage as stage

    class FakeLLM:
        def __init__(self):
            self.batch_sizes = []

        def generate(self, prompts, parameters, **kwargs):
            self.batch_sizes.append(len(prompts))
            return [
                SimpleNamespace(outputs=[SimpleNamespace(state="N" if prompt["prompt_token_ids"] == [1] else "O")])
                for prompt in prompts
            ]

    def fake_advance(node, decoded, max_imports, **kwargs):
        state = decoded["state"]
        return Node(
            target=node.target, state=state, next_map=1,
            actions=node.actions + [{"name": "finish_trace"}],
            visited=set(node.visited) | {state}, terminal=True,
        ), ""

    monkeypatch.setattr(stage, "_render_prompt", lambda tokenizer, task, state, mode, **kw: [1 if state == "C" else 2])
    monkeypatch.setattr(stage, "_decode_action", lambda tokenizer, value, eos_ids, mode: {"state": value.state})
    monkeypatch.setattr(stage, "_advance", fake_advance)
    args = SimpleNamespace(
        continuation_beam_width=2, max_new_tokens=16, max_context=128,
        max_imports=2, reject_target_retained_finish=False,
        value_score_weight=1.0, policy_score_weight=0.0,
        value_kind="state_abc", legacy_dual_prompt=False,
    )
    llm = FakeLLM()
    starts = [
        Node(target="C", state="C" if index % 2 == 0 else "CC", next_map=1)
        for index in range(8)
    ]
    results = _beam_continue_many(
        llm, None, None, None, None, None, [],
        SimpleNamespace(target="C"), starts, args,
        remaining_decisions=[2] * 8,
    )
    assert llm.batch_sizes == [8]
    assert [(node.state, error) for node, error in results] == [("N", ""), ("O", "")] * 4


def test_endpoint_potential_rewards_closer_structure_and_penalizes_extras():
    expected = "CC(=O)O.CN"
    exact = endpoint_potential(expected, expected)["combined"]
    close = endpoint_potential("CC(=O)O.C", expected)["combined"]
    remote = endpoint_potential("c1ccccc1.[Na+]", expected)["combined"]
    extra = endpoint_potential(expected + ".CCCCCCCC", expected)["combined"]
    assert exact == 1.0
    assert exact > close > remote
    assert exact > extra
    assert endpoint_potential("[CH3:1][OH:2]", "CO")["combined"] == 1.0


def test_shaped_reward_keeps_exact_unique_and_ranks_wrong_endpoints():
    common = {
        "anchor_state": "CCOC",
        "first_successor_state": "CC(=O)O",
        "expected_precursor": "CC(=O)O.CN",
        "invalid_penalty": 0.1,
        "wrong_terminal_penalty": 0.5,
        "endpoint_similarity_weight": 0.45,
        "first_successor_progress_weight": 0.25,
        "nonexact_reward_ceiling": 0.01,
    }
    exact = endpoint_shaped_reward(
        correct=True, terminal=True, final_state="CC(=O)O.CN", **common
    )
    close = endpoint_shaped_reward(
        correct=False, terminal=True, final_state="CC(=O)O.C", **common
    )
    remote = endpoint_shaped_reward(
        correct=False, terminal=True, final_state="c1ccccc1.[Na+]", **common
    )
    invalid = endpoint_shaped_reward(
        correct=False, terminal=False, final_state="", **common
    )
    assert exact["reward"] == 1.0
    assert 0.0 > close["reward"] > remote["reward"]
    assert invalid["reward"] < 0.0


def test_unchanged_target_terminal_is_a_no_transform_failure():
    assert contains_unchanged_target("CCO.[Na+]", "CCO")
    assert not contains_unchanged_target("CC=O.[Na+]", "CCO")
    result = endpoint_shaped_reward(
        correct=False,
        terminal=True,
        anchor_state="CCO",
        first_successor_state="CCO.[Na+]",
        final_state="CCO.[Na+]",
        expected_precursor="CC=O.CN",
        invalid_penalty=0.1,
        wrong_terminal_penalty=0.5,
        endpoint_similarity_weight=0.45,
        first_successor_progress_weight=0.25,
        nonexact_reward_ceiling=0.01,
        target_retained=True,
        target_retained_penalty=0.75,
    )
    assert result["outcome"] == "target_retained_no_transform"
    assert result["reward"] == -0.75


def test_historical_sparse_reward_contract_remains_replayable():
    result = endpoint_shaped_reward(
        correct=False,
        terminal=True,
        anchor_state="CCOC",
        first_successor_state="CC(=O)O",
        final_state="CC(=O)O.C",
        expected_precursor="CC(=O)O.CN",
        invalid_penalty=0.1,
        wrong_terminal_penalty=0.0,
        endpoint_similarity_weight=0.0,
        first_successor_progress_weight=0.0,
        nonexact_reward_ceiling=0.0,
    )
    assert result["reward"] == 0.0


def test_state_value_margin_distinguishes_continue_finish_and_off_path():
    continue_scores = {"A": -0.1, "B": -3.0, "C": -2.0}
    finish_scores = {"A": -3.0, "B": -0.1, "C": -2.0}
    off_path_scores = {"A": -3.0, "B": -2.0, "C": -0.1}
    assert state_value_margin(continue_scores, terminal=False) > 0
    assert state_value_margin(finish_scores, terminal=True) > 0
    assert state_value_margin(off_path_scores, terminal=False) < 0
    assert state_value_margin(off_path_scores, terminal=True) < 0


def _decoded(name, arguments):
    return {
        "error": "",
        "name": name,
        "arguments": arguments,
        "text": name,
        "logps": [],
        "ids": [],
    }


def test_executor_gate_rejects_finish_while_product_is_unchanged():
    task = task_from_episode(
        {
            "reaction_id": "train_gate",
            "target": "CO",
            "start_state": "[CH3:1][OH:2]",
            "expected_precursor": "C=O",
            "horizon": 1,
            "total_events": 1,
        }
    )
    child, error = _advance(
        _node(task),
        _decoded("finish_trace", {}),
        8,
        reject_target_retained_finish=True,
    )
    assert child is None
    assert error == "TARGET_RETAINED_NO_TRANSFORM"


def test_executor_allows_a_repeated_fragment_when_gold_needs_two_batches():
    root = Node(
        target="C",
        state="[CH4:1]",
        next_map=2,
        visited={"C"},
    )
    imported_once, error = execute(
        root,
        Action(
            "import_fragments",
            {
                "fragments": [
                    {"smiles": "[H]Cl", "count": 1, "purpose": "electron_participant"}
                ]
            },
            "",
            0.0,
            1,
        ),
        max_imports=8,
    )
    assert not error and imported_once is not None
    imported_twice, error = execute(
        imported_once,
        Action(
            "import_fragments",
            {
                "fragments": [
                    {"smiles": "[H]Cl", "count": 1, "purpose": "electron_participant"}
                ]
            },
            "",
            0.0,
            1,
        ),
        max_imports=8,
    )
    assert not error and imported_twice is not None
    assert imported_twice.imported["[H]Cl"] == 2


def test_finish_guard_rejects_only_a_true_noop_not_a_transformed_mixture():
    transformed = Node(
        target="C",
        state="[CH4:1].[OH2:2]",
        next_map=3,
        visited={"C", "C.O"},
        actions=[
            {
                "state_before": "[CH4:1]",
                "name": "apply_electron_flow",
                "arguments": {},
                "result": {"ok": True, "code": "PASS", "current_state": "C.O"},
            }
        ],
    )
    terminal, error = execute(
        transformed,
        Action("finish_trace", {}, "", 0.0, 1),
        max_imports=8,
        reject_target_retained_finish=True,
    )
    assert not error and terminal is not None and terminal.terminal
    assert visible(terminal.state) == "C.O"


def test_reference_first_successor_credit_remains_below_exact_reward():
    common = {
        "correct": False,
        "terminal": False,
        "anchor_state": "CCO",
        "first_successor_state": "CC=O",
        "final_state": "CC=O",
        "expected_precursor": "CC=O.CN",
        "invalid_penalty": 0.5,
        "wrong_terminal_penalty": 0.5,
        "endpoint_similarity_weight": 0.45,
        "first_successor_progress_weight": 0.25,
        "nonexact_reward_ceiling": 0.01,
        "reference_first_successor_weight": 0.25,
    }
    matched = endpoint_shaped_reward(
        reference_first_successor_exact=True, **common
    )["reward"]
    unmatched = endpoint_shaped_reward(
        reference_first_successor_exact=False, **common
    )["reward"]
    assert 0 > matched > unmatched
from types import SimpleNamespace
