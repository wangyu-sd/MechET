from types import SimpleNamespace

from mechet.natural_language_anchor_branch_rl import (
    assign_local_advantages,
    endpoint_potential,
    endpoint_shaped_reward,
    contains_unchanged_target,
    state_value_margin,
    successor_fingerprint,
    task_from_episode,
)
from scripts.natural_language_anchor_branch_stage import _advance, _beam_continue, _node
from scripts.run_natural_language_value_search import Action, Node, execute, visible


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
