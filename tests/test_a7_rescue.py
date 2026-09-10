import copy

from mechet.a7_rescue import (
    assistant_event_targets,
    assistant_events,
    audit_gold_event,
    audit_gold_row,
    mechanism_length_stratum,
    local_prediction_metrics,
    stratified_sample,
)


MOVES = [
    {
        "source": {"kind": "BOND", "atoms": [1, 2]},
        "sink": {"kind": "ATOM", "atoms": [2]},
        "electrons": 2,
    },
    {
        "source": {"kind": "LP", "atoms": [3]},
        "sink": {"kind": "BOND", "atoms": [1, 3]},
        "electrons": 2,
    },
]


def row(identifier, n_events):
    step = {
        "state_before": "[CH3:1][OH:2].[Br-:3]",
        "state_after": "[CH3:1][Br:3].[OH-:2]",
        "moves": copy.deepcopy(MOVES),
    }
    messages = []
    for index in range(n_events):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": "apply_coupled_electron_moves",
                            "arguments": {"moves": copy.deepcopy(MOVES)},
                        },
                    }
                ],
            }
        )
    return {
        "id": identifier,
        "metadata": {"trace_plan": {"steps": [copy.deepcopy(step) for _ in range(n_events)]}},
        "messages": messages,
    }


def test_mechanism_length_strata():
    assert [mechanism_length_stratum(value) for value in (1, 2, 3, 4, 5)] == [
        "short",
        "short",
        "medium",
        "medium",
        "long",
    ]


def test_stratified_sample_is_balanced_and_deterministic():
    rows = [row(f"s-{i}", 1) for i in range(5)]
    rows += [row(f"m-{i}", 3) for i in range(5)]
    rows += [row(f"l-{i}", 5) for i in range(5)]
    first = stratified_sample(rows, size=8, seed=17)
    second = stratified_sample(reversed(rows), size=8, seed=17)
    assert [item["id"] for item in first] == [item["id"] for item in second]
    counts = {
        name: sum(mechanism_length_stratum(len(item["metadata"]["trace_plan"]["steps"])) == name for item in first)
        for name in ("short", "medium", "long")
    }
    assert counts == {"short": 3, "medium": 3, "long": 2}


def test_gold_event_replays_and_has_legal_containers():
    result = audit_gold_event(row("one", 1)["metadata"]["trace_plan"]["steps"][0])
    assert result["gold_legal"]
    assert result["successor_exact"]
    assert result["candidate_source_containers_covered"] == 2
    assert result["candidate_sink_containers_covered"] == 2


def test_row_checks_supervised_event_alignment():
    value = row("one", 1)
    assert assistant_events(value) == [MOVES]
    assert audit_gold_row(value)["gold_legal"]
    value["messages"][0]["tool_calls"][0]["function"]["arguments"]["moves"][0]["sink"]["atoms"] = [1]
    result = audit_gold_row(value)
    assert not result["supervision_aligned"]
    assert not result["gold_legal"]


def test_event_targets_do_not_include_the_gold_message_in_prefix():
    value = row("one", 1)
    message_index, moves = assistant_event_targets(value)[0]
    assert message_index == 0
    assert moves == MOVES
    assert value["messages"][:message_index] == []


def test_local_prediction_metrics_accepts_exact_event():
    step = row("one", 1)["metadata"]["trace_plan"]["steps"][0]
    result = local_prediction_metrics(
        state_before=step["state_before"],
        state_after=step["state_after"],
        gold_moves=MOVES,
        predicted_name="apply_coupled_electron_moves",
        predicted_arguments={"moves": MOVES},
    )
    assert result["event_exact"]
    assert result["formal_execute"]
    assert result["successor_exact"]


def test_local_prediction_metrics_rejects_independent_answer():
    step = row("one", 1)["metadata"]["trace_plan"]["steps"][0]
    result = local_prediction_metrics(
        state_before=step["state_before"],
        state_after=step["state_after"],
        gold_moves=MOVES,
        predicted_name="finish_trace",
        predicted_arguments={},
    )
    assert not result["well_formed_event"]
    assert not result["formal_execute"]
