"""R3 cohort contracts: atomic corruption and honest stratum accounting."""

from __future__ import annotations

import pytest

from scripts.autoresearch.build_r3_corruptions import (
    _replacement_arguments,
    coordination,
    depth_positions,
    event_coordination,
    in_closed_shell_two_electron_scope,
    public_executor_result,
    states_closed_shell,
)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [([1, 1, 1], "one"), ([1, 2, 1], "two"), ([1, 3, 2], "three_plus")],
)
def test_coordination_is_reaction_maximum(counts, expected):
    row = {"metadata": {"trace_plan": {"steps": [
        {"moves": [{} for _ in range(count)]} for count in counts
    ]}}}
    assert coordination(row) == expected


def test_depths_are_distinct_electron_event_decisions():
    assert depth_positions([0, 2, 4, 6]) == {
        "early": 0, "middle": 4, "late": 6,
    }
    with pytest.raises(ValueError, match="three distinct"):
        depth_positions([0, 2])


def test_one_move_polar_event_is_not_confused_with_radical_reaction_maximum():
    polar = {"source": {"kind": "LP", "atoms": [1]},
             "sink": {"kind": "BOND", "atoms": [1, 2]}, "electrons": 2}
    other = {"source": {"kind": "BOND", "atoms": [2, 3]},
             "sink": {"kind": "ATOM", "atoms": [3]}, "electrons": 2}
    row = {"metadata": {"trace_plan": {"steps": [
        {"moves": [polar]}, {"moves": [polar, other]}, {"moves": [polar]},
    ]}}}
    assert coordination(row) == "two"
    assert event_coordination(row["metadata"]["trace_plan"]["steps"][0]) == "one"
    assert in_closed_shell_two_electron_scope(row)
    radical = {**polar, "source": {"kind": "RADICAL_PAIR", "atoms": [1, 2]}}
    row["metadata"]["trace_plan"]["steps"][1]["moves"].append(radical)
    assert not in_closed_shell_two_electron_scope(row)


def test_mutation_changes_one_move_without_touching_reference():
    arguments = {"direction": "retrosynthetic", "electron_flow": [
        {"source": "a lone pair on atom A01",
         "destination": "the bond to form between atoms A01 and A02",
         "instruction": "Move from A01 to bond A01 and A02"},
        {"source": "the bond between atoms A03 and A04",
         "destination": "atom A04"},
    ]}
    changed = _replacement_arguments(arguments, "destination", "A02", "A05", 0)
    assert changed["electron_flow"][0]["destination"].endswith("A05")
    assert changed["electron_flow"][1] == arguments["electron_flow"][1]
    assert arguments["electron_flow"][0]["destination"].endswith("A02")


def test_public_feedback_does_not_reveal_reference_relative_failure_label():
    result = public_executor_result({
        "failure_kind": "accepted_wrong_successor",
        "executor_error": None, "observed_successor": "CC.O",
    })
    assert result == {"ok": True, "code": "PASS", "current_state": "CC.O",
                      "error": None}
    assert "wrong" not in str(result)


def test_state_radical_gate_catches_open_shell_even_without_radical_move_kind():
    row = {"target_smiles": "[CH3:1][OH:2]", "metadata": {"trace_plan": {"steps": [
        {"state_before": "[CH3:1][OH:2]", "state_after": "[CH3:1].[OH:2]",
         "moves": [{"source": {"kind": "BOND"}, "sink": {"kind": "ATOM"},
                    "electrons": 2}]},
    ]}}}
    assert in_closed_shell_two_electron_scope(row)
    assert not states_closed_shell(row)
