import json
import re

import pytest

from mechet.a7_rescue import canonical_event, canonical_mapped_state
from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import (
    compile_flow,
    convert_trace_row,
    deterministic_unmapped_state,
    encode_grounded_event,
)


def substitution_moves():
    return [
        {
            "source": {"kind": "LP", "atoms": [1]},
            "sink": {"kind": "BOND", "atoms": [1, 2]},
            "electrons": 2,
        },
        {
            "source": {"kind": "BOND", "atoms": [2, 3]},
            "sink": {"kind": "ATOM", "atoms": [3]},
            "electrons": 2,
        },
    ]


def test_insertion_only_event_round_trip_and_replay():
    state = "[O-:1].[CH3:2][Br:3]"
    event = encode_grounded_event(state, substitution_moves())
    assert event.flow == "C>AC ; AB>B"
    assert re.sub(r"<[A-Z]>", "", event.marked_state) == deterministic_unmapped_state(
        state
    ).text
    compiled = compile_flow(
        mapped_state=state,
        marked_state=event.marked_state,
        flow=event.flow,
    )
    assert canonical_event(compiled) == canonical_event(substitution_moves())
    original = verify_electron_step(state, substitution_moves())
    replay = verify_electron_step(state, compiled)
    assert original["ok"] and replay["ok"]
    assert canonical_mapped_state(original["state_smiles"]) == canonical_mapped_state(
        replay["state_smiles"]
    )


def test_delta_event_round_trip():
    state = "[O-:1][CH:2]=[CH2:3]"
    moves = [
        {
            "mode": "BE_DELTA",
            "bond_deltas": [
                {"atoms": [1, 2], "delta": 1},
                {"atoms": [2, 3], "delta": -1},
            ],
            "charge_actions": [
                {"atom_map": 1, "q0": -1, "q1": 0},
                {"atom_map": 3, "q0": 0, "q1": -1},
            ],
        }
    ]
    event = encode_grounded_event(state, moves)
    assert event.flow.startswith("DELTA ")
    assert canonical_event(event.compiled_moves) == canonical_event(moves)


def test_marker_must_be_an_insertion_at_an_atom_boundary():
    state = "[O-:1].[CH3:2][Br:3]"
    event = encode_grounded_event(state, substitution_moves())
    with pytest.raises(ValueError, match="insertion-only|immediately before"):
        compile_flow(
            mapped_state=state,
            marked_state=event.marked_state.replace("Br", "Cl"),
            flow=event.flow,
        )


def test_complete_trace_conversion_hides_maps_and_finishes_from_executor():
    target = "[O-:1].[CH3:2][Br:3]"
    replay = verify_electron_step(target, substitution_moves())
    assert replay["ok"]
    row = {
        "id": "synthetic-sn2",
        "source_id": "synthetic-sn2",
        "target_smiles": target,
        "full_precursor_state": replay["state_smiles"],
        "messages": [],
        "metadata": {
            "endpoint_source": "environment_owned_trace",
            "executor_replayed": True,
            "trace_digest": "synthetic",
            "trace_plan": {
                "steps": [
                    {
                        "state_before": target,
                        "state_after": replay["state_smiles"],
                        "moves": substitution_moves(),
                    }
                ]
            },
        },
    }
    converted = convert_trace_row(row)
    assert converted["metadata"]["n_events"] == 1
    visible = json.dumps(
        {"messages": converted["messages"], "tools": converted["tools"]}
    )
    assert not re.search(r":\d+\]", visible)
    names = [
        call["function"]["name"]
        for message in converted["messages"]
        for call in message.get("tool_calls") or []
    ]
    assert names == ["apply_grounded_event", "finish_trace"]
