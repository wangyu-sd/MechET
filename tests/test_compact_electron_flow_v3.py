"""Compact electron-flow serialization and executor-parity tests."""
from __future__ import annotations

import pytest

from mechet.a7_rescue import canonical_event
from mechet.compact_electron_flow import (
    compact_from_natural_arguments,
    natural_from_compact_arguments,
    render_compact_event_arguments,
)
from mechet.natural_language_electron_flow import (
    compile_event_arguments, execute_event_arguments, render_event_arguments,
)

STATE = "[O:1]=[C:2]([OH:3])[CH3:4].[O-:5][CH2:6][CH3:7]"
MOVES = [
    {"source": {"kind": "BOND", "atoms": [1, 2]},
     "sink": {"kind": "ATOM", "atoms": [1]}, "electrons": 2},
    {"source": {"kind": "LP", "atoms": [5]},
     "sink": {"kind": "BOND", "atoms": [2, 5]}, "electrons": 2},
]


def test_source_sink_parity_and_shorter_tool_output():
    old = render_event_arguments(STATE, MOVES)
    new = compact_from_natural_arguments(old)
    assert "LP(" in new["flow"] and "B(" in new["flow"]
    assert len(str(new)) < len(str(old))
    assert natural_from_compact_arguments(new)["electron_flow"] == old["electron_flow"]
    assert canonical_event(compile_event_arguments(STATE, new)) == canonical_event(MOVES)
    assert execute_event_arguments(STATE, new) == execute_event_arguments(STATE, old)
    assert render_compact_event_arguments(STATE, MOVES) == new


def test_graph_delta_parity():
    state = "[CH2:1]=[CH:2][CH3:3]"
    moves = [{
        "mode": "BE_DELTA",
        "bond_deltas": [{"atoms": [1, 2], "delta": -1},
                        {"atoms": [2, 3], "delta": 1}],
        "charge_actions": [],
    }]
    compact = render_compact_event_arguments(state, moves)
    assert compact["flow"].startswith("DELTA|")
    assert canonical_event(compile_event_arguments(state, compact)) == canonical_event(moves)


def test_strict_fail_closed_payload():
    for payload in [
        {"flow": ""},
        {"flow": "B(A01,A02)>A01", "electron_flow": []},
        {"flow": "LP(A01)>LP(A02)"},
        {"flow": "B(A01,A02)>A02;__import__(os)"},
        {"flow": "DELTA|Q(A02):zero>+1"},
    ]:
        with pytest.raises(ValueError):
            compile_event_arguments(STATE, payload)
    with pytest.raises((KeyError, ValueError)):
        compile_event_arguments(STATE, {"flow": "B(A999,A01)>A999"})
