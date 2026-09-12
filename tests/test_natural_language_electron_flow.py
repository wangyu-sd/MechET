import re

from mechet.a7_rescue import canonical_event
from mechet.natural_language_electron_flow import (
    build_inventory,
    compile_event_arguments,
    execute_event_arguments,
    render_event_arguments,
)


def test_natural_language_event_round_trips_and_executes() -> None:
    state = "[O:1]=[C:2]([OH:3])[CH3:4].[O-:5][CH2:6][CH3:7]"
    moves = [
        {
            "source": {"kind": "BOND", "atoms": [1, 2]},
            "sink": {"kind": "ATOM", "atoms": [1]},
            "electrons": 2,
        },
        {
            "source": {"kind": "LP", "atoms": [5]},
            "sink": {"kind": "BOND", "atoms": [2, 5]},
            "electrons": 2,
        },
    ]
    arguments = render_event_arguments(state, moves)
    assert arguments["direction"] == "retrosynthetic"
    assert len(arguments["electron_flow"]) == 2
    assert "transfer the electron pair" in arguments["electron_flow"][0]["instruction"]
    assert canonical_event(compile_event_arguments(state, arguments)) == canonical_event(moves)
    result = execute_event_arguments(state, arguments)
    assert result["ok"] is True


def test_inventory_is_unmapped_and_covers_every_atom_and_bond() -> None:
    state = "[O:9]=[C:3]([OH:11])[CH3:7]"
    inventory = build_inventory(state)
    assert len(inventory.atom_to_map) == 4
    assert len(inventory.bond_to_maps) == 3
    assert set(inventory.atom_to_map) == {"A01", "A02", "A03", "A04"}
    assert not re.search(r":\d+\]", inventory.visible_smiles)
    assert "ANNOTATED CURRENT STATE:" in inventory.prompt
    assert all(f"<A{index:02d}>" in inventory.prompt for index in range(1, 5))


def test_be_delta_uses_natural_atom_aliases_and_round_trips() -> None:
    state = "[CH2:1]=[CH:2][CH3:3]"
    moves = [
        {
            "mode": "BE_DELTA",
            "bond_deltas": [
                {"atoms": [1, 2], "delta": -1},
                {"atoms": [2, 3], "delta": 1},
            ],
            "charge_actions": [],
        }
    ]
    arguments = render_event_arguments(state, moves)
    assert arguments["electron_flow"] == []
    assert all("bond order" in item["instruction"] for item in arguments["bond_order_changes"])
    assert canonical_event(compile_event_arguments(state, arguments)) == canonical_event(moves)


def test_direction_is_explicit_and_enforced() -> None:
    state = "[O:1]=[C:2]"
    moves = [
        {
            "source": {"kind": "BOND", "atoms": [1, 2]},
            "sink": {"kind": "ATOM", "atoms": [1]},
            "electrons": 2,
        }
    ]
    arguments = render_event_arguments(state, moves)
    arguments["direction"] = "forward"
    try:
        compile_event_arguments(state, arguments)
    except ValueError as exc:
        assert "retrosynthetic" in str(exc)
    else:
        raise AssertionError("forward direction must not pass the inverse event compiler")
