from mechet.grounded_event_smoke import (
    event_descriptor,
    render_grounded_event_prompt,
    state_with_imports,
    unmap_state,
)
from mechet.inverse_trace_data import invert_moves


def _sn2_forward_moves():
    return [
        {
            "source": {"kind": "LP", "atoms": [2]},
            "sink": {"kind": "BOND", "atoms": [1, 2]},
        },
        {
            "source": {"kind": "BOND", "atoms": [1, 3]},
            "sink": {"kind": "ATOM", "atoms": [3]},
        },
    ]


def test_materializes_first_use_imports_without_exposing_maps() -> None:
    step = {
        "state_before": "[CH3:1][OH:2]",
        "state_after": "[CH3:1][Br:3].[OH-:2]",
        "imports": ["[Br-:3]"],
        "moves": invert_moves(_sn2_forward_moves()),
    }
    state = state_with_imports(step)
    assert ":1]" in state and ":2]" in state and ":3]" in state
    visible = unmap_state(state)
    assert ":1]" not in visible and ":2]" not in visible and ":3]" not in visible


def test_event_description_preserves_directed_electron_flow_without_map_ids() -> None:
    state = "[Br-:3].[CH3:1][OH:2]"
    moves = invert_moves(_sn2_forward_moves())
    text = event_descriptor(state, moves)
    assert "move 1:" in text
    assert "move 2:" in text
    assert "->" in text
    assert "LP on Br" in text
    assert "BOND pair" in text
    assert ":1]" not in text and ":2]" not in text and ":3]" not in text


def test_prompt_contains_only_randomized_event_labels_not_private_moves() -> None:
    task = {
        "target_product": "CO",
        "current_state": "CO.[Br-]",
        "history": [],
        "options": [
            {"label": "A", "descriptor": "move 1: LP on Br -> BOND pair [C] --(unbonded)-- [Br]"},
            {"label": "B", "descriptor": "move 1: BOND pair [C] --(single)-- [O] -> ATOM on O"},
        ],
    }
    system, user = render_grounded_event_prompt(task)
    assert "Return only the option label" in system
    assert "[A]" in user and "[B]" in user
    assert "atom-map numbers" in system
    assert ":137]" not in user
