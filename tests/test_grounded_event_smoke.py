from mechet.grounded_event_smoke import (
    executable_event_candidates,
    event_descriptor,
    render_forward_event_prompt,
    render_grounded_event_prompt,
    reverse_event_moves,
    state_with_imports,
    unmap_state,
)
from mechet.forward_expert import verify_electron_step
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


def test_candidate_generation_handles_containers_on_the_gold_center() -> None:
    state = "[Br-:3].[CH3:1][OH:2]"
    candidates, gold_successor = executable_event_candidates(
        state=state,
        gold_moves=invert_moves(_sn2_forward_moves()),
        max_candidates=4,
    )
    assert candidates[0]["is_gold"]
    assert candidates[0]["successor"] == gold_successor


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


def test_reverse_event_round_trips_and_renders_forward_choice() -> None:
    state = "[Br-:3].[CH3:1][OH:2]"
    inverse = invert_moves(_sn2_forward_moves())
    result = verify_electron_step(state, inverse)
    assert result["ok"]
    successor = result["state_smiles"]
    round_trip = verify_electron_step(successor, reverse_event_moves(inverse))
    assert round_trip["ok"]
    assert unmap_state(round_trip["state_smiles"]) == unmap_state(state)

    task = {
        "target_product": "CO",
        "current_state": unmap_state(state),
        "options": [
            {"label": "A", "moves": inverse, "successor": successor},
        ],
    }
    system, user = render_forward_event_prompt(task)
    assert "forward chemistry" in system
    assert "CANDIDATE STARTING STATE" in user
    assert "[A]" in user
