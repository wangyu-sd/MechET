from mechet.grounded_event_search import (
    GroundedProposal,
    advance_grounded_beam,
    make_root,
)
from mechet.forward_expert import verify_electron_step


TARGET = "[O-:1].[CH3:2][Br:3]"
MOVES = [
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


def row():
    endpoint = verify_electron_step(TARGET, MOVES)["state_smiles"]
    return {
        "target_smiles": TARGET,
        "expected_precursor": endpoint,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "TARGET: [O-].CBr"},
        ],
    }


def event(score=-4.0):
    return GroundedProposal(
        name="apply_grounded_event",
        arguments={
            "imports": [],
            "marked_state": "<A>[O-].<B>C<C>Br",
            "flow": "A>AB ; BC>C",
        },
        raw_response="event",
        logprob_sum=score,
        token_count=4,
        seed=17,
    )


def finish(score=-1.0):
    return GroundedProposal(
        name="finish_trace",
        arguments={},
        raw_response="finish",
        logprob_sum=score,
        token_count=1,
        seed=18,
    )


def test_event_then_explicit_finish_derives_exact_endpoint():
    value = row()
    first = advance_grounded_beam([(make_root(value), [event()])], beam_width=1)
    assert len(first.selected) == 1
    assert not first.rejected

    second = advance_grounded_beam(
        [(first.selected[0], [finish()])],
        beam_width=1,
        expected_precursor=value["expected_precursor"],
    )
    assert not second.selected
    assert len(second.terminals) == 1
    assert second.terminals[0].result["formal_execute"] is True
    assert second.terminals[0].result["endpoint_exact"] is True
    assert second.terminals[0].result["endpoint_source"] == "environment_owned_trace"


def test_finish_without_a_committed_event_is_rejected():
    result = advance_grounded_beam(
        [(make_root(row()), [finish()])], beam_width=1
    )
    assert not result.terminals
    assert result.rejected[0].code == "TRACE_COMPILATION_FAILED"


def test_invalid_event_is_pruned_without_mutating_parent():
    root = make_root(row())
    invalid = GroundedProposal(
        name="apply_grounded_event",
        arguments={
            "imports": ["[Na+]"],
            "marked_state": "not a molecule",
            "flow": "A>AB",
        },
        raw_response="invalid",
        logprob_sum=-0.1,
        token_count=1,
        seed=17,
    )
    result = advance_grounded_beam([(root, [invalid])], beam_width=1)
    assert not result.selected
    assert result.rejected
    assert root.current_mapped_state == TARGET
    assert root.next_private_map == 4


def test_equivalent_successors_share_one_beam_slot():
    root = make_root(row())
    reordered = GroundedProposal(
        name="apply_grounded_event",
        arguments={
            "imports": [],
            "marked_state": "<A>[O-].<B>C<C>Br",
            "flow": "BC>C ; A>AB",
        },
        raw_response="same event",
        logprob_sum=-5.0,
        token_count=4,
        seed=19,
    )
    result = advance_grounded_beam(
        [(root, [event(), reordered])], beam_width=2
    )
    assert len(result.selected) == 1
    assert len(result.duplicate_pruned) == 1
