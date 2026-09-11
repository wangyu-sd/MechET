from mechet.constrained_event_search import (
    EventProposal,
    SearchNode,
    advance_event_beam,
)


INITIAL = "[O-:1].[CH3:2][Br:3]"

SUBSTITUTION = (
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
)

CLEAVAGE_ONLY = (
    {
        "source": {"kind": "BOND", "atoms": [2, 3]},
        "sink": {"kind": "ATOM", "atoms": [3]},
        "electrons": 2,
    },
)

INVALID_NONEXISTENT_BOND = (
    {
        "source": {"kind": "BOND", "atoms": [1, 2]},
        "sink": {"kind": "ATOM", "atoms": [1]},
        "electrons": 2,
    },
)

INVERSE_TO_INITIAL = (
    {
        "source": {"kind": "BOND", "atoms": [1, 2]},
        "sink": {"kind": "ATOM", "atoms": [1]},
        "electrons": 2,
    },
    {
        "source": {"kind": "LP", "atoms": [3]},
        "sink": {"kind": "BOND", "atoms": [2, 3]},
        "electrons": 2,
    },
)


def test_executor_gate_beats_model_score():
    root = SearchNode(INITIAL)
    result = advance_event_beam(
        [
            (
                root,
                [
                    EventProposal(
                        "invalid_high_score",
                        INVALID_NONEXISTENT_BOND,
                        logprob_sum=-0.1,
                        token_count=1,
                    ),
                    EventProposal(
                        "valid_lower_score",
                        SUBSTITUTION,
                        logprob_sum=-4.0,
                        token_count=4,
                    ),
                ],
            )
        ],
        beam_width=1,
    )
    assert [node.trace_labels for node in result.selected] == [
        ("valid_lower_score",)
    ]
    assert [item.code for item in result.executor_rejected] == [
        "CHEMICAL_STATE_INVALID"
    ]


def test_duplicate_successor_uses_one_beam_slot():
    root = SearchNode(INITIAL)
    result = advance_event_beam(
        [
            (
                root,
                [
                    EventProposal(
                        "better",
                        SUBSTITUTION,
                        logprob_sum=-4.0,
                        token_count=4,
                    ),
                    EventProposal(
                        "worse_same_successor",
                        tuple(reversed(SUBSTITUTION)),
                        logprob_sum=-5.0,
                        token_count=4,
                    ),
                ],
            )
        ],
        beam_width=2,
    )
    assert len(result.selected) == 1
    assert result.selected[0].trace_labels == ("better",)
    assert len(result.duplicate_pruned) == 1
    assert result.duplicate_pruned[0].proposal_label == "worse_same_successor"
    assert result.duplicate_pruned[0].code == "DUPLICATE_SUCCESSOR"


def test_likelihood_ranks_only_after_hard_filters():
    root = SearchNode(INITIAL)
    result = advance_event_beam(
        [
            (
                root,
                [
                    EventProposal(
                        "substitution",
                        SUBSTITUTION,
                        logprob_sum=-4.0,
                        token_count=4,
                    ),
                    EventProposal(
                        "cleavage_only",
                        CLEAVAGE_ONLY,
                        logprob_sum=-6.0,
                        token_count=4,
                    ),
                ],
            )
        ],
        beam_width=1,
    )
    assert result.selected[0].trace_labels == ("substitution",)
    assert len(result.beam_pruned) == 1
    assert result.beam_pruned[0].proposal_label == "cleavage_only"
    assert result.beam_pruned[0].code == "BEAM_CAPACITY"


def test_ancestor_state_cycle_is_pruned_immediately():
    root = SearchNode(INITIAL)
    first = advance_event_beam(
        [
            (
                root,
                [
                    EventProposal(
                        "substitution",
                        SUBSTITUTION,
                        logprob_sum=-4.0,
                        token_count=4,
                    )
                ],
            )
        ],
        beam_width=1,
    )
    child = first.selected[0]
    second = advance_event_beam(
        [
            (
                child,
                [
                    EventProposal(
                        "undo",
                        INVERSE_TO_INITIAL,
                        logprob_sum=-0.1,
                        token_count=1,
                    )
                ],
            )
        ],
        beam_width=1,
    )
    assert not second.selected
    assert len(second.executor_rejected) == 1
    assert second.executor_rejected[0].code == "STATE_CYCLE"


def test_optional_support_contract_is_a_hard_gate():
    root = SearchNode(INITIAL)

    def reject_charged_successor(smiles: str) -> tuple[bool, str]:
        return ("[Br-" not in smiles, "charged bromide excluded by test contract")

    result = advance_event_beam(
        [
            (
                root,
                [
                    EventProposal(
                        "substitution",
                        SUBSTITUTION,
                        logprob_sum=-4.0,
                        token_count=4,
                    )
                ],
            )
        ],
        beam_width=1,
        state_validator=reject_charged_successor,
    )
    assert not result.selected
    assert result.executor_rejected[0].code == "CHEMISTRY_SUPPORT_REJECTED"
