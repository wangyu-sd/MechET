#!/usr/bin/env python3
"""Run deterministic small-case checks for executor-constrained event search."""
from __future__ import annotations

import json

from mechet.constrained_event_search import EventProposal, SearchNode, advance_event_beam


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


def main() -> int:
    root = SearchNode(INITIAL)
    first = advance_event_beam(
        [
            (
                root,
                [
                    # Deliberately give the invalid proposal the highest model
                    # score: executor feasibility must dominate ranking.
                    EventProposal(
                        "invalid_high_score",
                        INVALID_NONEXISTENT_BOND,
                        logprob_sum=-0.2,
                        token_count=1,
                    ),
                    EventProposal(
                        "substitution",
                        SUBSTITUTION,
                        logprob_sum=-4.0,
                        token_count=4,
                    ),
                    # Same chemistry, different move serialization.  The search
                    # layer should spend only one beam slot on the successor.
                    EventProposal(
                        "substitution_reordered",
                        tuple(reversed(SUBSTITUTION)),
                        logprob_sum=-5.0,
                        token_count=4,
                    ),
                    # Formally executable but less likely partial chemistry.
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
    if len(first.selected) != 1:
        raise SystemExit("expected exactly one selected first-step branch")
    survivor = first.selected[0]
    if survivor.trace_labels != ("substitution",):
        raise SystemExit(f"unexpected selected branch: {survivor.trace_labels}")
    if not any(item.code == "CHEMICAL_STATE_INVALID" for item in first.executor_rejected):
        raise SystemExit("invalid branch was not rejected by executor")
    if not any(item.code == "DUPLICATE_SUCCESSOR" for item in first.duplicate_pruned):
        raise SystemExit("duplicate successor was not deduplicated")
    if not any(item.code == "BEAM_CAPACITY" for item in first.beam_pruned):
        raise SystemExit("lower-ranked valid branch was not beam-pruned")

    second = advance_event_beam(
        [
            (
                survivor,
                [
                    EventProposal(
                        "undo_substitution",
                        INVERSE_TO_INITIAL,
                        logprob_sum=-0.1,
                        token_count=1,
                    )
                ],
            )
        ],
        beam_width=1,
    )
    if second.selected:
        raise SystemExit("cycle branch unexpectedly survived")
    if not any(item.code == "STATE_CYCLE" for item in second.executor_rejected):
        raise SystemExit("ancestor-state cycle was not rejected")

    payload = {
        "case": "mapped_SN2_like_event",
        "initial_state": INITIAL,
        "first_layer": first.to_dict(),
        "second_layer": second.to_dict(),
        "checks": {
            "invalid_high_score_pruned_before_ranking": True,
            "duplicate_successor_deduplicated": True,
            "beam_uses_model_likelihood_after_hard_gates": True,
            "ancestor_cycle_pruned_immediately": True,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
