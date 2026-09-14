from __future__ import annotations

import json
from pathlib import Path

from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    mapped_state_signature,
)
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.eval_natural_language_event_suffix import reference_episode, select_reactions


ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl"


def _rows() -> list[dict]:
    if not VALID.is_file():
        return []
    with VALID.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_suffix_selection_is_deterministic_and_eligible() -> None:
    rows = _rows()
    if not rows:
        return
    left = select_reactions(rows, size=4, seed=17, minimum_events=3)
    right = select_reactions(reversed(rows), size=4, seed=17, minimum_events=3)
    assert [row["id"] for row in left] == [row["id"] for row in right]
    assert all(len(row["metadata"]["trace_plan"]["steps"]) >= 3 for row in left)


def test_reference_suffix_gold_actions_reach_endpoint() -> None:
    rows = _rows()
    if not rows:
        return
    for row in select_reactions(rows, size=2, seed=17, minimum_events=3):
        for horizon in (1, 2, 3, None):
            episode = reference_episode(row, horizon)
            current = episode["start_state"]
            for reference in episode["events"]:
                current = append_mapped_fragments_verbatim(current, reference["imports"])
                assert mapped_state_signature(current) == mapped_state_signature(
                    reference["event_state"]
                )
                moves = compile_event_arguments(current, reference["gold_arguments"])
                replay = verify_electron_step(current, moves)
                assert replay["ok"]
                current = replay["state_smiles"]
                assert mapped_state_signature(current) == mapped_state_signature(
                    reference["reference_successor"]
                )
            assert deterministic_unmapped_state(current).text == episode["expected_precursor"]

