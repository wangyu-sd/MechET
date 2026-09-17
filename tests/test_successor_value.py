import hashlib
import json

import scripts.build_successor_value_from_rollouts as builder
from mechet.successor_value import (
    successor_value_margin,
    successor_value_prompt,
    successor_value_row,
)


def test_successor_value_prompt_is_map_free_and_endpoint_free():
    prompt = successor_value_prompt(
        "[CH3:1][OH:2]",
        "[CH3:1][OH:2]",
        "[CH3:1].[OH:2]",
        terminal=False,
    )
    assert ":1" not in prompt and ":2" not in prompt
    assert "EXPECTED" not in prompt and "PRECURSOR" not in prompt
    assert successor_value_margin({"P": -0.1, "N": -2.0}) > 0


def test_successor_value_row_has_binary_transition_contract():
    row = successor_value_row(
        reaction_id="rxn",
        state_hash="abc",
        target="CO",
        current_state="CO",
        successor_state="C.O",
        terminal=False,
        label="P",
        provenance="reference",
    )
    assert row["messages"][-1]["content"] == "P"
    assert row["metadata"]["executor_successor"]
    assert not row["metadata"]["reference_endpoint_model_visible"]


def test_rollout_mining_pairs_reference_with_actor_hard_negative(monkeypatch):
    start = "[CH3:1][OH:2]"
    episode = {
        "reaction_id": "rxn",
        "target": start,
        "start_state": start,
        "expected_precursor": "[CH3:1].[OH:2]",
        "horizon": 1,
        "total_events": 1,
        "events": [
            {
                "imports": [],
                "reference_successor": "[CH3:1].[OH:2]",
                "event_state": start,
            }
        ],
    }
    monkeypatch.setattr(builder, "reference_episode", lambda row, horizon: episode)
    state_hash = hashlib.sha256(start.encode()).hexdigest()
    groups = {
        ("rxn", state_hash): [
            {
                "id": "rxn",
                "kind": "rl",
                "anchor": {"state_hash": state_hash, "horizon": 1},
                "reward": -0.2,
                "terminated": False,
                "score": {
                    "first_successor_state": "[CH3:1][Cl:3]",
                    "reference_first_successor_exact": False,
                },
            }
        ]
    }
    rows, report = builder.build_rows(
        groups, {"rxn": {"id": "rxn"}}, max_negatives=4
    )
    assert [row["messages"][-1]["content"] for row in rows] == ["P", "N"]
    assert report["positives"] == report["hard_negatives"] == 1


def test_source_lookup_uses_trace_source_id(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text(
        json.dumps({"id": "artifact:rxn", "source_id": "rxn"}) + "\n",
        encoding="utf-8",
    )
    found = builder._source_rows(path, {"rxn"})
    assert found["rxn"]["id"] == "artifact:rxn"
