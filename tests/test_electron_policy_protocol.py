import json

import torch

from mechet.electron_policy_protocol import (
    CompressedTrajectory,
    STAGE_STATE_BC,
    STAGE_TRAJECTORY_BC,
    event_from_decision,
    llm_training_record,
)
from mechet.graph_electron_policy import GraphElectronPolicy


TARGET = "[CH3:1][Br:2]"
STATE = "[CH3:1][Br:2].[O-:3]"
FLOW = {
    "reaction_id": "r1",
    "kind": "FLOW",
    "current": STATE,
    "target": TARGET,
    "moves": [
        {
            "source": {"kind": "LP", "atoms": [3]},
            "sink": {"kind": "BOND", "atoms": [1, 3]},
            "electrons": 2,
        },
        {
            "source": {"kind": "BOND", "atoms": [1, 2]},
            "sink": {"kind": "ATOM", "atoms": [2]},
            "electrons": 2,
        },
    ],
}


def test_compact_history_is_ordered_and_contains_no_private_maps():
    imported = {
        "kind": "IMPORT_REACTIVE",
        "program": {
            "role": "NUCLEOPHILE",
            "atoms": [{}, {}],
            "active_atoms": [0],
            "extra_bonds": [],
        },
    }
    history = CompressedTrajectory().append(imported).append(FLOW)
    payload = history.to_dict()
    assert history.family_path == ("IMPORT_REACTIVE", "FLOW")
    assert "NUCLEOPHILE" in history.render()
    serialized = json.dumps(payload)
    assert '"atoms": [3]' not in serialized
    assert '"atoms": [1, 3]' not in serialized
    assert CompressedTrajectory.from_dict(payload) == history


def test_llm_views_share_target_but_stage2_adds_history_without_maps():
    history = CompressedTrajectory().append(
        {
            "kind": "IMPORT_REACTIVE",
            "program": {
                "role": "NUCLEOPHILE",
                "atoms": [{}, {}],
                "active_atoms": [0],
                "extra_bonds": [],
            },
        }
    )
    decision = {**FLOW, "history": history.to_dict()}
    state_only = llm_training_record(decision, stage=STAGE_STATE_BC)
    trajectory = llm_training_record(decision, stage=STAGE_TRAJECTORY_BC)
    assert "COMPRESSED_TRAJECTORY" not in state_only["input"]
    assert "COMPRESSED_TRAJECTORY" in trajectory["input"]
    assert "NUCLEOPHILE" in trajectory["input"]
    assert "TARGET_PRODUCT: CBr" in trajectory["input"]
    assert "[CH3:1]" not in trajectory["input"]
    target = json.loads(trajectory["output"])
    assert target["action"] == "FLOW"
    assert target["arguments"]["direction"] == "retrosynthetic"
    assert "A03" in target["arguments"]["electron_flow"][0]["source"]


def test_graph_history_gate_preserves_stage1_initialization_then_adds_context():
    torch.manual_seed(11)
    model = GraphElectronPolicy(hidden_dim=24, num_layers=1, dropout=0.0).eval()
    history = CompressedTrajectory().append(FLOW)
    with torch.no_grad():
        state_only = model.encode_context(STATE, TARGET).vector
        initially_gated = model.encode_context(STATE, TARGET, history=history).vector
        assert torch.allclose(state_only, initially_gated, atol=1e-7)
        model.history_gate.fill_(1.0)
        trajectory = model.encode_context(STATE, TARGET, history=history).vector
    assert not torch.allclose(state_only, trajectory)


def test_training_nll_can_skip_host_metric_synchronization():
    model = GraphElectronPolicy(hidden_dim=24, num_layers=1, dropout=0.0)
    loss, parts = model.direct_flow_nll(
        STATE, TARGET, FLOW["moves"], return_parts=False
    )
    assert torch.isfinite(loss)
    assert parts == {}


def test_stage3_recomputes_direct_action_logprob_without_candidates():
    model = GraphElectronPolicy(hidden_dim=24, num_layers=1, dropout=0.0)
    history = CompressedTrajectory().append(
        {
            "kind": "IMPORT_REACTIVE",
            "program": {
                "role": "NUCLEOPHILE",
                "atoms": [{}, {}],
                "active_atoms": [0],
                "extra_bonds": [],
            },
        }
    )
    nll = model.canonical_action_nll(FLOW, history=history)
    sampled = model.sample_next_family(
        STATE, TARGET, trajectory=history, greedy=True
    )
    assert torch.isfinite(nll)
    assert sampled["candidate_enumeration"] is False
    assert sampled["family"] in {
        "FLOW",
        "BE_DELTA",
        "IMPORT_ENV",
        "IMPORT_REACTIVE",
        "FINISH",
    }


def test_event_projection_records_types_not_locations():
    event = event_from_decision(FLOW)
    assert event.source_kinds == ("LP", "BOND")
    assert event.sink_kinds == ("BOND", "ATOM")
    assert event.electron_moves == 2
