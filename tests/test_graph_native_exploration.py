"""CPU component contracts for pointer sampling; no trained model is loaded."""
import random

import pytest
import torch

from mechet.crn_graph_bridge import GraphFlowCandidate
from mechet import graph_endpoint_rl as objective
from scripts import train_crn_graph_earho as trainer


def require(module, name):
    result = getattr(module, name, None)
    if result is None:
        pytest.fail(f"missing exploration integration: {name}")
    return result


def candidate(source, score):
    move = {"source": {"kind": "LP", "atoms": [source]},
            "sink": {"kind": "ATOM", "atoms": [3]}, "electrons": 2}
    return GraphFlowCandidate({"event": source}, (move,), score)


def test_exploration_expands_support_without_adding_gold_or_reordering_events():
    merge = require(trainer, "merge_proposal_support")
    beam = candidate(1, -1.)
    outside = candidate(2, -2.)
    pool = merge([beam], [outside, beam])
    assert pool == (beam, outside)
    # A one-event beam is not rejected just because K is eight: IID repeats are legal.
    assert merge([beam], []) == (beam,)


def real_pointer_fixture():
    # Optional external CRN implementation, not a checkpoint or data experiment.
    pytest.importorskip("crn_jepa")
    from crn_jepa.models.grounded_event_policy import GroundedElectronFlowPolicy
    from crn_jepa.models.grounded_graph import build_grounded_graph
    from crn_jepa.grounded_actions import MechETGroundedInventory
    state = "[CH3:1][OH:2].[Cl-:3]"
    graph = build_grounded_graph(state, None)
    inventory = MechETGroundedInventory(state)
    torch.manual_seed(17)
    policy = GroundedElectronFlowPolicy(hidden_dim=16, num_graph_layers=1,
                                       dropout=0., max_event_moves=2).eval()
    nodes, context, prefix = policy.encode_observation(graph, event_depth=0, remaining_budget=4)
    return policy, graph, inventory, nodes, context, prefix


def test_native_sample_logprob_matches_actual_source_sink_commit_heads():
    sample = require(objective, "sample_grounded_event")
    policy, graph, inventory, nodes, context, prefix = real_pointer_fixture()
    with torch.no_grad():
        policy.decision_head[-1].weight.zero_()
        policy.decision_head[-1].bias.copy_(torch.tensor([-100., 100.]))
        event = sample(policy, graph, inventory, nodes, context, prefix, random.Random(17))
    assert event is not None
    moves, sampled_logp = event
    assert len(moves) == 1
    scored = objective.grounded_event_log_probability(policy, graph, inventory,
                                                    moves, nodes, context, prefix)
    assert sampled_logp == pytest.approx(scored.item(), abs=1e-5)


def test_native_exploration_does_not_force_commit_at_move_limit():
    sample = require(objective, "sample_grounded_event")
    policy, graph, inventory, nodes, context, prefix = real_pointer_fixture()
    with torch.no_grad():
        policy.decision_head[-1].weight.zero_()
        policy.decision_head[-1].bias.copy_(torch.tensor([100., -100.]))
        assert sample(policy, graph, inventory, nodes, context, prefix, random.Random(17)) is None
