import torch

from mechet.forward_expert import ElectronContainer
from mechet.graph_electron_policy import (
    GraphElectronPolicy,
    expectile_loss,
    graph_iql_losses,
    smiles_to_graph,
    strip_atom_maps,
)
from mechet.transactional_event_space import MoveInventory


TARGET = "[CH3:1][Br:2]"
STATE = "[CH3:1][Br:2].[O-:3]"
MOVES = [
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
]


def tiny_policy() -> GraphElectronPolicy:
    torch.manual_seed(7)
    return GraphElectronPolicy(hidden_dim=32, num_layers=2, dropout=0.0)


def test_graph_features_do_not_contain_map_ids():
    left = smiles_to_graph("[CH3:1][Br:2]")
    right = smiles_to_graph("[CH3:101][Br:909]")
    assert torch.equal(left.atoms, right.atoms)
    assert torch.equal(left.edge_index, right.edge_index)
    assert left.maps != right.maps


def test_source_scores_are_permutation_equivariant_by_private_map():
    model = tiny_policy().eval()
    states = (STATE, "[O-:3].[Br:2][CH3:1]")
    scored = []
    with torch.no_grad():
        for state in states:
            context = model.encode_context(state, TARGET)
            sources = MoveInventory.from_state(state).sources
            logits, _ = model.source_logits(context, sources)
            scored.append({item: float(value) for item, value in zip(sources, logits)})
    assert scored[0].keys() == scored[1].keys()
    for source in scored[0]:
        assert abs(scored[0][source] - scored[1][source]) < 1e-5


def test_flow_nll_is_finite_and_updates_policy():
    model = tiny_policy().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.family_head.weight.detach().clone()
    loss, parts = model.flow_nll(STATE, TARGET, MOVES)
    assert torch.isfinite(loss)
    assert set(parts) == {"family", "source", "sink", "commit"}
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.family_head.weight.detach())


def test_import_retrieval_uses_canonical_unmapped_fragment_identity():
    assert strip_atom_maps("[Na+:77]") == "[Na+]"
    model = tiny_policy().eval()
    loss = model.import_nll(
        TARGET,
        TARGET,
        fragments=("[Na+]", "[OH-]", "O"),
        gold_index=1,
    )
    assert torch.isfinite(loss)


def test_sparse_be_matrix_head_scores_bond_and_charge_edits():
    model = tiny_policy().train()
    payload = {
        "mode": "BE_DELTA",
        "bond_deltas": [
            {"atoms": [1, 2], "delta": -1},
            {"atoms": [1, 3], "delta": 1},
        ],
        "charge_actions": [{"atom_map": 3, "q0": -1, "q1": 0}],
    }
    loss, parts = model.be_delta_nll(STATE, TARGET, payload)
    assert torch.isfinite(loss)
    assert set(parts) == {"family", "operation", "position", "delta"}
    loss.backward()
    assert model.be_pair_head[-1].weight.grad is not None


def test_finish_and_iql_objectives_are_finite():
    model = tiny_policy().eval()
    assert torch.isfinite(model.finish_nll(STATE, TARGET))
    q = torch.tensor([1.0, 0.2], requires_grad=True)
    value = torch.tensor([0.4, 0.4], requires_grad=True)
    next_value = torch.tensor([0.5, 0.1])
    reward = torch.tensor([1.0, -0.1])
    done = torch.tensor([1.0, 0.0])
    log_prob = torch.tensor([-0.2, -1.0], requires_grad=True)
    losses = graph_iql_losses(
        q=q,
        value=value,
        next_value=next_value,
        reward=reward,
        done=done,
        log_prob=log_prob,
    )
    assert all(torch.isfinite(item) for item in losses.values())
    assert expectile_loss(torch.tensor([-1.0, 1.0])) > 0
    sum(losses.values()).backward()
    assert q.grad is not None and value.grad is not None and log_prob.grad is not None


def test_legal_action_mask_excludes_nonlocal_lp_bond_sink():
    inventory = MoveInventory.from_state(STATE)
    donor = ElectronContainer("LP", (3,))
    assert ElectronContainer("BOND", (1, 3)) in inventory.compatible_sinks(donor)
    assert ElectronContainer("BOND", (1, 2)) not in inventory.compatible_sinks(donor)
