import importlib.util
import json
from pathlib import Path

import torch

from mechet.forward_expert import ElectronContainer
from mechet.graph_electron_policy import (
    GraphElectronPolicy,
    expectile_loss,
    graph_iql_losses,
    smiles_to_graph,
    strip_atom_maps,
)
from mechet.graph_fragment_actions import (
    bind_reactive_import,
    classify_imports,
    decompose_reactive_fragment,
    replay_reactive_fragment,
)
from mechet.transactional_event_space import MoveInventory

_PILOT_SPEC = importlib.util.spec_from_file_location(
    "train_graph_electron_pilot",
    Path(__file__).parents[1] / "scripts" / "train_graph_electron_pilot.py",
)
assert _PILOT_SPEC is not None and _PILOT_SPEC.loader is not None
_PILOT_MODULE = importlib.util.module_from_spec(_PILOT_SPEC)
_PILOT_SPEC.loader.exec_module(_PILOT_MODULE)
build_decisions = _PILOT_MODULE.build_decisions


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


def test_imports_split_into_environment_and_reactive_actions():
    plan = {
        "initial_imports": ["[O-:3][CH3:4]", "[Na+:5]"],
        "steps": [{"moves": MOVES}],
    }
    imports = classify_imports(plan)
    assert [item.kind for item in imports] == ["IMPORT_REACTIVE", "IMPORT_ENV"]
    assert imports[0].role == "NUCLEOPHILE"
    assert imports[0].participating_maps == (3,)


def test_reactive_fragment_program_replays_without_atom_maps():
    program = decompose_reactive_fragment(
        "[cH:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1",
        participating_maps=(7,),
        role="BOND_DONOR",
    )
    assert replay_reactive_fragment(program) == "c1ccccc1"
    assert program.active_atoms


def test_reactive_import_must_be_used_by_the_next_electron_event():
    program = decompose_reactive_fragment(
        "[O-:3][CH3:4]",
        participating_maps=(3,),
        role="NUCLEOPHILE",
    )
    guard = bind_reactive_import(program, assigned_maps=(30, 40))
    active_map = guard.active_maps[0]
    accepted = guard.validate(
        [
            {
                "source": {"kind": "LP", "atoms": [active_map]},
                "sink": {"kind": "BOND", "atoms": [1, active_map]},
                "electrons": 2,
            }
        ]
    )
    rejected = guard.validate(MOVES)
    assert accepted["ok"] and accepted["used_maps"] == (active_map,)
    assert not rejected["ok"]
    assert rejected["code"] == "UNUSED_REACTIVE_IMPORT"


def test_reactive_fragment_generator_has_finite_supervised_loss():
    model = tiny_policy().train()
    program = decompose_reactive_fragment(
        "[O-:3][CH3:4]",
        participating_maps=(3,),
        role="NUCLEOPHILE",
    )
    loss, parts = model.reactive_fragment_nll(TARGET, TARGET, program)
    assert torch.isfinite(loss)
    assert set(parts) == {
        "family", "role", "operation", "atom", "position", "bond", "active"
    }
    loss.backward()
    assert model.fragment_element_head.weight.grad is not None


def test_reactive_fragment_rollout_returns_map_free_executor_option():
    model = tiny_policy().eval()
    with torch.no_grad():
        for head in (
            model.fragment_element_head,
            model.fragment_charge_head,
            model.fragment_h_head,
            model.fragment_no_implicit_head,
            model.fragment_radical_head,
            model.fragment_chiral_head,
        ):
            head.weight.zero_()
            head.bias.zero_()
        model.fragment_element_head.bias[6] = 10.0
        model.fragment_charge_head.bias[5] = 10.0
        model.fragment_h_head.bias[4] = 10.0
        model.fragment_no_implicit_head.bias[1] = 10.0
        model.fragment_radical_head.bias[0] = 10.0
        model.fragment_chiral_head.bias[0] = 10.0
    result = model.rollout_reactive_fragment(
        TARGET,
        TARGET,
        role="NUCLEOPHILE",
        max_atoms=1,
        max_extra_bonds=0,
        greedy=True,
    )
    assert result["ok"]
    assert result["fragment"] == "C"
    assert result["active_atom"] == 0
    assert ":" not in result["fragment"]


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


def test_step_import_is_present_in_following_be_delta_observation(tmp_path):
    row = {
        "id": "step-import-regression",
        "metadata": {
            "trace_plan": {
                "target_smiles": "[BH3:1]",
                "initial_imports": [],
                "steps": [
                    {
                        "step_index": 0,
                        "state_before": "[BH3:1]",
                        "state_after": "[BH2:1][H:29]",
                        "imports": ["[H:29]"],
                        "moves": [
                            {
                                "mode": "BE_DELTA",
                                "bond_deltas": [{"atoms": [1, 29], "delta": 1}],
                                "charge_actions": [],
                            }
                        ],
                    }
                ],
            }
        },
    }
    source = tmp_path / "one.jsonl"
    source.write_text(json.dumps(row) + "\n")
    decisions, counts = build_decisions(
        source, reaction_limit=1, decision_limit=10
    )
    event = next(item for item in decisions if item["kind"] == "BE_DELTA")
    assert "[H:29]" in event["current"]
    model = tiny_policy().train()
    loss, _ = model.be_delta_nll(
        event["current"], event["target"], event["moves"][0]
    )
    assert torch.isfinite(loss)
    assert counts == {"IMPORT_REACTIVE": 1, "BE_DELTA": 1, "FINISH": 1}
