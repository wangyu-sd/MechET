import json

import pytest
import torch

from build_graph_electron_full import compile_row
from mechet.direct_graph_rl_env import DirectGraphElectronEnv
from mechet.electron_policy_protocol import CompressedTrajectory
from mechet.graph_electron_policy import (
    ACTION_FAMILIES,
    GraphElectronPolicy,
    smiles_to_graph,
)
from mechet.graph_fragment_actions import decompose_reactive_fragment


TARGET = "[CH3:1][Br:2]"


def tiny_policy() -> GraphElectronPolicy:
    torch.manual_seed(19)
    return GraphElectronPolicy(hidden_dim=24, num_layers=1, dropout=0.0)


def test_compiler_schedules_reactive_import_at_first_use_and_environment_at_end():
    row = {
        "id": "first-use",
        "expected_precursor": "[Br-:2].[CH3:1][O:3][CH3:4].[Na+:5]",
        "metadata": {
            "trace_plan": {
                "target_smiles": TARGET,
                "initial_imports": ["[O-:3][CH3:4]", "[Na+:5]"],
                "steps": [
                    {
                        "state_after": "[Br-:2].[CH3:1][O:3][CH3:4].[Na+:5]",
                        "imports": [],
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
                ],
            }
        },
    }
    _, decisions = compile_row((0, json.dumps(row)))
    assert [item["kind"] for item in decisions] == [
        "IMPORT_REACTIVE",
        "FLOW",
        "IMPORT_ENV",
        "FINISH",
    ]
    assert "Na" not in decisions[1]["current"]
    assert "Na" in decisions[-1]["current"]
    assert len(decisions[0]["program"]["active_atoms"]) == 1


def test_rejected_executor_action_is_visible_in_next_history():
    env = DirectGraphElectronEnv(max_steps=3)
    env.reset(target=TARGET)
    transition = env.step({"kind": "FLOW", "moves": []})
    event = transition.next_observation.history.events[-1]
    assert not transition.accepted
    assert not event.accepted
    assert event.error_code == "EMPTY_ELECTRON_EVENT"


def test_product_alignment_is_map_invariant_and_detects_local_change():
    target_a = "[CH3:1][OH:2]"
    current_a = "[CH2:1]=[O:2]"
    target_b = "[CH3:101][OH:909]"
    current_b = "[CH2:101]=[O:909]"
    left = smiles_to_graph(current_a, target_smiles=target_a)
    right = smiles_to_graph(current_b, target_smiles=target_b)
    assert torch.equal(left.atoms, right.atoms)
    assert torch.allclose(left.alignment, right.alignment)
    assert bool((left.alignment[:, 1:7].abs() > 0).any())


def test_multi_active_site_sampling_matches_nonempty_set_contract():
    logits = torch.tensor([5.0, 4.0, -5.0])
    selected, logprob = GraphElectronPolicy._sample_active_set(
        logits, greedy=True, require_nonempty=True
    )
    assert selected == (0, 1)
    assert logprob <= 0.0


def test_illegal_lone_pair_source_is_masked_not_learned_as_candidate():
    model = tiny_policy()
    with pytest.raises(ValueError, match="not legal"):
        model.direct_flow_nll(
            "[CH4:1].[CH4:2]",
            "[CH4:1]",
            [
                {
                    "source": {"kind": "LP", "atoms": [1]},
                    "sink": {"kind": "BOND", "atoms": [1, 2]},
                    "electrons": 2,
                }
            ],
        )


def test_unified_sampler_supports_finish_and_be_delta():
    model = tiny_policy().eval()
    with torch.no_grad():
        model.family_head.weight.zero_()
        model.family_head.bias.fill_(-10.0)
        model.family_head.bias[ACTION_FAMILIES.index("FINISH")] = 10.0
    finished = model.sample_action(TARGET, TARGET, greedy=True)
    assert finished["action"] == {"kind": "FINISH"}
    env = DirectGraphElectronEnv(max_steps=4)
    observation = env.reset(
        target=TARGET,
        expected_precursor="[Br-:2].[CH3:1][O:4][CH3:3]",
    )
    program = decompose_reactive_fragment(
        "[O-:3][CH3:4]", participating_maps=(3,), role="NUCLEOPHILE"
    )
    observation = env.step(
        {"kind": "IMPORT_REACTIVE", "program": program.to_dict()}
    ).next_observation
    observation = env.step(
        {
            "kind": "FLOW",
            "moves": [
                {
                    "source": {"kind": "LP", "atoms": [4]},
                    "sink": {"kind": "BOND", "atoms": [1, 4]},
                    "electrons": 2,
                },
                {
                    "source": {"kind": "BOND", "atoms": [1, 2]},
                    "sink": {"kind": "ATOM", "atoms": [2]},
                    "electrons": 2,
                },
            ],
        }
    ).next_observation
    terminal = env.step(
        model.sample_action(
            observation.current,
            observation.target,
            trajectory=observation.history,
            greedy=True,
        )["action"]
    )
    assert terminal.done and terminal.reward > 0

    with torch.no_grad():
        model.be_operation_head[-1].weight.zero_()
        model.be_operation_head[-1].bias[:] = torch.tensor([10.0, -10.0, -10.0])
    be = model.rollout_be_delta(TARGET, TARGET, max_edits=1, greedy=True)
    assert be["ok"]
    assert be["moves"][0]["mode"] == "BE_DELTA"
    assert be["moves"][0]["bond_deltas"]


def test_history_keeps_map_free_site_identity_and_is_bounded():
    decision = {
        "kind": "FLOW",
        "current": "[CH3:1][Br:2].[O-:3]",
        "moves": [
            {
                "source": {"kind": "LP", "atoms": [3]},
                "sink": {"kind": "BOND", "atoms": [1, 3]},
                "electrons": 2,
            }
        ],
    }
    history = CompressedTrajectory()
    for _ in range(40):
        history = history.append(decision)
    assert len(history.events) == 32
    assert history.events[-1].site_codes
    assert ":3" not in json.dumps(history.to_dict())
