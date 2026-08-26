import json

from rdkit import Chem

from mechet.agent_env import AgentEnvConfig
from mechet.compact_observation import (
    compact_terminal_observation,
    compact_transition_observation,
    mapped_reaction_center_smiles,
    move_atom_maps,
)
from mechet.trace_agent_env import TraceOwnedAgentEnv


def _maps(smiles: str) -> set[int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return {atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}


def test_local_mapped_smiles_keeps_center_and_one_hop_only():
    state = "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[Na+:6]"
    local = mapped_reaction_center_smiles(state, [2, 3], radius=1)

    assert _maps(local) == {1, 2, 3, 4}
    assert 5 not in _maps(local)
    assert 6 not in _maps(local)


def test_compact_transition_contains_delta_not_complete_states():
    before = "[CH3:1][Br:2].[OH-:3]"
    after = "[Br-:2].[CH3:1][OH:3]"
    moves = [
        {
            "source": {"kind": "LP", "atoms": [3]},
            "sink": {"kind": "BOND", "atoms": [1, 3]},
            "electrons": 2,
        }
    ]
    result = compact_transition_observation(
        result={"ok": True, "code": "PASS", "remaining_tool_calls": 4},
        state_before=before,
        state_after=after,
        moves=moves,
        include_local_state=True,
    )

    assert move_atom_maps(moves) == (1, 3)
    assert result["changed_atom_maps"] == [1, 3]
    assert "state_smiles" not in result
    assert "state_before" not in result
    assert "state_after" not in result
    assert _maps(result["local_state_before"]) == {1, 2, 3}
    assert _maps(result["local_state_after"]) == {1, 3}


def test_trace_environment_compact_mode_keeps_full_state_internal():
    env = TraceOwnedAgentEnv(
        config=AgentEnvConfig(
            max_tool_calls=8,
            observation_mode="reaction_center_delta",
        )
    )
    env.reset(
        target_smiles="[CH3:1][Br:2]",
        expected_precursor="[Br-:2].[CH3:1][OH:3]",
    )
    imported = json.loads(env.import_fragment("[OH-:3]"))
    assert "state_smiles" not in imported
    assert "imported_fragment" not in imported

    transition = json.loads(
        env.apply_coupled_electron_moves(
            json.dumps(
                [
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
            )
        )
    )
    assert transition["ok"] is True
    assert "state_smiles" not in transition
    assert "trace_step" not in transition
    assert transition["changed_atom_maps"] == [1, 2, 3]
    assert env.current_state == "[Br-:2].[CH3:1][OH:3]"

    terminal = json.loads(env.finish_trace())
    assert terminal["derived_precursor"] == env.current_state
    assert "compiled_proof" not in terminal
    assert "full_precursor_state" not in terminal
    assert "MECH_PROOF v1" in env.final_result["compiled_proof"]


def test_action_delta_inspection_exposes_inventory_but_not_state_smiles():
    env = TraceOwnedAgentEnv(
        config=AgentEnvConfig(max_tool_calls=4, observation_mode="action_delta")
    )
    env.reset(target_smiles="[CH3:1][Br:2]")

    visible = json.loads(env.inspect_state())

    assert visible["ok"] is True
    assert visible["observation_mode"] == "action_delta_v1"
    assert visible["sources"]
    assert visible["sinks"]
    assert "state_smiles" not in visible
    assert env.current_state == "[CH3:1][Br:2]"


def test_compact_terminal_emits_endpoint_once_without_proof_duplication():
    value = compact_terminal_observation(
        {
            "ok": True,
            "formal_execute": True,
            "derived_precursor": "[Br-:2].[CH3:1][OH:3]",
            "full_precursor_state": "[Br-:2].[CH3:1][OH:3]",
            "structural_precursor": "[CH3:1][OH:3]",
            "compiled_proof": "MECH_PROOF v1\n...",
            "trace_digest": "abc",
        }
    )
    assert value["derived_precursor"] == "[Br-:2].[CH3:1][OH:3]"
    assert "full_precursor_state" not in value
    assert "structural_precursor" not in value
    assert "compiled_proof" not in value


def test_action_delta_mode_exposes_no_intermediate_molecular_state():
    value = compact_transition_observation(
        result={"ok": True, "code": "PASS", "remaining_tool_calls": 4},
        state_before="[CH3:1][Br:2].[OH-:3]",
        state_after="[Br-:2].[CH3:1][OH:3]",
        moves=[
            {
                "source": {"kind": "LP", "atoms": [3]},
                "sink": {"kind": "BOND", "atoms": [1, 3]},
            }
        ],
    )

    assert value["observation_mode"] == "action_delta_v1"
    assert "local_state_before" not in value
    assert "local_state_after" not in value
    assert "changed_atom_maps" not in value
    assert set(value) == {
        "ok",
        "code",
        "observation_mode",
        "remaining_tool_calls",
    }


def test_failed_action_delta_result_cannot_leak_state():
    value = compact_transition_observation(
        result={
            "ok": False,
            "code": "MOVE_FAILED",
            "state_smiles": "[CH3:1][Br:2]",
            "trace_step": {"state_before": "secret", "state_after": "secret"},
        },
        state_before="[CH3:1][Br:2]",
        state_after="[CH3:1][Br:2]",
        moves=[],
    )

    assert value == {
        "ok": False,
        "code": "MOVE_FAILED",
        "observation_mode": "action_delta_v1",
    }
