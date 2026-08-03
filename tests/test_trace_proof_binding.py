import json

from mechet.proof_program import execute_proof, sides_equal
from mechet.trace_agent_env import TraceOwnedAgentEnv


def reverse_sn2_moves():
    return [
        {
            "source": {"kind": "BOND", "atoms": [1, 2]},
            "sink": {"kind": "ATOM", "atoms": [2]},
            "electrons": 2,
        },
        {
            "source": {"kind": "LP", "atoms": [3]},
            "sink": {"kind": "BOND", "atoms": [1, 3]},
            "electrons": 2,
        },
    ]


def test_finish_trace_compiles_the_only_terminal_proof():
    env = TraceOwnedAgentEnv()
    observation = json.loads(
        env.reset(
            target_smiles="[Br-:3].[CH3:1][OH:2]",
            expected_precursor="[CH3:1][Br:3].[OH-:2]",
        )
    )
    assert observation["faithfulness_contract"]["free_form_proof_submission"] is False

    move_result = json.loads(
        env.apply_coupled_electron_moves(json.dumps(reverse_sn2_moves()))
    )
    assert move_result["ok"]
    assert move_result["trace_bound"]

    result = json.loads(env.finish_trace())
    assert result["ok"]
    assert result["trace_bound"]
    assert result["endpoint_source"] == "environment_owned_trace"
    assert result["endpoint_exact"]
    assert sides_equal(
        result["derived_precursor"],
        "[CH3:1][Br:3].[OH-:2]",
        ignore_maps=False,
    )
    replay = execute_proof(result["compiled_proof"])
    assert replay.ok
    assert sides_equal(replay.precursor_smiles, result["derived_precursor"], ignore_maps=False)


def test_imported_atoms_are_recorded_in_compiled_proof():
    env = TraceOwnedAgentEnv()
    env.reset(
        target_smiles="[CH3:1][OH:2]",
        expected_precursor="[CH3:1][Br:3].[OH-:2]",
    )
    imported = json.loads(env.import_fragment("[Br-:3]"))
    assert imported["ok"]
    moved = json.loads(
        env.apply_coupled_electron_moves(json.dumps(reverse_sn2_moves()))
    )
    assert moved["ok"]
    result = json.loads(env.finish_trace())
    assert result["ok"]
    assert 'IMPORT "[Br-:3]"' in result["compiled_proof"]
    assert result["endpoint_exact"]


def test_free_form_proof_submission_is_rejected_without_finalizing():
    env = TraceOwnedAgentEnv()
    env.reset(target_smiles="[Br-:3].[CH3:1][OH:2]")
    result = json.loads(env.submit_proof("<proof>invented</proof>"))
    assert not result["ok"]
    assert result["code"] == "FREE_FORM_PROOF_DISABLED"
    assert not env.finalized


def test_finish_rejects_empty_trace_and_uncommitted_imports():
    env = TraceOwnedAgentEnv()
    env.reset(target_smiles="[CH3:1][OH:2]")
    empty = json.loads(env.finish_trace())
    assert empty["code"] == "TRACE_COMPILATION_FAILED"

    env = TraceOwnedAgentEnv()
    env.reset(target_smiles="[CH3:1][OH:2]")
    assert json.loads(env.import_fragment("[Br-:3]"))["ok"]
    pending = json.loads(env.finish_trace())
    assert pending["code"] == "UNCOMMITTED_IMPORTS"


def test_trace_digest_changes_with_action_history():
    first = TraceOwnedAgentEnv()
    first.reset(target_smiles="[Br-:3].[CH3:1][OH:2]")
    first.apply_coupled_electron_moves(json.dumps(reverse_sn2_moves()))

    second = TraceOwnedAgentEnv()
    second.reset(target_smiles="[Br-:3].[CH3:1][OH:2]")
    second.apply_coupled_electron_moves(json.dumps(list(reversed(reverse_sn2_moves()))))

    # Coupled arrows are chemically equivalent, but the serialized action trace
    # remains auditable and records the exact model-issued order.
    assert first.flow_trace.digest() != second.flow_trace.digest()
