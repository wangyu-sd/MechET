from mechet.proof_curriculum import corrupt_proof
from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram, format_proof_output
from mechet.proof_rlvr import score_proof_group


def proof() -> str:
    return format_proof_output(
        ProofProgram(
            target_smiles="[CH3:1][OH:2]",
            roots={"s0": ["[Br-:3]"]},
            precursor_state_id="s1",
            edges=[ProofEdge(
                "s0",
                "s1",
                bonds=[(1, 2, -1), (1, 3, +1)],
                lone_pairs=[(2, +2), (3, -2)],
                charges=[ChargeAction(2, 0, -1), ChargeAction(3, -1, 0)],
            )],
        )
    )


def row():
    value = proof()
    return {
        "id": "toy",
        "messages": [
            {"role": "user", "content": "TARGET: [CH3:1][OH:2]"},
            {"role": "assistant", "content": value},
        ],
        "metadata": {"core_precursor": "[CH3:1][Br:3].[OH-:2]"},
    }


def test_diversity_bonus_never_rewards_invalid_strings():
    valid = proof()
    invalid = corrupt_proof(valid, corruption_type="LP_WRONG_DELTA").corrupted_proof
    scored = score_proof_group(
        row(),
        [valid, valid, invalid],
        config={
            "mode": "hypothesis",
            "execute": 3.0,
            "endpoint_core_exact": 0.0,
            "composition_match": 0.0,
            "invalid_proof": -2.0,
            "new_class_bonus": 1.0,
            "duplicate_class_penalty": 0.5,
        },
    )
    assert scored[0]["execute_ok"]
    assert scored[2]["execute_ok"] is False
    assert scored[2]["diversity_adjustment"] == 0.0
    assert scored[0]["equivalence_digest"] == scored[1]["equivalence_digest"]
