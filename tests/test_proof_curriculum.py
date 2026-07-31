from mechet.proof_curriculum import (
    corrupt_proof,
    preference_pair_from_corruption,
    repair_row_from_corruption,
)
from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram, execute_proof, format_proof_output
from mechet.proof_variants import build_equivalent_variants
from mechet.proof_equivalence import proofs_equivalent


def substitution_proof() -> str:
    program = ProofProgram(
        target_smiles="[CH3:1][OH:2]",
        roots={"s0": ["[Br-:3]", "[Na+:4]"]},
        precursor_state_id="s1",
        edges=[
            ProofEdge(
                "s0",
                "s1",
                bonds=[(1, 2, -1), (1, 3, +1)],
                lone_pairs=[(2, +2), (3, -2)],
                charges=[
                    ChargeAction(2, 0, -1),
                    ChargeAction(3, -1, 0),
                ],
            )
        ],
    )
    return format_proof_output(program)


def test_equivalent_variants_execute_and_match_partial_order():
    proof = substitution_proof()
    variants = build_equivalent_variants(proof, n_variants=3, seed=7)
    assert len(variants) >= 2
    for variant in variants:
        assert execute_proof(variant).ok
        assert proofs_equivalent(proof, variant)


def test_corruption_preference_and_repair_rows_are_verifier_grounded():
    proof = substitution_proof()
    corruption = corrupt_proof(
        proof,
        corruption_type="LP_WRONG_DELTA",
        source_id="rxn-1",
    )
    assert not corruption.observed_execute
    assert corruption.failure_code == "LP_EXECUTION_MISMATCH"
    pair = preference_pair_from_corruption(
        corruption,
        prompt_messages=[{"role": "user", "content": "TARGET: [CH3:1][OH:2]"}],
    )
    assert pair is not None
    assert pair["chosen_verdict"] == "EXECUTABLE"
    repair = repair_row_from_corruption(corruption, product="[CH3:1][OH:2]")
    assert repair is not None
    assert repair["task_type"] == "mech_proof_repair"
    assert "CERTIFICATE" in repair["messages"][1]["content"]
