from mechet.proof_diagnostics import (
    diagnose_proof,
    format_repair_feedback,
    repair_proof_once,
)
from mechet.proof_program import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
    format_proof_output,
)


def substitution_program() -> ProofProgram:
    return ProofProgram(
        target_smiles="[CH3:1][OH:2]",
        roots={"s0": ["[Br-:3]"]},
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


def test_lp_mismatch_yields_repairable_certificate():
    tampered = format_proof_output(substitution_program()).replace(
        "LP 2 +2",
        "LP 2 +4",
    )
    certificate = diagnose_proof(tampered)
    assert certificate is not None
    assert certificate.code == "LP_EXECUTION_MISMATCH"
    assert certificate.edge == "s0->s1"
    assert certificate.repairable
    assert "LP 2 +2" in certificate.repair_lines
    feedback = format_repair_feedback(certificate)
    assert "FAIL LP_EXECUTION_MISMATCH" in feedback
    assert "EDGE s0->s1" in feedback


def test_deterministic_lp_repair_restores_execution():
    tampered = format_proof_output(substitution_program()).replace(
        "LP 2 +2",
        "LP 2 +4",
    )
    repaired = repair_proof_once(tampered)
    assert repaired.changed
    assert repaired.execute_ok, repaired.diagnostics
    assert "LP 2 +2" in repaired.repaired_text
