"""Regression test for the public README quickstart."""

from mechet import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
    format_proof_output,
    verify_proof,
)


def test_readme_quickstart_executes():
    program = ProofProgram(
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
    score = verify_proof(
        format_proof_output(program),
        expected_precursor="[CH3:1][Br:3].[OH-:2]",
    )
    assert score["execute_ok"]
    assert score["endpoint_exact"]
