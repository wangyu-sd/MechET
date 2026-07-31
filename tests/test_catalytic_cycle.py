from mechet.catalytic_cycle import CatalyticCycle, CatalyticCycleStep, verify_catalytic_cycle
from mechet.proof_program import ChargeAction, ProofEdge, ProofProgram, format_proof_output


def proof() -> str:
    return format_proof_output(
        ProofProgram(
            target_smiles="[CH3:1][OH:2]",
            roots={"s0": ["[Br-:3]", "[Na+:4]"]},
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


def test_cycle_verifier_checks_regeneration_and_oxidation_closure():
    cycle = CatalyticCycle(
        cycle_id="toy",
        catalyst_initial="[Na+:4]",
        steps=(CatalyticCycleStep(
            label="substitution",
            proof=proof(),
            catalyst_before="[Na+:4]",
            catalyst_after="[Na+:4]",
            oxidation_state_before=1,
            oxidation_state_after=1,
        ),),
    )
    result = verify_catalytic_cycle(cycle)
    assert result.ok
    assert result.proof_execution_ok
    assert result.catalyst_regenerated
    assert result.oxidation_state_closed


def test_cycle_rejects_missing_catalyst_regeneration():
    cycle = CatalyticCycle(
        cycle_id="broken",
        catalyst_initial="[Na+:4]",
        steps=(CatalyticCycleStep(
            label="step",
            proof=proof(),
            catalyst_before="[Na+:4]",
            catalyst_after="[K+:5]",
        ),),
    )
    result = verify_catalytic_cycle(cycle)
    assert not result.ok
    assert not result.catalyst_regenerated
