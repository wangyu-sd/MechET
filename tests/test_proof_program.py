from mechet.proof_program import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
    compile_from_states,
    execute_proof,
    format_proof_output,
    parse_proof_program,
    verify_proof,
)


def substitution_program() -> ProofProgram:
    # Reverse a C-O product bond into alkyl bromide + hydroxide.
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


def test_proof_executes_and_derives_precursor_without_answer():
    text = format_proof_output(substitution_program())
    assert "<answer>" not in text
    parsed = parse_proof_program(text)
    result = execute_proof(parsed)
    assert result.ok, result.diagnostics
    assert "[CH3:1][Br:3]" in result.precursor_smiles
    assert "[OH-:2]" in result.precursor_smiles


def test_tampered_lone_pair_fails_execution():
    text = format_proof_output(substitution_program()).replace(
        "LP 2 +2",
        "LP 2 +4",
    )
    result = execute_proof(text)
    assert not result.ok
    assert "LP execution mismatch" in result.diagnostics[0]["message"]


def test_endpoint_is_scored_from_executor_output():
    text = format_proof_output(substitution_program())
    score = verify_proof(
        text,
        expected_precursor="[CH3:1][Br:3].[OH-:2]",
    )
    assert score["execute_ok"]
    assert score["endpoint_exact"]


def test_state_annotated_mechanism_compiles_to_action_only_proof():
    program = compile_from_states(
        target_smiles="[CH3:1][OH:2]",
        target_state_ids=["s0"],
        precursor_state_id="s1",
        states={
            "s0": "[CH3:1][OH:2].[Br-:3]",
            "s1": "[CH3:1][Br:3].[OH-:2]",
        },
        edges=[("s0", "s1")],
    )
    text = format_proof_output(program)
    assert "STATE s0" not in text
    assert "<answer>" not in text
    result = execute_proof(text)
    assert result.ok, result.diagnostics
    score = verify_proof(
        text,
        expected_precursor="[CH3:1][Br:3].[OH-:2]",
    )
    assert score["endpoint_exact"]
