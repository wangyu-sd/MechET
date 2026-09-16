import pytest

from mechet.electron_flow_trace import ElectronFlowTrace
from mechet.proof_program import ProofProgramError
from mechet.python_program import (
    PythonElectronProgram,
    PythonElectronStep,
    execute_python_program,
    format_python_program,
    parse_python_program,
)


def substitution_program() -> PythonElectronProgram:
    return PythonElectronProgram(
        target_smiles="[CH3:1][OH:2]",
        steps=(
            PythonElectronStep(
                imports=("[Br-:3]",),
                moves=(
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
                ),
            ),
        ),
    )


def test_python_program_round_trip_and_execution():
    text = format_python_program(substitution_program())
    assert "move(bond(1, 2), atom(2))" in text
    assert "move(lp(3), bond(1, 3))" in text
    parsed = parse_python_program(text)
    result = execute_python_program(parsed, target_smiles="[CH3:1][OH:2]")
    assert result.ok, result.diagnostics
    assert result.flow_trace is not None
    assert "[CH3:1][Br:3]" in result.precursor_smiles
    assert "[OH-:2]" in result.precursor_smiles


def test_trace_can_be_serialized_without_model_visible_states():
    executed = execute_python_program(substitution_program())
    assert executed.ok and executed.flow_trace is not None
    text = format_python_program(executed.flow_trace)
    assert "state_before" not in text
    assert execute_python_program(text, target_smiles="[CH3:1][OH:2]").ok


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('echo unsafe')",
        "mechanism(target=f(), steps=[])",
        "import os\nmechanism()",
        "program = mechanism()",
        "mechanism(**payload)",
        "mechanism(target='C', steps=[step(eval('1'))])",
    ],
)
def test_python_program_rejects_executable_python(payload: str):
    with pytest.raises(ProofProgramError):
        parse_python_program(payload)


def test_python_program_rejects_unknown_step_arguments():
    payload = """mechanism(
        target='[CH3:1][OH:2]',
        steps=[step(move(bond(1, 2), atom(2)), shell=True)],
    )"""
    with pytest.raises(ProofProgramError, match="unknown step argument"):
        parse_python_program(payload)


def test_python_program_replays_be_delta_fallback():
    text = """mechanism(
      target='[CH3:1][OH:2]',
      steps=[step(delta(bonds=[(1, 2, -1)], charges=[(1, 0, 1), (2, 0, -1)]))],
    )"""
    result = execute_python_program(text, target_smiles="[CH3:1][OH:2]")
    assert result.ok, result.diagnostics
    assert result.precursor_smiles
