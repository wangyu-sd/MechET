import pytest

from mechet.proof_program import ProofProgramError
from mechet.python_program import PythonElectronProgram, PythonElectronStep
from mechet.python_template_slots import (
    execute_template_slots,
    format_template_slots,
    parse_template_slots,
)


def example_program() -> PythonElectronProgram:
    return PythonElectronProgram(
        target_smiles="[CH3:1][OH:2]",
        steps=(
            PythonElectronStep(
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
                imports=("[Br-:3]",),
            ),
        ),
    )


def test_slots_round_trip_and_execution():
    slots = format_template_slots(example_program())
    assert slots.startswith("[step(")
    assert "mechanism" not in slots
    parsed = parse_template_slots(slots, target_smiles="[CH3:1][OH:2]")
    assert parsed == example_program()
    result = execute_template_slots(slots, target_smiles="[CH3:1][OH:2]")
    assert result.ok, result.diagnostics
    assert result.precursor_smiles == "[CH3:1][Br:3].[OH-:2]"


@pytest.mark.parametrize(
    "slots",
    [
        "__import__('os').system('id')",
        "[open('/tmp/x')]",
        "[step(move(bond(1,2),atom(2)),**x)]",
        "[]",
        "[unknown()]",
    ],
)
def test_slots_reject_code_and_invalid_schema(slots):
    with pytest.raises(ProofProgramError):
        parse_template_slots(slots, target_smiles="[CH3:1][OH:2]")


def test_slots_delta_round_trip():
    program = PythonElectronProgram(
        target_smiles="[CH3:1][OH:2]",
        steps=(
            PythonElectronStep(
                moves=(
                    {
                        "mode": "BE_DELTA",
                        "bond_deltas": [{"atoms": [1, 2], "delta": -1}],
                        "charge_actions": [
                            {"atom_map": 2, "q0": 0, "q1": -1}
                        ],
                    },
                )
            ),
        ),
    )
    slots = format_template_slots(program)
    assert parse_template_slots(slots, target_smiles=program.target_smiles) == program
