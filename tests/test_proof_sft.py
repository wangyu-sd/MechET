from mechet.proof_program import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
)
from mechet.proof_sft import convert_mech_et_row_to_proof_sft


def test_conversion_removes_answer_and_intermediate_states(monkeypatch):
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
    monkeypatch.setattr(
        "mechet.proof_sft.compile_mech_et_body",
        lambda _body: program,
    )
    row = {
        "id": "flower_mech_et_train_1",
        "messages": [
            {"role": "system", "content": "old"},
            {
                "role": "user",
                "content": "TARGET: [CH3:1][OH:2]\nold instruction",
            },
            {
                "role": "assistant",
                "content": (
                    "<mechanism>\nMECH_ET v3\n...</mechanism>\n"
                    "<answer>\n[CH3:1][Br:3].[OH-:2]\n</answer>"
                ),
            },
        ],
        "task_type": "mech_et_cot_retro",
        "metadata": {
            "initial_reactants": "[CH3:1][Br:3].[OH-:2]"
        },
    }
    converted = convert_mech_et_row_to_proof_sft(row)
    assistant = converted["messages"][-1]["content"]
    assert converted["task_type"] == "mech_proof_retro"
    assert "<proof>" in assistant
    assert "<answer>" not in assistant
    assert "STATE s0" not in assistant
    assert converted["metadata"]["answer_channel"] == "executor_derived"
