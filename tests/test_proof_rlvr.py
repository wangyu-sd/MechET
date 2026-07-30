from mechet.proof_program import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
    format_proof_output,
)
from mechet.proof_sft import MECH_PROOF_SYSTEM_PROMPT
from mechet.rlvr import (
    build_assistant_completion,
    compute_mechvr_reward,
    compute_rollout_reward,
    extract_prompt_messages,
)


def proof_row() -> dict:
    return {
        "id": "proof-1",
        "task_type": "mech_proof_retro",
        "messages": [
            {
                "role": "system",
                "content": MECH_PROOF_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "TARGET: [CH3:1][OH:2]\nprove it",
            },
        ],
        "metadata": {
            "derived_precursor": "[CH3:1][Br:3].[OH-:2]"
        },
    }


def proof_text() -> str:
    return format_proof_output(
        ProofProgram(
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
    )


def test_proof_reward_uses_execution_and_endpoint():
    scored = compute_rollout_reward(proof_row(), proof_text())
    assert scored["gate_ok"]
    assert scored["endpoint_exact"]
    assert scored["rlvr_total"] == 7.0
    assert scored["reward_mode"] == "mech_proof_v1"


def test_existing_trainer_reward_entrypoint_dispatches_proof_rows():
    scored = compute_mechvr_reward(proof_row(), proof_text())
    assert scored["gate_ok"]
    assert scored["reward_mode"] == "mech_proof_v1"


def test_proof_system_prompt_is_preserved():
    _product, messages = extract_prompt_messages(proof_row())
    assert messages[0]["content"] == MECH_PROOF_SYSTEM_PROMPT


def test_proof_completion_normalization_does_not_add_answer():
    normalized = build_assistant_completion({}, proof_text())
    assert normalized.startswith("<proof>")
    assert "<answer>" not in normalized
