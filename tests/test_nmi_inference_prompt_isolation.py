"""The H2 test file may retain reference answers for scoring, not prompting."""

from scripts.infer_mechet import _direct_messages, _trace_messages


def test_h2_direct_and_trace_prompts_do_not_use_reference_continuation():
    target = "[CH3:1][OH:2]"
    poison = "REFERENCE_ANSWER_MUST_NOT_APPEAR_IN_MODEL_PROMPT"
    row = {
        "target_smiles": target,
        "structural_precursor": poison,
        "metadata": {"compiled_proof": poison, "trace_plan": [{"secret": poison}]},
        "messages": [
            {"role": "system", "content": "Predict reactants from the product."},
            {"role": "user", "content": f"TARGET: {target}"},
            {"role": "assistant", "content": poison},
            {"role": "tool", "name": "finish_trace", "content": poison},
        ],
    }

    direct = _direct_messages(row)
    assert [message["role"] for message in direct] == ["system", "user"]
    assert all(poison not in message["content"] for message in direct)

    trace = _trace_messages(
        row, "trace", '{"task":"trace_owned_inverse_electron_flow","max_tool_calls":40}',
        prompt_source="runtime",
    )
    assert [message["role"] for message in trace] == ["system", "user"]
    assert all(poison not in message["content"] for message in trace)
    assert f"TARGET: {target}" in trace[1]["content"]
