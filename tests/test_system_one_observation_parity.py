import json

import pytest

from mechet.in_place_grounded_flow import atom_token_spans
from mechet.trajectory_history import TrajectoryHistory
from scripts.audit_system_one_observation_parity import (
    audit_rows,
    chemically_equivalent_prompt,
    runtime_prompt,
)


def _annotate(smiles: str) -> str:
    pieces = []
    cursor = 0
    for index, (start, end) in enumerate(atom_token_spans(smiles), 1):
        pieces.extend((smiles[cursor:start], f"<A{index:02d}>", smiles[start:end]))
        cursor = end
    pieces.append(smiles[cursor:])
    return "".join(pieces)


def _prompt_lines(smiles: str, history: str = "HISTORY") -> str:
    return (
        f"CURRENT STATE SMILES: {smiles}\n"
        f"ANNOTATED CURRENT STATE: {_annotate(smiles)}\n"
        f"{history}"
    )


def test_stereo_text_equivalence_preserves_atom_addresses_and_history():
    left = "CN(C[C@H]1C[C@@H](OS(C)(=O)([O-])Cl)C1)C(=O)OC(C)(C)C"
    right = "CN(C[C@@H]1C[C@H](OS(C)(=O)([O-])Cl)C1)C(=O)OC(C)(C)C"
    assert chemically_equivalent_prompt(_prompt_lines(left), _prompt_lines(right))
    assert not chemically_equivalent_prompt(
        _prompt_lines(left), _prompt_lines(right, "CHANGED HISTORY")
    )
    assert not chemically_equivalent_prompt(
        _prompt_lines(left), _prompt_lines(right.replace("Cl)", "Br)"))
    )


def test_runtime_prompt_exactly_matches_minimal_accepted_trace():
    prompt = runtime_prompt("CCO", "CCO", TrajectoryHistory())
    row = {
        "id": "r1::finish",
        "source_id": "r1",
        "target_smiles": "CCO",
        "metadata": {"decision_index": 0},
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "tool_calls": [{
                "function": {"name": "finish_trace", "arguments": {}}
            }]},
            {"role": "tool", "name": "finish_trace", "content": json.dumps({
                "ok": True, "code": "PASS"
            })},
        ],
    }
    counts = audit_rows([row])
    assert counts["reactions"] == 1
    assert counts["decisions"] == 1
    assert counts["prompt_byte_exact"] == 1
    row["messages"][0]["content"] += " leaked"
    with pytest.raises(ValueError, match="runtime observation differs"):
        audit_rows([row])
