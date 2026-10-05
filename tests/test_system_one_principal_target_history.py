from __future__ import annotations

import copy

import pytest

from scripts.build_system_one_principal_target_history import (
    CONTRACT, convert_row, principal_component_from_mixture,
)
from scripts.audit_system_one_principal_target_history import check_row


def _row() -> dict:
    return {
        "id": "r::step0", "source_id": "r", "target_smiles": "C[C@H](O)F.[Cl-]",
        "expected_precursor": "CC(O)F",
        "messages": [
            {"role": "system", "content": "infer reverse chemistry"},
            {"role": "user", "content": (
                "TARGET PRODUCT SMILES: C[C@H](O)F.[Cl-]\n"
                "CURRENT STATE SMILES: C[C@H](O)F.[Cl-]\n"
                "ANNOTATED CURRENT STATE: <A01>C<A02>[C@H](<A03>O)<A04>F.<A05>[Cl-]"
            )},
            {"role": "assistant", "tool_calls": [{"function": {"name": "finish_trace"}}]},
            {"role": "tool", "content": '{"ok":true}'},
        ],
        "metadata": {"reaction_id": "r", "decision_index": 0},
    }


def test_principal_component_keeps_strict_view_stereo():
    principal, symmetric = principal_component_from_mixture(
        "C[C@H](O)F.[Cl-]", "CC(O)F"
    )
    assert principal == "C[C@H](O)F"
    assert symmetric is False


def test_equivalent_duplicate_components_allow_any_one():
    principal, symmetric = principal_component_from_mixture("CO.CO", "CO")
    assert principal == "CO" and symmetric


def test_prompt_conversion_preserves_actions_and_executor_state():
    source = _row()
    original = copy.deepcopy(source)
    converted = convert_row(source, "CC(O)F")
    assert source == original
    assert converted["target_smiles"] == source["target_smiles"]
    assert converted["messages"][1]["content"].startswith(
        "TARGET PRODUCT SMILES: CC(O)F\n"
        "CURRENT STATE SMILES: C[C@H](O)F.[Cl-]\n"
    )
    assert converted["messages"][0] == source["messages"][0]
    assert converted["messages"][2:] == source["messages"][2:]
    assert converted["metadata"]["target_prompt_contract"] == CONTRACT


def test_prompt_conversion_rejects_mismatched_source_target():
    source = _row()
    source["messages"][1]["content"] = source["messages"][1]["content"].replace(
        "TARGET PRODUCT SMILES: C[C@H](O)F.[Cl-]",
        "TARGET PRODUCT SMILES: CO",
    )
    with pytest.raises(ValueError, match="source target prompt differs"):
        convert_row(source, "CC(O)F")


def test_independent_audit_rejects_changed_executor_response():
    source = _row()
    converted = convert_row(source, "CC(O)F")
    check_row(source, converted, "CC(O)F")
    converted["messages"][-1]["content"] = '{"ok":false}'
    with pytest.raises(ValueError, match="action, executor state"):
        check_row(source, converted, "CC(O)F")


def test_independent_audit_rejects_stereo_leaking_target_line():
    source = _row()
    converted = convert_row(source, "C[C@H](O)F")
    with pytest.raises(ValueError, match="frozen endpoint input"):
        check_row(source, converted, "CC(O)F")
