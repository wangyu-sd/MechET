"""New main-only supervision preserves electronic actions, not just endpoints."""
from copy import deepcopy

import pytest

from mechet.inverse_trace_data import build_inverse_tool_sft_row
from scripts import build_natural_language_event_sft as nl
from test_mech31k_main_product_root import forward_sn2


def legacy_sn2():
    return build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]")


def test_retarget_legacy_trace_withholds_product_counterion_without_changing_actions():
    from scripts.build_main_product_training_data import prepare_reference, audit_reference
    legacy = legacy_sn2()
    frozen = deepcopy(legacy)
    corrected = prepare_reference(legacy)
    assert corrected["target_smiles"] == "[CH3:1][OH:2]"
    assert corrected["metadata"]["trace_plan"]["initial_imports"] == ["[Br-:3]"]
    assert corrected["metadata"]["trace_plan"]["steps"] == frozen["metadata"]["trace_plan"]["steps"]
    assert corrected["full_precursor_state"] == frozen["full_precursor_state"]
    assert legacy == frozen
    result = audit_reference(corrected)
    assert result["audit"]["first_tool"] == "import_fragments"
    assert result["audit"]["original_moves_preserved"]
    assert result["audit"]["flow_successors_match_source"] == 1
    assert result["audit"]["reference_endpoint_exact"]
    assert [r["metadata"]["decision_type"] for r in result["decisions"]] == ["import", "event", "finish"]


def test_reference_audit_rejects_changed_intermediate_even_if_endpoint_is_unchanged():
    from scripts.build_main_product_training_data import prepare_reference, audit_reference
    row = prepare_reference(legacy_sn2())
    row["metadata"]["trace_plan"]["steps"][0]["state_after"] = "[CH3:1][F:3].[OH-:2]"
    with pytest.raises(ValueError, match="SUCCESSOR_MISMATCH"):
        audit_reference(row)


def test_strict_converter_never_invokes_graph_delta_replacement(monkeypatch):
    # A wrong reference successor must not cause different supervised arrows.
    row = build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]", main_product_only=True)
    row["metadata"]["trace_plan"]["steps"][0]["state_after"] = "[CH3:1][F:3].[OH-:2]"
    def forbidden(*args):
        pytest.fail("strict conversion attempted to replace source arrows")
    monkeypatch.setattr(nl, "_raw_graph_delta", forbidden)
    rows = nl.convert_row(row, preserve_source_moves=True)
    assert rows[1]["metadata"]["source_electron_moves_preserved"] is True


def test_full_denominator_gate_rejects_missing_or_unaccounted_reactions():
    from scripts.build_main_product_training_data import validate_coverage
    validate_coverage(original=3, accepted={"a", "b"}, unresolved={"c"})
    with pytest.raises(ValueError, match="DENOMINATOR"):
        validate_coverage(original=3, accepted={"a"}, unresolved={"c"})
    with pytest.raises(ValueError, match="OVERLAP"):
        validate_coverage(original=3, accepted={"a", "b"}, unresolved={"b"})


def test_main_only_does_not_force_root_import_or_import_future_species_early():
    from scripts.build_main_product_training_data import prepare_reference, audit_reference
    initial = "[CH:1](=[O:2])[CH2:4][Br:3].[OH-:5]"
    middle = "[CH:1](=[O:2])[CH2:4][OH:5].[Br-:3]"
    final = "[CH+:1]([O-:2])[CH2:4][OH:5].[Br-:3]"
    forward = {
        "id": "deferred_br", "split": "valid", "initial_state": initial,
        "final_state": final, "steps": [
            {"step_index": 0, "state_smiles": initial, "target_product": middle,
             "moves": [
                 {"source": {"kind": "LP", "atoms": [5]}, "sink": {"kind": "BOND", "atoms": [4, 5]}},
                 {"source": {"kind": "BOND", "atoms": [4, 3]}, "sink": {"kind": "ATOM", "atoms": [3]}}]},
            {"step_index": 1, "state_smiles": middle, "target_product": final,
             "moves": [{"source": {"kind": "BOND", "atoms": [1, 2]},
                        "sink": {"kind": "ATOM", "atoms": [2]}}]},
        ],
    }
    row = build_inverse_tool_sft_row(forward, product_reference="[O-][CH+]CO.[Br-]")
    result = audit_reference(prepare_reference(row))
    calls = [d["messages"][2]["tool_calls"][0]["function"] for d in result["decisions"]]
    assert [c["name"] for c in calls] == ["apply_electron_flow", "import_fragments", "apply_electron_flow", "finish_trace"]
    assert "Br" not in result["decisions"][0]["messages"][1]["content"]
    assert calls[1]["arguments"]["fragments"] == [{"smiles": "[Br-]", "count": 1, "purpose": "electron_participant"}]
