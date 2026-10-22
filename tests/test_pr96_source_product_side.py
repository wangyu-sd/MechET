from copy import deepcopy
import json
from pathlib import Path

import pytest

from mechet.in_place_grounded_flow import mapped_atom_numbers


def fixture_rows():
    return json.loads((Path(__file__).parent / "fixtures/pr96_flower_inorganic_legacy_targets.json").read_text())


@pytest.mark.parametrize("index", [0, 1])
def test_flower_rechecks_complete_product_side_instead_of_rejecting_inorganic_legacy_target(index):
    from scripts.pr96_source_product_side import prepare_source_reference
    from rdkit import Chem
    source = fixture_rows()[index]
    untouched = deepcopy(source)
    result = prepare_source_reference(source)
    root = Chem.MolFromSmiles(result["target_smiles"])
    assert any(a.GetAtomicNum() == 6 for a in root.GetAtoms())
    assert all(a.GetAtomicNum() != 15 for a in root.GetAtoms())
    assert root.GetNumHeavyAtoms() == 12
    assert result["metadata"]["trace_plan"]["steps"] == source["metadata"]["trace_plan"]["steps"]
    assert result["full_precursor_state"] == source["full_precursor_state"]
    pieces = [result["target_smiles"], *result["metadata"]["trace_plan"]["initial_imports"]]
    union = set()
    for p in pieces:
        maps = mapped_atom_numbers(p)
        assert not union & maps
        union.update(maps)
    assert union == mapped_atom_numbers(source["metadata"]["trace_plan"]["steps"][0]["state_before"])
    assert result["metadata"]["product_side_definition"] == "flower_full_compiled_product_side_v1"
    assert source == untouched


def flower_row(side, target, imports):
    return {"source_id": "flower_mech_proof_train_fixture", "target_smiles": target,
            "metadata": {"original_proof_sha256": "fixture", "trace_plan": {
                "steps": [{"state_before": side, "state_after": side, "moves": []}],
                "initial_imports": imports}},
            "messages": [{"role": "assistant", "tool_calls": [{"function": {
                "name": "import_fragment", "arguments": {"fragment_smiles": v}}}]} for v in imports]}


def test_full_side_non_equivalent_tie_cannot_be_hidden_by_preselected_legacy_target():
    from scripts.pr96_source_product_side import prepare_source_reference
    row = flower_row("[CH3:1][OH:2].[CH3:3][NH2:4]", "[CH3:1][OH:2]", ["[CH3:3][NH2:4]"])
    with pytest.raises(ValueError, match="MAIN_PRODUCT_AMBIGUOUS"):
        prepare_source_reference(row)


def test_declared_carbon_rule_does_not_silently_exclude_organometallic_targets():
    from scripts.pr96_source_product_side import prepare_source_reference
    row = flower_row("[CH3:1][OH:2].[CH3:3][Mg:4][CH3:5]", "[CH3:3][Mg:4][CH3:5]", ["[CH3:1][OH:2]"])
    result = prepare_source_reference(row)
    assert mapped_atom_numbers(result["target_smiles"]) == {3, 4, 5}


def test_mech31k_annotated_product_side_and_first_use_contract_are_unchanged():
    from scripts.pr96_source_product_side import prepare_source_reference
    from scripts.build_main_product_training_data import audit_reference
    from mechet.inverse_trace_data import build_inverse_tool_sft_row
    from test_mech31k_main_product_root import forward_sn2
    source = build_inverse_tool_sft_row(forward_sn2(), product_reference="CO.[Br-]")
    result = prepare_source_reference(source)
    assert result["target_smiles"] == "[CH3:1][OH:2]"
    assert result["metadata"]["trace_plan"]["initial_imports"] == ["[Br-:3]"]
    assert result["metadata"]["original_product_side"] == source["target_smiles"]
    assert "product_side_definition" not in result["metadata"]
    assert result["metadata"]["trace_plan"]["steps"] == source["metadata"]["trace_plan"]["steps"]
    assert audit_reference(result)["audit"]["reference_endpoint_exact"] is True


def test_main_selection_cannot_use_reactant_endpoint_or_future_electron_steps():
    from scripts.pr96_source_product_side import prepare_source_reference
    row = fixture_rows()[0]
    baseline = prepare_source_reference(row)
    changed = deepcopy(row)
    changed["full_precursor_state"] = "not a smiles"
    changed["expected_precursor"] = "not a smiles"
    changed["metadata"]["trace_plan"]["steps"][1:] = []
    changed["metadata"]["trace_plan"]["steps"][0]["state_after"] = "not a smiles"
    changed["metadata"]["trace_plan"]["steps"][0]["moves"] = []
    selected = prepare_source_reference(changed)
    assert selected["target_smiles"] == baseline["target_smiles"]
    assert selected["metadata"]["trace_plan"]["initial_imports"] == baseline["metadata"]["trace_plan"]["initial_imports"]


@pytest.mark.parametrize("index", [0, 1])
def test_formal_builder_replays_repaired_real_references_to_original_gt_without_changed_arrows(index):
    from scripts.build_main_product_training_data import prepare_reference, audit_reference
    source = fixture_rows()[index]
    result = audit_reference(prepare_reference(source))
    assert result["audit"]["reference_endpoint_exact"] is True
    assert result["audit"]["original_moves_preserved"] is True
    assert result["inverse"]["metadata"]["trace_plan"]["steps"] == source["metadata"]["trace_plan"]["steps"]
    from scripts.finalize_pr96_reference import verify_source_identity
    verify_source_identity(source, result["inverse"])
