"""Recheck the actual product side, not a legacy preselected FlowER component.

The declared rule stays unchanged: the unique largest carbon-containing
component by heavy-atom count. Organometallics are not silently excluded. No
reactant endpoint or future electronic action participates in root selection.
"""
from copy import deepcopy
import hashlib
import json

from mechet.in_place_grounded_flow import extract_import_fragments, mapped_atom_numbers
from mechet.inverse_trace_data import select_mapped_target


def product_side(source):
    if str(source["source_id"]).startswith("flower_mech_proof_"):
        if not source["metadata"].get("original_proof_sha256"):
            raise ValueError("FLOWER_PRODUCT_SIDE_SOURCE_PROVENANCE_MISSING")
        plan = source["metadata"]["trace_plan"]
        # The first electronic state can already be projected for first-use
        # import scheduling. The compiled INITIAL product inventory, not that
        # active prefix or any later state/action, defines the complete side.
        side = ".".join([str(source["target_smiles"]), *plan["initial_imports"]])
        inventory = set(mapped_atom_numbers(source["target_smiles"]))
        for fragment in extract_import_fragments(source):
            maps = mapped_atom_numbers(fragment)
            if inventory & maps:
                raise ValueError("IMPORT_MAP_OVERLAP")
            inventory.update(maps)
        if (inventory != mapped_atom_numbers(side)
                or not mapped_atom_numbers(plan["steps"][0]["state_before"]) <= inventory):
            raise ValueError("FLOWER_PRODUCT_SIDE_ATOM_INVENTORY_MISMATCH")
        return side, "flower_full_compiled_product_side_v1"
    return str(source["target_smiles"]), "annotated_product_side_v1"


def prepare_source_reference(source):
    row = deepcopy(source)
    original_target = str(source["target_smiles"])
    side, definition = product_side(source)
    target, removed, selection = select_mapped_target(side, side, main_product_only=True)
    original_imports = extract_import_fragments(source)
    plan = row["metadata"]["trace_plan"]
    plan["target_smiles"] = target
    # A FlowER legacy target is already a single preselected component; it can
    # itself be inorganic or a catalyst. Repartition the complete recorded
    # product side instead of retaining that old selection implicitly.
    plan["initial_imports"] = (list(removed) if definition == "flower_full_compiled_product_side_v1"
                               else original_imports + list(removed))
    present = set(mapped_atom_numbers(target))
    for fragment in plan["initial_imports"]:
        maps = mapped_atom_numbers(fragment)
        if maps & present:
            raise ValueError("IMPORT_MAP_OVERLAP")
        present.update(maps)
    first_state_maps = mapped_atom_numbers(plan["steps"][0]["state_before"])
    if not first_state_maps <= present:
        raise ValueError("SOURCE_ATOMS_NOT_ACCOUNTED_FOR")
    row["target_smiles"] = target
    row["metadata"]["target_selection"] = selection
    row["metadata"]["input_condition"] = "main_product_only_v1"
    row["metadata"]["original_product_side"] = side
    row["metadata"]["removed_product_components"] = list(removed)
    if definition == "flower_full_compiled_product_side_v1":
        row["metadata"]["product_side_definition"] = definition
        row["metadata"]["legacy_selected_target"] = original_target
    row["metadata"]["source_row_sha256"] = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    row["metadata"]["n_trace_imports"] = len(plan["initial_imports"])
    row["artifact_type"] = "private_reference_plan_not_training_conversation"
    row["messages"] = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": f"reference_import_{i}", "type": "function",
        "function": {"name": "import_fragment", "arguments": {"fragment_smiles": f}},
    }]} for i, f in enumerate(plan["initial_imports"])]
    return row
