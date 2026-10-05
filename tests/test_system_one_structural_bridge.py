from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.in_place_grounded_flow import deterministic_unmapped_state
from scripts.score_system_one_structural_bridge import (
    append_private_import, initialize_provenance,
)


def test_product_origin_survives_auxiliary_context_and_import():
    mapped, product, next_map = initialize_provenance("CCO", "CCO.[Cl-]")
    first = split_precursor_endpoints(mapped, product)
    assert structural_exact(first.structural, "CCO")
    assert len(first.auxiliary) == 1
    mapped, next_map = append_private_import(mapped, [["[Na+]", 1]], next_map)
    after = split_precursor_endpoints(mapped, product)
    assert structural_exact(after.structural, "CCO")
    assert len(after.auxiliary) == 2
    assert deterministic_unmapped_state(mapped).text == "CCO.[Cl-].[Na+]"
    assert next_map == 6
