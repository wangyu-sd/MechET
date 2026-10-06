from scripts.classify_reliable_mapping_failures import heavy_atom_skeleton


def test_heavy_atom_skeleton_ignores_bond_order_charge_and_atom_maps():
    assert heavy_atom_skeleton("[CH2:1]=[CH:2][CH3:3]") == heavy_atom_skeleton(
        "[CH3:8][CH:7]=[CH2:6]"
    )
    assert heavy_atom_skeleton("[NH3+]C") == heavy_atom_skeleton("NC")


def test_heavy_atom_skeleton_preserves_element_connectivity():
    assert heavy_atom_skeleton("CCO") != heavy_atom_skeleton("COC")
    assert heavy_atom_skeleton("C1CC1") != heavy_atom_skeleton("CCC")
