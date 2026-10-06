from scripts.classify_reliable_mapping_failures import (
    heavy_atom_skeleton, probe_kekule_branches,
)


def test_heavy_atom_skeleton_ignores_bond_order_charge_and_atom_maps():
    assert heavy_atom_skeleton("[CH2:1]=[CH:2][CH3:3]") == heavy_atom_skeleton(
        "[CH3:8][CH:7]=[CH2:6]"
    )
    assert heavy_atom_skeleton("[NH3+]C") == heavy_atom_skeleton("NC")


def test_heavy_atom_skeleton_preserves_element_connectivity():
    assert heavy_atom_skeleton("CCO") != heavy_atom_skeleton("COC")
    assert heavy_atom_skeleton("C1CC1") != heavy_atom_skeleton("CCC")


def test_audit_only_kekule_variants_expose_aromatic_action_ambiguity():
    from rdkit import Chem
    from mechet.forward_expert import verify_electron_step
    from mechet.natural_language_electron_flow import render_event_arguments
    from scripts.run_natural_language_value_search import visible

    state = "[cH:1]1[n:2][cH:3][cH:4][cH:5][cH:6]1.[Cl-:7]"
    moves = [
        {"source": {"kind": "BOND", "atoms": [1, 2]},
         "sink": {"kind": "ATOM", "atoms": [2]}, "electrons": 2},
        {"source": {"kind": "LP", "atoms": [7]},
         "sink": {"kind": "BOND", "atoms": [1, 7]}, "electrons": 2},
    ]
    variants = list(Chem.ResonanceMolSupplier(
        Chem.MolFromSmiles(state), Chem.ResonanceFlags.KEKULE_ALL,
    ))
    outputs = [
        verify_electron_step(state, moves, _prepared_kekule_mol=variant)
        for variant in variants
    ]
    assert len(outputs) == 2 and all(item["ok"] for item in outputs)
    assert verify_electron_step(state, moves) == outputs[0]
    assert heavy_atom_skeleton(outputs[0]["state_smiles"]) != heavy_atom_skeleton(
        outputs[1]["state_smiles"]
    )
    action = render_event_arguments(state, moves)
    for item in outputs:
        probe = probe_kekule_branches(
            state, action, visible(item["state_smiles"]), max_structures=2,
        )
        assert probe["reference_reachable"] is True
    invalid = verify_electron_step(
        state, moves, _prepared_kekule_mol=Chem.MolFromSmiles("[cH:1]1ccccc1"),
    )
    assert not invalid["ok"] and "changes atoms" in invalid["message"]
