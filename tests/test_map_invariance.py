from mechet.map_invariance import record_map_permutation, remap_proof_text, remap_smiles


def test_remap_is_consistent_for_product_and_imported_atoms():
    product = "[CH3:1][OH:2]"
    proof = '''<proof>\nMECH_PROOF v1\nTARGET_SMILES "[CH3:1][OH:2]"\nROOT s0\n  IMPORT "[Br-:3]"\nPRECURSOR_STATE s1\nEDGE s0 s1\n  BOND 1 2 -1\n  BOND 1 3 +1\n  LP 2 +2\n  LP 3 -2\n  CHARGE 2 0 -1\n  CHARGE 3 -1 0\n</proof>'''
    mapping = record_map_permutation(product=product, proof=proof, precursor="[CH3:1][Br:3].[OH-:2]", seed=4)
    assert set(mapping) == {1, 2, 3}
    remapped = remap_proof_text(proof, mapping)
    assert remap_smiles(product, mapping) in remapped
    assert f"BOND {mapping[1]} {mapping[3]} +1" in remapped
    assert f"LP {mapping[3]} -2" in remapped
