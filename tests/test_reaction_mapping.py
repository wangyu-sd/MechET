import pytest
from rdkit import Chem

from mechet.reaction_mapping import (
    assert_product_contained_in_reference,
    product_only_reindex_reaction,
)


def _maps(smiles: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def test_product_only_reindex_is_invariant_to_source_map_numbers():
    first = product_only_reindex_reaction(
        "[CH3:91][Br:33].[OH-:17].[Na+]>>[CH3:91][OH:17]"
    )
    second = product_only_reindex_reaction(
        "[CH3:4][Br:8].[OH-:207].[Na+]>>[CH3:4][OH:207]"
    )
    assert first.reaction_smiles == second.reaction_smiles
    assert first.products_unmapped == "CO"
    assert set(_maps(first.products)) == {1, 2}
    assert all(value > 0 for value in _maps(first.reactants))
    assert len(first.auxiliary_fragments) == 1
    assert "Na" in first.auxiliary_fragments[0]


def test_product_requires_complete_cross_side_mapping():
    with pytest.raises(ValueError, match="every product atom must"):
        product_only_reindex_reaction("[CH3:1][Br:2].[OH-:3]>>[CH3:1]O")
    with pytest.raises(ValueError, match="must occur on the reactant side"):
        product_only_reindex_reaction(
            "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:99]"
        )


def test_original_product_must_be_in_mechanism_final_mixture():
    mapped = product_only_reindex_reaction(
        "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3]"
    )
    assert_product_contained_in_reference(mapped.products, "CO.[Br-]")
    with pytest.raises(ValueError, match="absent from rxn_prod_min"):
        assert_product_contained_in_reference(mapped.products, "CC.[Br-]")
