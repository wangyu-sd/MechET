from rdkit import Chem

from mechet.forward_data import normalize_reaction_row
from mechet.forward_expert import ElectronMove, verify_electron_step


def _maps(smiles: str) -> list[int]:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    assert mol is not None
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def test_mech_uspto_partial_maps_are_completed_and_target_is_executed():
    row = normalize_reaction_row(
        {
            "rxn_idx": 7,
            "step_idx_forward": 0,
            "split": "train",
            "mech_smi_ori": (
                "CC[CH2:1][Br:3].[OH-:2]|(2, 1);((1, 3), 3)"
            ),
            "elem_reac_ori": "CCCBr.[OH-]",
            "elem_prod_ori": "CCCO.[Br-]",
        },
        source="mech_uspto_31k",
        row_index=0,
    )

    step = row["steps"][0]
    assert row["metadata"]["mapped_target_source"] == (
        "formal_electron_step_execution"
    )
    assert row["metadata"]["map_completion"]["completed"]
    assert row["metadata"]["reference_reactant_match"]
    assert row["metadata"]["reference_product_match"]

    for smiles in (step["state_smiles"], step["target_product"]):
        maps = _maps(smiles)
        assert all(value > 0 for value in maps)
        assert len(maps) == len(set(maps))

    replay = verify_electron_step(step["state_smiles"], step["moves"])
    assert replay["ok"]
    assert replay["state_smiles"] == step["target_product"]


def test_mech_uspto_explicit_mapped_hydrogen_is_preserved():
    row = normalize_reaction_row(
        {
            "rxn_idx": 8,
            "step_idx_forward": 0,
            "split": "train",
            "mech_smi_ori": (
                "C=[O:1].[H:2][Br:3]|(1, 2);((2, 3), 3)"
            ),
            "elem_reac_ori": "C=O.Br",
            "elem_prod_ori": "C=[OH+].[Br-]",
        },
        source="mech_uspto_31k",
        row_index=0,
    )
    step = row["steps"][0]
    move_ids = [ElectronMove.parse(value).id for value in step["moves"]]
    assert "LP:1->BOND:1,2/2e" in move_ids
    assert "BOND:2,3->ATOM:3/2e" in move_ids
    assert 2 in _maps(step["state_smiles"])
    assert row["metadata"]["reference_product_match"]
