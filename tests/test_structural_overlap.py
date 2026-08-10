from rdkit import Chem

from mechet.structural_overlap import (
    annotate_rows_with_overlap,
    audit_structural_overlap,
    canonical_unmapped_smiles,
    reaction_center_context_signature,
)


def row(row_id: str, target: str, precursor: str, family: str) -> dict:
    return {
        "id": row_id,
        "target_smiles": target,
        "expected_precursor": precursor,
        "metadata": {
            "reaction_family": family,
            "trace_plan": {
                "target_smiles": target,
                "expected_precursor": precursor,
                "initial_imports": [],
                "steps": [
                    {
                        "step_index": 0,
                        "state_before": target,
                        "state_after": precursor,
                        "imports": [],
                        "moves": [
                            {
                                "source": {"kind": "LP", "atoms": [2]},
                                "sink": {"kind": "BOND", "atoms": [1, 3]},
                                "electrons": 2,
                            }
                        ],
                    }
                ],
            },
        },
    }


def imported_center_row() -> dict:
    target = "[CH3:1][OH:2]"
    precursor = "[CH3:1][Cl:3].[OH-:2]"
    return {
        "id": "imported-center",
        "target_smiles": target,
        "expected_precursor": precursor,
        "metadata": {
            "reaction_family": "substitution",
            "trace_plan": {
                "target_smiles": target,
                "expected_precursor": precursor,
                "initial_imports": [],
                "steps": [
                    {
                        "step_index": 0,
                        "state_before": target,
                        "state_after": precursor,
                        "imports": ["[Cl-:3]"],
                        "moves": [
                            {
                                "source": {"kind": "LP", "atoms": [3]},
                                "sink": {"kind": "BOND", "atoms": [1, 3]},
                                "electrons": 2,
                            },
                            {
                                "source": {"kind": "BOND", "atoms": [1, 2]},
                                "sink": {"kind": "ATOM", "atoms": [2]},
                                "electrons": 2,
                            },
                        ],
                    }
                ],
            },
        },
    }


def test_canonical_unmapped_smiles_removes_map_labels():
    assert canonical_unmapped_smiles("[CH3:1][OH:2]") == "CO"


def test_reaction_center_context_uses_step_imports_not_only_target_atoms():
    value = imported_center_row()
    mol = Chem.MolFromSmiles(value["target_smiles"])
    assert mol is not None
    target_maps = {atom.GetAtomMapNum() for atom in mol.GetAtoms()}
    assert 3 not in target_maps
    signature = reaction_center_context_signature(value)
    assert len(signature) == 64


def test_structural_overlap_reports_exact_scaffold_center_and_similarity():
    train = [
        row(
            "train-1",
            "[CH3:1][OH:2].[Cl-:3]",
            "[CH3:1][Cl:3].[OH-:2]",
            "substitution",
        )
    ]
    heldout = [
        row(
            "test-1",
            "[CH3:1][NH2:2].[Cl-:3]",
            "[CH3:1][Cl:3].[NH2-:2]",
            "substitution",
        )
    ]
    report = audit_structural_overlap(
        train,
        heldout,
        similarity_threshold=0.2,
        compute_similarity=True,
        max_near_duplicate_rate=1.0,
    )
    assert report["n_heldout"] == 1
    assert report["exact_reaction_overlap_count"] == 0
    assert report["family_seen_count"] == 1
    assert report["reaction_center_context_definition"] == "step_state_plus_edge_imports_v2"
    assert report["maximum_product_tanimoto_summary"] is not None
    assert "test-1" in report["row_annotations"]


def test_annotations_are_written_to_split_metadata():
    rows = [
        row(
            "test-1",
            "[CH3:1][NH2:2].[Cl-:3]",
            "[CH3:1][Cl:3].[NH2-:2]",
            "substitution",
        )
    ]
    output = annotate_rows_with_overlap(
        rows,
        {"test-1": {"murcko_scaffold_seen_in_train": False}},
    )
    assert output[0]["metadata"]["mechcomp_structural_overlap"][
        "murcko_scaffold_seen_in_train"
    ] is False
