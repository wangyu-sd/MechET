from scripts.build_reverse_et_transfer_supervision import (
    endpoint_example,
    net_edit_target,
    reverse_et_target,
)


ROW = {
    "id": "rxn-1",
    "source_id": "1",
    "target_smiles": "[CH3:1][OH:2]",
    "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
    "metadata": {
        "source_dataset": "toy",
        "source_split": "train",
        "trace_plan": {
            "initial_imports": ["[Br-:3]"],
            "steps": [
                {
                    "imports": [],
                    "moves": [
                        {
                            "source": {"kind": "BOND", "atoms": [1, 2]},
                            "sink": {"kind": "ATOM", "atoms": [2]},
                        },
                        {
                            "source": {"kind": "ATOM", "atoms": [3]},
                            "sink": {"kind": "BOND", "atoms": [1, 3]},
                        },
                    ],
                }
            ],
        },
    },
}


def test_endpoint_example_contains_answer():
    row = endpoint_example(ROW)
    assert row["messages"][-1]["content"] == (
        "<answer>\n[CH3:1][Br:3].[OH-:2]\n</answer>"
    )


def test_net_edit_aux_has_no_precursor_answer():
    target = net_edit_target(ROW)
    assert "NET_EDIT_AUX v1" in target
    assert "IMPORT [Br-:3]" in target
    assert "<answer>" not in target
    assert "[CH3:1][Br:3].[OH-:2]" not in target


def test_reverse_et_aux_has_direction_and_no_answer():
    target = reverse_et_target(ROW)
    assert "REVERSE_ET_AUX v1" in target
    assert "BOND(1,2) -> ATOM(2)" in target
    assert "ATOM(3) -> BOND(1,3)" in target
    assert "<answer>" not in target
    assert "[CH3:1][Br:3].[OH-:2]" not in target
