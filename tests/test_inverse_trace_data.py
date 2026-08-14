from mechet.forward_expert import verify_electron_step
from mechet.inverse_trace_data import (
    build_inverse_tool_sft_row,
    clear_atom_stereo,
    invert_moves,
    select_mapped_target,
    underdetermined_stereo_maps,
)


def _sn2_moves():
    return [
        {"source": {"kind": "LP", "atoms": [2]}, "sink": {"kind": "BOND", "atoms": [1, 2]}},
        {"source": {"kind": "BOND", "atoms": [1, 3]}, "sink": {"kind": "ATOM", "atoms": [3]}},
    ]


def test_inverse_moves_recover_precursor() -> None:
    product = "[Br-:3].[CH3:1][OH:2]"
    precursor = "[CH3:1][Br:3].[OH-:2]"
    result = verify_electron_step(product, invert_moves(_sn2_moves()))
    assert result["ok"]
    assert result["state_smiles"] == precursor
    assert set(invert_moves(_sn2_moves())[0]) == {"source", "sink", "electrons"}
    assert set(invert_moves(_sn2_moves())[0]["source"]) == {"kind", "atoms"}


def test_selects_structural_target_and_root_imports() -> None:
    target, imports, metadata = select_mapped_target(
        "[Br-:3].[CH3:1][OH:2]", "CO"
    )
    assert target == "[CH3:1][OH:2]"
    assert imports == ("[Br-:3]",)
    assert metadata["root_imports"] == 1


def test_builds_finish_trace_owned_inverse_supervision() -> None:
    row = {
        "id": "sn2",
        "source": "mech_uspto_31k",
        "split": "train",
        "initial_state": "[CH3:1][Br:3].[OH-:2]",
        "final_state": "[Br-:3].[CH3:1][OH:2]",
        "steps": [
            {
                "step_index": 0,
                "state_smiles": "[CH3:1][Br:3].[OH-:2]",
                "target_product": "[Br-:3].[CH3:1][OH:2]",
                "moves": _sn2_moves(),
            }
        ],
    }

    value = build_inverse_tool_sft_row(row, product_reference="CO")

    assert value["target_smiles"] == "[CH3:1][OH:2]"
    assert value["expected_precursor"] == "[CH3:1][Br:3].[OH-:2]"
    assert value["metadata"]["direction"] == "inverse"
    assert value["metadata"]["endpoint_source"] == "environment_owned_trace"
    assert any(
        message.get("name") == "finish_trace" for message in value["messages"]
    )


def test_normalizes_only_underdetermined_reaction_center_stereo() -> None:
    steps = [
        {
            "moves": [
                {
                    "source": {"kind": "BOND", "atoms": [1, 2]},
                    "sink": {"kind": "ATOM", "atoms": [2]},
                }
            ]
        }
    ]
    initial = "[C@H:1]([Cl:2])([CH3:3])[C@H:4]([F:5])[CH3:6]"
    final = "[CH+:1]([CH3:3])[C@H:4]([F:5])[CH3:6].[Cl-:2]"

    assert underdetermined_stereo_maps(initial, final, steps) == (1,)
    normalized = clear_atom_stereo(initial, (1,))
    assert "[CH:1]" in normalized
    assert "[C@H:4]" in normalized or "[C@@H:4]" in normalized
