from mechet.forward_expert import verify_electron_step
from mechet.trace_stitching import complete_reaction_ids, stitch_steps


def test_selects_only_reactions_with_every_raw_step() -> None:
    raw_ids = ["10", "10", "20"]
    standardized = [
        {"id": "10", "steps": [{"step_index": 0}]},
        {"id": "20", "steps": [{"step_index": 0}]},
    ]

    assert complete_reaction_ids(raw_ids, standardized) == [
        {
            "reaction_id": "20",
            "expected_steps": 1,
            "standardized_steps": 1,
        }
    ]


def test_stitches_local_maps_and_replays_every_step() -> None:
    steps = [
        {
            "step_index": 0,
            "state_smiles": "[CH3:1][Br:3].[OH-:2]",
            "target_product": "[Br-:3].[CH3:1][OH:2]",
            "moves": [
                {"source": {"kind": "LP", "atoms": [2]}, "sink": {"kind": "BOND", "atoms": [1, 2]}},
                {"source": {"kind": "BOND", "atoms": [1, 3]}, "sink": {"kind": "ATOM", "atoms": [3]}},
            ],
        },
        {
            "step_index": 1,
            "state_smiles": "[Br-:7].[CH3:5][OH:9]",
            "target_product": "[Br-:7].[CH3+:5].[OH-:9]",
            "moves": [
                {"source": {"kind": "BOND", "atoms": [5, 9]}, "sink": {"kind": "ATOM", "atoms": [9]}},
            ],
        },
    ]

    stitched, metadata = stitch_steps(steps)

    assert stitched[1]["state_smiles"] == stitched[0]["target_product"]
    assert stitched[1]["moves"][0]["source"]["atoms"] == [1, 2]
    assert stitched[1]["moves"][0]["source"]["id"] == "BOND:1,2"
    assert stitched[1]["moves"][0]["id"].startswith("BOND:1,2->ATOM:2")
    assert verify_electron_step(
        stitched[1]["state_smiles"], stitched[1]["moves"]
    )["ok"]
    assert metadata["links"] == 1


def test_preserves_explicit_mapped_hydrogen_during_stitching() -> None:
    steps = [
        {
            "step_index": 0,
            "state_smiles": "[O-:1][CH3:2].[H:3][Br:4]",
            "target_product": "[CH3:2][O:1][H:3].[Br-:4]",
            "moves": [
                {"source": {"kind": "LP", "atoms": [1]}, "sink": {"kind": "BOND", "atoms": [1, 3]}},
                {"source": {"kind": "BOND", "atoms": [3, 4]}, "sink": {"kind": "ATOM", "atoms": [4]}},
            ],
        },
        {
            "step_index": 1,
            "state_smiles": "[CH3:8][O:6][H:5].[Br-:7]",
            "target_product": "[CH3+:8].[O-:6][H:5].[Br-:7]",
            "moves": [
                {"source": {"kind": "BOND", "atoms": [8, 6]}, "sink": {"kind": "ATOM", "atoms": [6]}},
            ],
        },
    ]

    stitched, _ = stitch_steps(steps)

    assert stitched[1]["state_smiles"] == stitched[0]["target_product"]
    assert all("[H:" in step["state_smiles"] or step["step_index"] for step in stitched)
