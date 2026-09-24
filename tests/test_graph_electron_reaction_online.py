import json
from pathlib import Path

from train_graph_electron_reaction_online import (
    ReactionOnlineDataset,
    build_offset_index,
    collate_reactions,
)


def source_row(index: int) -> dict:
    return {
        "id": f"reaction-{index}",
        "expected_precursor": "[Br-:2].[CH3:1][O:3][CH3:4].[Na+:5]",
        "metadata": {
            "trace_plan": {
                "target_smiles": "[CH3:1][Br:2]",
                "initial_imports": ["[O-:3][CH3:4]", "[Na+:5]"],
                "steps": [
                    {
                        "state_after": "[Br-:2].[CH3:1][O:3][CH3:4].[Na+:5]",
                        "imports": [],
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
            }
        },
    }


def test_reaction_dataset_expands_only_in_memory(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(source_row(i)) + "\n" for i in range(3)))
    offsets = tmp_path / "train.reaction_offsets.u64"
    build_offset_index(source, offsets, 3)
    dataset = ReactionOnlineDataset(
        source, offsets, rank=0, world_size=1, seed=17
    )
    first = dataset[0]
    assert [item.value["kind"] for item in first] == [
        "IMPORT_REACTIVE",
        "FLOW",
        "IMPORT_ENV",
        "FINISH",
    ]
    assert not list(tmp_path.glob("*.rank*.jsonl"))
    assert len(collate_reactions([first, dataset[1]])) == 8


def test_reaction_dataset_shards_whole_reactions(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(source_row(i)) + "\n" for i in range(5)))
    offsets = tmp_path / "train.reaction_offsets.u64"
    build_offset_index(source, offsets, 5)
    left = ReactionOnlineDataset(source, offsets, rank=0, world_size=2, seed=3)
    right = ReactionOnlineDataset(source, offsets, rank=1, world_size=2, seed=3)
    left_ids = {batch[0].value["reaction_id"] for batch in map(left.__getitem__, range(len(left)))}
    right_ids = {batch[0].value["reaction_id"] for batch in map(right.__getitem__, range(len(right)))}
    assert not (left_ids & right_ids)
    assert len(left_ids | right_ids) == 5
    resumed = ReactionOnlineDataset(
        source, offsets, rank=0, world_size=2, seed=3, skip_reactions=1
    )
    assert len(resumed) == len(left) - 1
    limited = ReactionOnlineDataset(
        source,
        offsets,
        rank=0,
        world_size=1,
        seed=3,
        reaction_limit_per_rank=2,
    )
    assert len(limited) == 2
