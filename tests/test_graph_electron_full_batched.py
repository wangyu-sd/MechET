import json

from torch.utils.data import DataLoader

from train_graph_electron_full_batched import (
    JsonlDecisionDataset,
    TensorChunkDecisionDataset,
    collate_prepared,
    write_tensor_chunk,
)


def test_torch_dataloader_spawn_roundtrip(tmp_path):
    decision = {
        "reaction_id": "spawn-smoke",
        "kind": "FLOW",
        "current": "[CH3:1][Br:2].[O-:3]",
        "target": "[CH3:1][Br:2]",
        "moves": [
            {
                "source": {"kind": "LP", "atoms": [3]},
                "sink": {"kind": "BOND", "atoms": [1, 3]},
                "electrons": 2,
            }
        ],
    }
    path = tmp_path / "rank.jsonl"
    path.write_text("".join(json.dumps(decision) + "\n" for _ in range(5)))
    dataset = JsonlDecisionDataset(path)
    batches = list(
        DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            collate_fn=collate_prepared,
            multiprocessing_context="spawn",
            persistent_workers=False,
            prefetch_factor=2,
        )
    )
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert all(item.current_graph.atoms.ndim == 2 for batch in batches for item in batch)


def test_tensor_cache_roundtrip_avoids_json_and_rdkit_in_training_loader(tmp_path):
    decision = {
        "reaction_id": "cache-smoke",
        "kind": "FINISH",
        "current": "[CH3:1][Br:2]",
        "target": "[CH3:1][Br:2]",
        "history": {"events": []},
    }
    source = tmp_path / "rank.jsonl"
    source.write_text(json.dumps(decision) + "\n")
    prepared = JsonlDecisionDataset(source)[0]
    cache = tmp_path / "cache"
    cache.mkdir()
    write_tensor_chunk([prepared], cache / "chunk00000.pt")
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "rows": 1,
                "chunks": [{"file": "chunk00000.pt", "rows": 1}],
            }
        )
    )
    source.unlink()
    restored = TensorChunkDecisionDataset(cache)[0]
    assert restored.value["reaction_id"] == "cache-smoke"
    assert restored.current_graph.maps == (1, 2)
    assert restored.history.events == ()
