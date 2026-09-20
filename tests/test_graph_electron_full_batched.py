import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from train_graph_electron_full_batched import prefetched_batches


def test_spawn_process_prefetch_roundtrip(tmp_path):
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
    with ProcessPoolExecutor(
        max_workers=2, mp_context=mp.get_context("spawn")
    ) as executor:
        batches = list(
            prefetched_batches(
                path,
                batch_size=2,
                skip=0,
                executor=executor,
                prefetch=2,
            )
        )
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert all(item.current_graph.atoms.ndim == 2 for batch in batches for item in batch)
