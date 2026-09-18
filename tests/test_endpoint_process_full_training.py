from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from train_endpoint_process_rlvr import latest_resumable_checkpoint, rank_shard_path


def test_rank_shard_path_formats_rank_without_cross_rank_reuse():
    path = rank_shard_path(
        "/tmp/train.rank{rank:02d}.of{world_size:02d}.jsonl", rank=3, world_size=8
    )
    assert path == Path("/tmp/train.rank03.of08.jsonl")


def test_rank_shard_path_requires_rank_placeholder():
    try:
        rank_shard_path("/tmp/train.jsonl", rank=0, world_size=8)
    except ValueError as exc:
        assert "{rank}" in str(exc)
    else:
        raise AssertionError("shared training path was accepted as a rank shard")


def test_latest_resumable_checkpoint_ignores_partial_saves(tmp_path: Path):
    for update, complete in ((8, True), (16, False), (12, True)):
        path = tmp_path / f"checkpoint-{update}"
        path.mkdir()
        (path / "adapter_model.safetensors").touch()
        if complete:
            (path / "trainer_state.pt").touch()
    assert latest_resumable_checkpoint(tmp_path) == (12, tmp_path / "checkpoint-12")
