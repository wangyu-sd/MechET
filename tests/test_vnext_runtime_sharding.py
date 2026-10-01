import json
import sys

import pytest

from scripts.benchmark_vnext_structured_vllm import shard_assignment
from scripts.summarize_vnext_runtime_benchmark import main as summarize_main


def test_vllm_shards_require_single_visible_gpu(monkeypatch):
    monkeypatch.setenv("VNEXT_RANK", "3")
    monkeypatch.setenv("VNEXT_WORLD_SIZE", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert shard_assignment() == (3, 8)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    with pytest.raises(ValueError, match="exactly one visible GPU"):
        shard_assignment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("VNEXT_RANK", "8")
    with pytest.raises(ValueError, match="invalid vNext shard"):
        shard_assignment()


def test_runtime_summary_rejects_partial_and_duplicate_rank_coverage(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "summary.json"
    report = {
        "rank": 0, "mode": "eager_prefix", "n_states": 2,
        "wall_seconds": 1.0, "states_per_second": 2.0,
        "parse_success": 1.0, "handle_valid": 1.0,
        "exact_action_vs_eager_prefix": 1.0,
    }
    (raw / "eager_prefix.rank00.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", [
        "summary", "--input-dir", str(raw), "--output", str(output),
        "--expected-ranks", "2", "--expected-states", "4",
        "--expected-modes", "eager_prefix",
    ])
    with pytest.raises(ValueError, match="rank coverage"):
        summarize_main()
    (raw / "eager_prefix.rank01.json").write_text(json.dumps({**report, "rank": 1}))
    assert summarize_main() == 0
    assert json.loads(output.read_text())["modes"]["eager_prefix"]["n_states"] == 4

    output.unlink()
    (raw / "eager_prefix.rank02.json").write_text(json.dumps({**report, "rank": 1}))
    with pytest.raises(ValueError, match="rank coverage"):
        summarize_main()
