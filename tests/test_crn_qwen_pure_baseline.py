"""CPU-only integration checks for the resumable PR83 pure-Qwen control."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from mechet.crn_graph_bridge import file_sha256
from scripts import run_crn_qwen_pure_baseline as pure


def test_stage1_control_uses_state_sft_prompt_and_rejects_history_mismatch(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    (adapter / "adapter_manifest.json").write_text(json.dumps({
        "environment_revision": "natural_language_electron_event_v2",
        "executor_revision": "MECH_PROOF_v1_full_coverage_v4",
        "base_model": "Qwen/Qwen3-0.6B", "base_model_revision": revision,
    }))
    argv = ["--data", str(tmp_path / "valid.jsonl"), "--source-sha256", "source",
            "--output", str(tmp_path / "output"), "--sample-reactions", "2890",
            "--policy-adapter", str(adapter)]
    args = pure.prepare_args(pure.parser().parse_args(argv))
    assert args.compact_history is False
    with pytest.raises(ValueError, match="environment_revision"):
        pure.prepare_args(pure.parser().parse_args(argv + ["--compact-history"]))


def test_pure_control_writes_all_rows_and_resumes_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "valid.jsonl"
    source.write_text(
        "".join(json.dumps({"id": f"case_{i}", "source_id": f"source_{i}"}) + "\n"
                for i in range(5)),
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"test-only-adapter")
    output = tmp_path / "output"
    calls: list[str] = []

    class FakeRuntime:
        def __init__(self, args, local_rank):
            assert args.matched_v2 and args.product_only_remap
            assert args.branching == args.early_beam == args.late_beam == 1
            assert local_rank == 0

    def fake_rollout(runtime, row, args):
        calls.append(row["id"])
        return {"id": row["id"], "top1_structural_exact": row["id"] == "case_0",
                "top_terminal": True, "attempts": []}

    monkeypatch.setattr(pure, "Runtime", FakeRuntime)
    monkeypatch.setattr(pure, "rollout", fake_rollout)
    monkeypatch.setattr(pure, "validate_v2_adapter_manifest", lambda *a, **k: None)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    arguments = [
        "run_crn_qwen_pure_baseline.py", "--data", str(source),
        "--source-sha256", file_sha256(source), "--output", str(output),
        "--sample-reactions", "5", "--policy-adapter", str(adapter),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert pure.main() == 0
    results_path = output / "results.shard-00-of-01.jsonl"
    original = results_path.read_bytes()
    assert len(calls) == 5
    assert len(original.splitlines()) == 5

    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    assert pure.main() == 0
    assert len(calls) == 5
    assert results_path.read_bytes() == original

    monkeypatch.setattr(sys, "argv", arguments + ["--seed", "18", "--resume"])
    with pytest.raises(ValueError, match="resume contract differs"):
        pure.main()
    assert results_path.read_bytes() == original
