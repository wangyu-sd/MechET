"""Regression tests for the one-command evaluation launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mechet_eval.py"
    spec = importlib.util.spec_from_file_location("run_mechet_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_use_vllm_is_a_flag():
    module = _load_script()
    args = module.build_parser().parse_args(["--use-vllm", "--tensor-parallel-size", "2"])
    assert args.use_vllm
    assert args.tensor_parallel_size == 2


def test_tensor_parallel_is_only_forwarded_to_vllm(monkeypatch, tmp_path):
    module = _load_script()
    calls: list[list[str]] = []
    monkeypatch.setattr(module.subprocess, "check_call", lambda command: calls.append(command))

    module.main(["--data", str(tmp_path / "data.jsonl"), "--out-dir", str(tmp_path / "plain")])
    assert "infer_mechet.py" in calls[0][1]
    assert "--tensor-parallel-size" not in calls[0]

    calls.clear()
    module.main(
        [
            "--data",
            str(tmp_path / "data.jsonl"),
            "--out-dir",
            str(tmp_path / "vllm"),
            "--use-vllm",
            "--tensor-parallel-size",
            "2",
        ]
    )
    assert "infer_mechet_vllm.py" in calls[0][1]
    assert calls[0][calls[0].index("--tensor-parallel-size") + 1] == "2"
