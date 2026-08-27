import importlib.util
import json
from pathlib import Path

import pytest


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_tool_sft.py"
    spec = importlib.util.spec_from_file_location("train_tool_sft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_initial_adapter_config = (
    _load_train_module().validate_initial_adapter_config
)


def _adapter(tmp_path: Path, *, rank: int = 16) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_warm_start_adapter_matches_lora_contract(tmp_path):
    adapter = _adapter(tmp_path)
    report = validate_initial_adapter_config(
        {
            "initial_adapter_path": str(adapter),
            "lora": {
                "r": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
        },
        {"initial_adapter_sha256_actual": "abc"},
    )
    assert report is not None
    assert report["adapter_lora_r"] == 16
    assert report["adapter_target_modules"] == [
        "k_proj",
        "o_proj",
        "q_proj",
        "v_proj",
    ]


def test_warm_start_adapter_rejects_lora_mismatch(tmp_path):
    adapter = _adapter(tmp_path, rank=8)
    with pytest.raises(ValueError, match="LoRA config mismatch"):
        validate_initial_adapter_config(
            {
                "initial_adapter_path": str(adapter),
                "lora": {"r": 16, "alpha": 32},
            },
            {},
        )
