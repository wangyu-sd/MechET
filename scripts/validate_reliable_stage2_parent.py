#!/usr/bin/env python3
"""Fail closed before continuing Reliable MechET Stage II from Stage I."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing final Stage-I artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_parent(
    parent: Path,
    *,
    expected_sha256: str,
    stage1_config: dict[str, Any],
    stage1_manifest: dict[str, Any],
) -> str:
    """Return weight SHA only for a complete, lineage-matched final adapter."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected parent SHA-256 must be 64 lowercase hex digits")
    weight = parent / "adapter_model.safetensors"
    if not weight.is_file() or weight.stat().st_size == 0:
        raise ValueError("missing final Stage-I adapter weights")
    if not (parent / "adapter_config.json").is_file():
        raise ValueError("missing final Stage-I PEFT config")
    sha = hashlib.sha256()
    with weight.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    if digest != expected_sha256:
        raise ValueError("Stage-I final adapter SHA-256 mismatch")

    adapter = _read_json(parent / "adapter_manifest.json")
    contract = _read_json(parent / "data_contract.json")
    revision = stage1_config["training"]["model_revision"]
    expected_fields = {
        "artifact_type": "trainable_peft_adapter",
        "condition_name": stage1_config["condition_name"],
        "base_model": stage1_config["model_name_or_path"],
        "requested_model_revision": revision,
        "train_file_sha256": stage1_manifest["splits"]["train"]["output_sha256"],
    }
    for key, expected in expected_fields.items():
        if adapter.get(key) != expected:
            raise ValueError(f"Stage-I adapter lineage mismatch: {key}")
    expected_contract = {
        "artifact_type": "tool_sft_data_contract",
        "condition_name": stage1_config["condition_name"],
        "model_name_or_path": stage1_config["model_name_or_path"],
        "train_file_sha256": stage1_manifest["splits"]["train"]["output_sha256"],
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "assistant_only_loss": True,
        "packing": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"Stage-I training contract mismatch: {key}")
    expected_report = str(Path(stage1_config["output_dir"]) / "data_contract.json")
    if adapter.get("data_contract") != expected_report:
        raise ValueError("Stage-I adapter points to a different data contract")
    if contract.get("base_model_revision") != revision:
        raise ValueError("Stage-I resolved base model revision mismatch")
    if contract.get("initial_adapter_path"):
        raise ValueError("Stage-I parent was itself initialized from another adapter")
    return digest


def validate_stage2_child(
    parent: Path, *, project_root: Path,
    stage1_config: dict[str, Any], stage2_config: dict[str, Any],
) -> None:
    """Require Stage II to consume the exact adapter that passed the parent gate."""
    def resolve_config_path(value: Any, field: str) -> Path:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"Stage-II lineage path is missing: {field}")
        path = Path(text)
        return (path if path.is_absolute() else project_root / path).resolve()

    configured = str(stage2_config.get("initial_adapter_path") or "").strip()
    if not configured:
        raise ValueError("Stage-II initial_adapter_path is missing")
    stage1_output = resolve_config_path(stage1_config.get("output_dir"), "Stage-I output_dir")
    if parent.resolve() != stage1_output:
        raise ValueError("validated Stage-I parent path differs from Stage-I output_dir")
    resolved = resolve_config_path(configured, "initial_adapter_path")
    if resolved != parent.resolve():
        raise ValueError("Stage-II initial_adapter_path differs from validated Stage-I parent")
    if stage2_config.get("model_name_or_path") != stage1_config.get("model_name_or_path"):
        raise ValueError("Stage-II base model differs from Stage I")
    if ((stage2_config.get("training") or {}).get("model_revision")
            != (stage1_config.get("training") or {}).get("model_revision")):
        raise ValueError("Stage-II base revision differs from Stage I")
    if (stage2_config.get("lora") or {}) != (stage1_config.get("lora") or {}):
        raise ValueError("Stage-II LoRA settings differ from inherited Stage-I adapter")
    contract = dict(stage2_config.get("contract") or {})
    if contract.get("stage") != "trajectory_sft" or contract.get("parent_stage") != "state_sft":
        raise ValueError("Stage-II contract is not trajectory SFT from State-SFT")
    training = dict(stage2_config.get("training") or {})
    if training.get("assistant_only_loss") is not True or training.get("packing") is not False:
        raise ValueError("Stage-II changed assistant-only or packing supervision")
    if resolve_config_path(stage2_config.get("output_dir"), "output_dir") == stage1_output:
        raise ValueError("Stage-II output would overwrite Stage-I artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--child-config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    import yaml

    stage1_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage2_config = yaml.safe_load(args.child_config.read_text(encoding="utf-8"))
    digest = validate_parent(
        args.parent,
        expected_sha256=args.expected_sha256,
        stage1_config=stage1_config,
        stage1_manifest=_read_json(args.manifest),
    )
    validate_stage2_child(
        args.parent, project_root=args.project_root,
        stage1_config=stage1_config, stage2_config=stage2_config,
    )
    print(f"[reliable-mechet] Stage-I final adapter lineage verified sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
