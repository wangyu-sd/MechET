"""Shared model and checkpoint-lineage helpers for trace-owned agent training."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def path_sha256(path: str | Path) -> str:
    value = Path(path)
    if not value.exists():
        return ""
    digest = hashlib.sha256()
    if value.is_file():
        digest.update(value.read_bytes())
        return digest.hexdigest()
    for file in sorted(item for item in value.rglob("*") if item.is_file()):
        digest.update(str(file.relative_to(value)).encode())
        digest.update(b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def lineage_report(cfg: dict[str, Any]) -> dict[str, Any]:
    adapter = str(cfg.get("initial_adapter_path") or "").strip()
    expected = str(cfg.get("initial_adapter_sha256") or "").strip()
    path = Path(adapter) if adapter else None
    actual = path_sha256(path) if path else ""
    return {
        "initial_adapter_path": adapter or None,
        "initial_adapter_exists": bool(path and path.exists()),
        "initial_adapter_sha256_declared": expected or None,
        "initial_adapter_sha256_actual": actual or None,
        "initial_adapter_hash_matches": (
            None if not expected or not actual else expected == actual
        ),
        "require_initial_adapter": bool(cfg.get("require_initial_adapter", False)),
        "tool_sft_manifest": cfg.get("tool_sft_manifest") or None,
        "environment_revision": cfg.get("environment_revision") or None,
        "executor_revision": cfg.get("executor_revision") or None,
    }


def validate_lineage(cfg: dict[str, Any]) -> dict[str, Any]:
    report = lineage_report(cfg)
    if report["require_initial_adapter"] and not report["initial_adapter_exists"]:
        raise FileNotFoundError(
            "require_initial_adapter=true but initial_adapter_path does not exist: "
            f"{report['initial_adapter_path']}"
        )
    if report["initial_adapter_hash_matches"] is False:
        raise ValueError(
            "initial adapter hash mismatch: "
            f"declared={report['initial_adapter_sha256_declared']} "
            f"actual={report['initial_adapter_sha256_actual']}"
        )
    manifest = report["tool_sft_manifest"]
    if report["require_initial_adapter"] and manifest and not Path(str(manifest)).exists():
        raise FileNotFoundError(f"Tool-SFT manifest does not exist: {manifest}")
    return report


def build_trainable_model(cfg: dict[str, Any], torch_module):
    """Return ``(model, peft_config)`` for GRPOTrainer.

    Without an initial adapter, the trainer receives a model name plus a new
    LoRA config. With an initial Tool-SFT adapter, the base model and trainable
    PEFT adapter are loaded explicitly and no second nested adapter is created.
    """

    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM

    model_name = str(cfg.get("model_name_or_path") or "")
    if not model_name:
        raise ValueError("model_name_or_path is required")
    validate_lineage(cfg)

    initial_adapter = str(cfg.get("initial_adapter_path") or "").strip()
    training = dict(cfg.get("training") or {})
    if initial_adapter:
        dtype = None
        if bool(training.get("bf16", False)):
            dtype = torch_module.bfloat16
        elif bool(training.get("fp16", False)):
            dtype = torch_module.float16
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=bool(training.get("trust_remote_code", True)),
            torch_dtype=dtype,
            device_map=training.get("device_map"),
        )
        model = PeftModel.from_pretrained(
            base_model,
            initial_adapter,
            is_trainable=True,
        )
        return model, None

    lora = dict(cfg.get("lora") or {})
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(
            lora.get("target_modules")
            or ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        task_type="CAUSAL_LM",
    )
    return model_name, peft_config
