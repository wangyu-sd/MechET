"""Shared model and checkpoint-lineage helpers for trace-owned training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.model_revision import is_immutable_revision, revision_contract

_EXCLUDED_HASH_FILES = {"adapter_manifest.json", "data_contract.json"}


def path_sha256(path: str | Path) -> str:
    value = Path(path)
    if not value.exists():
        return ""
    digest = hashlib.sha256()
    if value.is_file():
        digest.update(value.read_bytes())
        return digest.hexdigest()
    for file in sorted(
        item
        for item in value.rglob("*")
        if item.is_file() and item.name not in _EXCLUDED_HASH_FILES
    ):
        digest.update(str(file.relative_to(value)).encode())
        digest.update(b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(cfg: dict[str, Any], adapter: str) -> Path | None:
    explicit = str(cfg.get("initial_adapter_manifest") or "").strip()
    if explicit:
        return Path(explicit)
    return Path(adapter) / "adapter_manifest.json" if adapter else None


def _load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"adapter manifest must be an object: {path}")
    return dict(value)


def lineage_report(cfg: dict[str, Any]) -> dict[str, Any]:
    adapter = str(cfg.get("initial_adapter_path") or "").strip()
    path = Path(adapter) if adapter else None
    manifest_path = _manifest_path(cfg, adapter)
    manifest = _load_manifest(manifest_path)
    declared = str(cfg.get("initial_adapter_sha256") or "").strip()
    if declared.lower() in {"", "auto"}:
        declared = str(manifest.get("adapter_sha256") or "").strip()
    actual = path_sha256(path) if path else ""
    environment_revision = str(cfg.get("environment_revision") or "").strip()
    executor_revision = str(cfg.get("executor_revision") or "").strip()
    training = dict(cfg.get("training") or {})
    configured_model_revision = str(
        training.get("model_revision") or cfg.get("model_revision") or ""
    ).strip()
    revision = revision_contract(
        configured_revision=configured_model_revision,
        adapter_manifest=manifest,
    )
    return {
        "initial_adapter_path": adapter or None,
        "initial_adapter_exists": bool(path and path.exists()),
        "initial_adapter_manifest": str(manifest_path) if manifest_path else None,
        "initial_adapter_manifest_exists": bool(
            manifest_path and manifest_path.exists()
        ),
        "initial_adapter_sha256_declared": declared or None,
        "initial_adapter_sha256_actual": actual or None,
        "initial_adapter_hash_matches": (
            None if not declared or not actual else declared == actual
        ),
        "require_initial_adapter": bool(cfg.get("require_initial_adapter", False)),
        "adapter_artifact_type": manifest.get("artifact_type"),
        "adapter_base_model": manifest.get("base_model"),
        "adapter_requested_model_revision": manifest.get("requested_model_revision"),
        "adapter_base_model_revision": revision.get("adapter_base_model_revision"),
        "adapter_tokenizer_revision": manifest.get("tokenizer_revision"),
        "adapter_condition_name": manifest.get("condition_name"),
        "adapter_environment_revision": manifest.get("environment_revision"),
        "adapter_executor_revision": manifest.get("executor_revision"),
        "adapter_seed": manifest.get("seed"),
        "adapter_data_seed": manifest.get("data_seed"),
        "tool_sft_data_contract": manifest.get("data_contract"),
        "evidence_suite_manifest": cfg.get("evidence_suite_manifest") or None,
        "environment_revision": environment_revision or None,
        "executor_revision": executor_revision or None,
        "configured_model_revision": configured_model_revision or None,
        "configured_model_revision_is_immutable": is_immutable_revision(
            configured_model_revision
        ),
        "resolved_model_revision": revision.get("resolved_model_revision"),
        "resolved_model_revision_is_immutable": revision.get(
            "resolved_revision_is_immutable"
        ),
    }


def validate_lineage(cfg: dict[str, Any]) -> dict[str, Any]:
    report = lineage_report(cfg)
    required = bool(report["require_initial_adapter"])
    if required and not report["initial_adapter_exists"]:
        raise FileNotFoundError(
            "require_initial_adapter=true but initial_adapter_path does not exist: "
            f"{report['initial_adapter_path']}"
        )
    if required and not report["initial_adapter_manifest_exists"]:
        raise FileNotFoundError(
            "Tool-SFT adapter manifest does not exist: "
            f"{report['initial_adapter_manifest']}"
        )
    if report["initial_adapter_hash_matches"] is False:
        raise ValueError(
            "initial adapter hash mismatch: "
            f"declared={report['initial_adapter_sha256_declared']} "
            f"actual={report['initial_adapter_sha256_actual']}"
        )
    if required and not report["initial_adapter_sha256_declared"]:
        raise ValueError("required Tool-SFT adapter has no frozen SHA-256")
    if required and report["adapter_artifact_type"] != "trainable_peft_adapter":
        raise ValueError("initial adapter manifest has an invalid artifact_type")
    model_name = str(cfg.get("model_name_or_path") or "")
    if required and report["adapter_base_model"] not in (None, model_name):
        raise ValueError(
            "Tool-SFT base model mismatch: "
            f"{report['adapter_base_model']} != {model_name}"
        )
    if required and not report["resolved_model_revision"]:
        raise ValueError(
            "required Tool-SFT adapter has no frozen base-model revision; "
            "retrain it with the current Tool-SFT pipeline"
        )
    if required and not report["resolved_model_revision_is_immutable"]:
        raise ValueError(
            "required Tool-SFT adapter base-model revision is mutable; "
            "the adapter manifest must record the resolved 40-hex commit SHA"
        )
    for key, adapter_key in (
        ("environment_revision", "adapter_environment_revision"),
        ("executor_revision", "adapter_executor_revision"),
    ):
        expected = report[key]
        observed = report[adapter_key]
        if required and expected and observed and expected != observed:
            raise ValueError(
                f"Tool-SFT {key} mismatch: {observed} != {expected}"
            )
    contract = report["tool_sft_data_contract"]
    if required and (not contract or not Path(str(contract)).exists()):
        raise FileNotFoundError(
            f"Tool-SFT data contract does not exist: {contract}"
        )
    return report


def build_trainable_model(cfg: dict[str, Any], torch_module):
    """Return ``(model, peft_config)`` for GRPOTrainer."""

    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM

    model_name = str(cfg.get("model_name_or_path") or "")
    if not model_name:
        raise ValueError("model_name_or_path is required")
    lineage = validate_lineage(cfg)

    initial_adapter = str(cfg.get("initial_adapter_path") or "").strip()
    training = dict(cfg.get("training") or {})
    revision = lineage.get("resolved_model_revision")
    if initial_adapter:
        dtype = None
        if bool(training.get("bf16", False)):
            dtype = torch_module.bfloat16
        elif bool(training.get("fp16", False)):
            dtype = torch_module.float16
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=bool(training.get("trust_remote_code", True)),
            torch_dtype=dtype,
            device_map=training.get("device_map"),
        )
        if bool(training.get("gradient_checkpointing", True)) and hasattr(
            base_model, "config"
        ):
            base_model.config.use_cache = False
        model = PeftModel.from_pretrained(
            base_model, initial_adapter, is_trainable=True
        )
        return model, None

    configured_revision = str(training.get("model_revision") or "").strip()
    if configured_revision and not is_immutable_revision(configured_revision):
        raise ValueError(
            "from-base GRPO requires training.model_revision to be an immutable "
            "40-hex commit SHA; mutable aliases are allowed only for Tool-SFT, "
            "which resolves and records the actual commit"
        )
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
