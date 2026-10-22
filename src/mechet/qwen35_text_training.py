"""Fail-closed Qwen3.5-0.8B *text-only* LoRA compatibility for MechET.

Qwen3.5-0.8B is a native multimodal Qwen3_5ForConditionalGeneration:
AutoModelForCausalLM(Qwen3.5) is not a valid drop-in replacement for Qwen3.
Retain the official checkpoint, freeze vision, and target the language decoder.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# Include DeltaNet projections, self attention, and FFN; only language_model.
TEXT_LORA_TARGETS = (
    r"^.*model\.language_model\..*\."
    r"(?:q_proj|k_proj|v_proj|o_proj|out_proj|"
    r"in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|"
    r"gate_proj|up_proj|down_proj)$"
)


def is_qwen35_text_only(config: Mapping[str, Any]) -> bool:
    return str(config.get("model_architecture") or "") == "qwen3_5_vl_text_only"


def assert_model_configuration(config: Mapping[str, Any]) -> None:
    if not is_qwen35_text_only(config):
        return
    if config.get("model_name_or_path") != MODEL_ID:
        raise ValueError("Qwen3.5 training must use pinned 0.8B model identity")
    training = dict(config.get("training") or {})
    if training.get("model_revision") != MODEL_REVISION:
        raise ValueError("Qwen3.5 model revision must be immutable and audited")
    if training.get("qlora") is True:
        raise ValueError("Qwen3.5 text-only experiment uses bf16 LoRA, not QLoRA")
    if not bool(training.get("assistant_only_loss")):
        raise ValueError("Qwen3.5 requires explicit assistant-only loss")


def language_lora_modules(model: Any) -> str:
    names = [name for name, _ in model.named_modules()
             if re.fullmatch(TEXT_LORA_TARGETS, name)]
    if not names:
        raise ValueError("Qwen3.5 language decoder LoRA targets absent")
    if any(".visual." in name or ".vision." in name for name in names):
        raise ValueError("LoRA matched Qwen3.5 vision tower")
    # Gated DeltaNet is 3/4 layers in 0.8B: a q/k/v-only LoRA silently
    # misses them, so insist on both projection families.
    if not any(x.endswith(".in_proj_qkv") for x in names):
        raise ValueError("Qwen3.5 Gated DeltaNet not covered by LoRA")
    if not any(x.endswith(".q_proj") for x in names):
        raise ValueError("Qwen3.5 full-attention layers not covered by LoRA")
    if not any(x.endswith(".gate_proj") for x in names):
        raise ValueError("Qwen3.5 FFN not covered by LoRA")
    return TEXT_LORA_TARGETS


def load_text_only_qwen35(*, revision: str, dtype: Any,
                          quantization_config: Any,
                          device_map: Any, attn_implementation: str | None) -> Any:
    if revision != MODEL_REVISION:
        raise ValueError("Wrong Qwen3.5 base SHA")
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        try:
            from transformers import AutoModelForMultimodalLM as AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError("Upgrade to transformers with Qwen3.5 multimodal support") from exc
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=False,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    if getattr(model.config, "model_type", None) != "qwen3_5":
        raise ValueError("Qwen3.5 checkpoint architecture changed")
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise ValueError("Expected Qwen3.5 visual tower for explicit freeze")
    visual.requires_grad_(False)
    language_lora_modules(model)
    return model


def check_trainable_language_only(model: Any) -> dict[str, int]:
    trainable = [(name, p.numel()) for name, p in model.named_parameters()
                 if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters")
    leaked = [name for name, _ in trainable
              if ".visual." in name or ".vision." in name]
    if leaked:
        raise ValueError(f"Qwen3.5 visual weights unexpectedly trainable: {leaked[:5]}")
    if not all("lora_" in name for name, _ in trainable):
        raise ValueError("Qwen3.5 expected LoRA-only trainable parameter set")
    if not any("in_proj_qkv" in name for name, _ in trainable):
        raise ValueError("Qwen3.5 DeltaNet LoRA weights missing")
    return {"trainable_tensors": len(trainable),
            "trainable_parameters": sum(n for _, n in trainable)}
