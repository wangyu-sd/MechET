#!/usr/bin/env python3
"""Real-weight two-example Qwen3.5 text-only/tool-mask/LoRA gradient preflight.

Must pass on the assigned training node before distributed tokenization/SFT.
This is a compatibility check, NOT a substitute for full multi-epoch training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.assistant_masking import (
    encode_assistant_only_conversation, render_chat,
    render_qwen_sft_tool_prefix, tokenize_text,
)
from mechet.qwen35_text_training import (
    MODEL_ID, MODEL_REVISION, check_trainable_language_only,
    language_lora_modules, load_text_only_qwen35,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to replace Qwen3.5 model preflight audit")
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU needed for actual 0.8B forward/backward preflight")
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    rows = []
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("tools") or row.get("metadata", {}).get("input_condition") != "main_product_only_v1":
                raise ValueError("Not a corrected tool SFT record")
            rows.append(row)
            if len(rows) == 2:
                break
    if len(rows) != 2:
        raise ValueError("Need at least two real corrected decisions")
    validated = []
    for row in rows:
        full = render_chat(tok, row["messages"], tools=row["tools"], add_generation_prompt=False)
        prefix = render_qwen_sft_tool_prefix(
            tok, row["messages"][:2], tools=row["tools"]
        )
        ids = tokenize_text(tok, full)
        prefix_ids = tokenize_text(tok, prefix)
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Qwen3.5 native tool prefix drift between SFT and inference")
        encoded, audit = encode_assistant_only_conversation(
            tok, row, max_length=4096
        )
        if not any(x != -100 for x in encoded["labels"]):
            raise ValueError("No assistant supervision")
        if any(x != -100 for x in encoded["labels"][:len(prefix_ids)]):
            raise ValueError("Gold tool call leaked into model prefix labels")
        validated.append((encoded, audit))
    dtype = torch.bfloat16
    model = load_text_only_qwen35(
        revision=MODEL_REVISION, dtype=dtype, quantization_config=None,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    target = language_lora_modules(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=target, task_type="CAUSAL_LM",
    ))
    stats = check_trainable_language_only(model)
    model.train()
    examples = validated[0][0]
    # Small real loss/backward validates the multimodal wrapper with text-only
    # input and gradients to the intended DeltaNet/attention LoRA matrices.
    n = min(384, len(examples["input_ids"]))
    inputs = {
        k: torch.tensor([v[:n]], device="cuda")
        for k, v in examples.items()
        if k in ("input_ids", "attention_mask", "labels")
    }
    if not (inputs["labels"] != -100).any():
        # Supervision usually comes at the sequence tail; retain last 384
        inputs = {
            k: torch.tensor([v[-n:]], device="cuda")
            for k, v in examples.items()
            if k in ("input_ids", "attention_mask", "labels")
        }
    outputs = model(**inputs)
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss):
        raise ValueError("Qwen3.5 text-only LoRA gradient loss is invalid")
    loss.backward()
    have_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.requires_grad)
    if not have_grad:
        raise ValueError("Qwen3.5 LoRA backward produced no finite gradient")
    report = {
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "architecture": type(model.base_model.model).__name__,
        "assistant_mask": "validated_real_rows",
        "prefix_exact": True, "lora_gradients": True,
        "vision_frozen": True, "trainable": stats,
        "preflight_not_training": True,
        "sample_count": len(rows),
        "sample_loss": float(loss.detach().cpu()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
