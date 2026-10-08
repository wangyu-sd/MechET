#!/usr/bin/env python3
"""One unsaved Stage-II optimizer step from a provisional Stage-I adapter.

This checks the real Qwen tokenizer, compressed-history supervision, PEFT
warm-start, forward/backward pass and adapter update. It is not an evaluation
or a substitute for Stage I's final adapter lineage gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.assistant_masking import encode_assistant_only_conversation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def history_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "::history_v2" not in str(row.get("id", "")):
                continue
            user_text = "\n".join(
                str(message.get("content") or "")
                for message in row.get("messages") or []
                if message.get("role") == "user"
            )
            if "TRAJECTORY HISTORY" in user_text and "accepted_actions: 1" in user_text:
                return row
    raise ValueError("no Stage-II decision with one accepted historical action")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    args = parser.parse_args()

    os.environ["HF_HUB_CACHE"] = str(args.hf_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA optimizer smoke requires a GPU")
    adapter_file = args.parent_adapter / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    if not (args.parent_adapter / "trainer_state.json").is_file():
        raise FileNotFoundError("provisional parent needs trainer_state.json")

    row = history_row(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision=args.revision, local_files_only=True
    )
    encoded, audit = encode_assistant_only_conversation(tokenizer, row, max_length=4096)
    if audit["exceeds_max_length"] or audit["supervised_tokens"] <= 0:
        raise ValueError("Stage-II smoke row failed the SFT token contract")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        revision=args.revision,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(args.parent_adapter), is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda").train()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("warm-start did not expose only trainable LoRA parameters")
    name, parameter = trainable[0]
    before = parameter.detach().float().clone()
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=5e-6)
    batch = {
        key: torch.tensor([value], dtype=torch.long, device="cuda")
        for key, value in encoded.items()
    }
    loss = model(**batch).loss
    if not bool(torch.isfinite(loss)):
        raise ValueError("Stage-II warm-start produced nonfinite loss")
    loss.backward()
    grad_norm = sum(
        float(param.grad.detach().float().norm().item())
        for _, param in trainable
        if param.grad is not None
    )
    if not 0 < grad_norm < float("inf"):
        raise ValueError("Stage-II warm-start produced no finite LoRA gradients")
    optimizer.step()
    update = float((parameter.detach().float() - before).abs().max().item())
    if update <= 0:
        raise ValueError("Stage-II warm-start made no LoRA parameter update")
    print(json.dumps({
        "scope": "provisional_stage2_single_optimizer_step_no_weights_saved",
        "row_id": row["id"],
        "input_tokens": audit["raw_length"],
        "supervised_tokens": audit["supervised_tokens"],
        "parent_adapter_sha256": sha256(adapter_file),
        "base_revision": args.revision,
        "loss": float(loss.detach().item()),
        "sum_lora_grad_norm": grad_norm,
        "updated_parameter": name,
        "max_parameter_update": update,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
