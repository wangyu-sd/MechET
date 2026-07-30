#!/usr/bin/env python3
"""Generate MECH_PROOF v1 programs with Qwen and an optional LoRA adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.model import resolve_qwen_model_path


def _load_rows(path: Path, limit: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    base = (
        args.model_path
        or os.environ.get("QWEN_MODEL_PATH")
        or resolve_qwen_model_path()
        or ""
    )
    if not base or not Path(base).exists():
        raise FileNotFoundError("set QWEN_MODEL_PATH or --model-path")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    quant = None
    if torch.cuda.is_available() and not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(
            model,
            str(args.adapter),
            is_trainable=False,
        )
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle, torch.no_grad():
        for row in _load_rows(args.data, args.limit):
            prompt_messages = [
                message
                for message in row.get("messages", [])
                if message.get("role") != "assistant"
            ]
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            )
            if torch.cuda.is_available():
                inputs = {
                    key: value.to(model.device)
                    for key, value in inputs.items()
                }
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.sample,
                temperature=(
                    max(args.temperature, 1e-5) if args.sample else None
                ),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
            prediction = tokenizer.decode(
                output[inputs["input_ids"].shape[1] :],
                skip_special_tokens=False,
            )
            handle.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
