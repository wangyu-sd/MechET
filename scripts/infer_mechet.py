#!/usr/bin/env python3
"""Generate MechET MECH_ET v3 CoT predictions with Qwen (+ optional LoRA adapter)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.chat_template import build_generation_prompt
from mechet.metrics import extract_gold_answer, extract_product_from_user
from mechet.model import resolve_qwen_model_path
from mechet.sft import MECH_ET_SYSTEM_PROMPT


def _load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _build_model(base_model: str, adapter: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tok_path = Path(adapter).parent / "tokenizer" if adapter and (Path(adapter).parent / "tokenizer").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    ) if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/mechet_sft/valid.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/mechet_eval/generations.jsonl")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    base = args.model_path or os.environ.get("QWEN_MODEL_PATH") or resolve_qwen_model_path() or ""
    if not base or not Path(base).exists():
        print(json.dumps({"error": "missing base model; set QWEN_MODEL_PATH"}))
        return 1

    rows = _load_rows(args.data, args.limit)
    model, tokenizer = _build_model(base, str(args.adapter) if args.adapter else None)

    import torch

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            messages = row.get("messages") or []
            user = next((m for m in messages if m.get("role") == "user"), {"content": ""})
            prompt_messages = [
                {"role": "system", "content": MECH_ET_SYSTEM_PROMPT},
                {"role": "user", "content": str(user.get("content") or "")},
            ]
            prompt = build_generation_prompt(tokenizer, prompt_messages)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            record = {
                "id": row.get("id"),
                "prediction": text,
                "product": extract_product_from_user(str(user.get("content") or "")),
                "gold_answer": extract_gold_answer(row),
                "topology": (row.get("metadata") or {}).get("topology"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"wrote": str(args.out), "n": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
