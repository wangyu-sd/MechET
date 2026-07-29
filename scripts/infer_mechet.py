#!/usr/bin/env python3
"""Generate MechET MECH_ET v3 CoT predictions with Qwen (+ optional LoRA adapter)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.chat_template import build_generation_prompt
from mechet.metrics import extract_answer_from_prediction, extract_gold_answer, extract_product_from_user, normalize_candidates
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


def _build_model(base_model: str, adapter: str | None, *, use_4bit: bool):
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
    quant = None
    if use_4bit and torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
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
    model.eval()
    return model, tokenizer


def _generate_one(
    model,
    tokenizer,
    user_content: str,
    *,
    max_new_tokens: int,
    max_input_tokens: int,
    num_beams: int,
    num_return_sequences: int,
    temperature: float,
    do_sample: bool,
):
    import torch

    prompt_messages = [
        {"role": "system", "content": MECH_ET_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = build_generation_prompt(tokenizer, prompt_messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if num_beams > 1:
        gen_kwargs.update(
            num_beams=num_beams,
            num_return_sequences=min(num_return_sequences, num_beams),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
    else:
        gen_kwargs["do_sample"] = do_sample
        if do_sample:
            gen_kwargs["temperature"] = temperature
        gen_kwargs["num_return_sequences"] = 1

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    texts: list[str] = []
    for seq in out:
        texts.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=False))
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/mechet_sft/valid.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/mechet_eval/generations.jsonl")
    parser.add_argument("--manifest", type=Path, default=None, help="defaults to <out>.manifest.json")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    base = args.model_path or os.environ.get("QWEN_MODEL_PATH") or resolve_qwen_model_path() or ""
    if not base or not Path(base).exists():
        print(json.dumps({"error": "missing base model; set QWEN_MODEL_PATH"}))
        return 1

    rows = _load_rows(args.data, args.limit)
    model, tokenizer = _build_model(base, str(args.adapter) if args.adapter else None, use_4bit=not args.no_4bit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix + ".manifest.json")
    started = datetime.now(timezone.utc).isoformat()

    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            messages = row.get("messages") or []
            user = next((m for m in messages if m.get("role") == "user"), {"content": ""})
            user_content = str(user.get("content") or "")
            raw_texts = _generate_one(
                model,
                tokenizer,
                user_content,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                num_beams=max(1, args.num_beams),
                num_return_sequences=max(1, args.num_return_sequences),
                temperature=args.temperature,
                do_sample=args.sample,
            )
            prediction = raw_texts[0]
            candidates = normalize_candidates(prediction, raw_texts[1:])
            record = {
                "id": row.get("id"),
                "prediction": prediction,
                "candidates": [extract_answer_from_prediction(t) for t in raw_texts if extract_answer_from_prediction(t)],
                "raw_generations": raw_texts,
                "product": extract_product_from_user(user_content),
                "gold_answer": extract_gold_answer(row),
                "topology": (row.get("metadata") or {}).get("topology"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "out": str(args.out),
        "n": len(rows),
        "model_path": base,
        "adapter": str(args.adapter) if args.adapter else None,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "sample": args.sample,
        "use_4bit": not args.no_4bit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "manifest": str(manifest_path), "n": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
