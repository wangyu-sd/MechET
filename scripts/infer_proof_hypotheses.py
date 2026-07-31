#!/usr/bin/env python3
"""Sample, execute, deduplicate, and rank sets of proof hypotheses."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.model import resolve_qwen_model_path
from mechet.proof_curriculum import proof_text_from_row
from mechet.proof_hypotheses import (
    deduplicate_hypotheses,
    endpoint_groups,
    rank_hypotheses,
    score_hypothesis,
    summarize_hypotheses,
    survival_curve,
)


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def expected_precursor(row: dict) -> str:
    metadata = row.get("metadata") or {}
    return str(
        metadata.get("core_precursor")
        or metadata.get("derived_precursor")
        or metadata.get("initial_reactants")
        or ""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--samples-per-target", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base = args.model_path or os.environ.get("QWEN_MODEL_PATH") or resolve_qwen_model_path() or ""
    if not base or not Path(base).exists():
        raise FileNotFoundError("set QWEN_MODEL_PATH or --model-path")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    quant = None
    if torch.cuda.is_available() and not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=True)
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
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle, torch.no_grad():
        for row in load_rows(args.data, args.limit):
            prompt_messages = [
                dict(message)
                for message in row.get("messages") or []
                if message.get("role") != "assistant"
            ]
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            if torch.cuda.is_available():
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
            hypotheses = []
            for sample_index in range(args.samples_per_target):
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=max(args.temperature, 1e-5),
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sequence = output.sequences[0]
                generated = sequence[encoded["input_ids"].shape[1] :]
                prediction = tokenizer.decode(generated, skip_special_tokens=False)
                logprob = 0.0
                if output.scores:
                    for token, scores in zip(generated, output.scores):
                        logprob += float(torch.log_softmax(scores[0], dim=-1)[token].detach().cpu())
                    logprob /= max(len(generated), 1)
                hypotheses.append(
                    score_hypothesis(
                        prediction,
                        source_index=sample_index,
                        expected_precursor=expected_precursor(row) or None,
                        gold_proof=proof_text_from_row(row) or None,
                        model_logprob=logprob,
                    )
                )
            unique = deduplicate_hypotheses(hypotheses)
            ranked = rank_hypotheses(unique)
            payload = {
                "id": row.get("id"),
                "summary": summarize_hypotheses(hypotheses).to_dict(),
                "survival": survival_curve(hypotheses),
                "endpoint_groups": endpoint_groups(unique),
                "hypotheses": [item.to_dict() for item in ranked],
                "generation": {
                    "samples_per_target": args.samples_per_target,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
