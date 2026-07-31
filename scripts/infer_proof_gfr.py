#!/usr/bin/env python3
"""Generate--Falsify--Repair inference with deterministic failure certificates."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.model import resolve_qwen_model_path
from mechet.proof_diagnostics import diagnose_proof, format_repair_feedback
from mechet.proof_hypotheses import score_hypothesis
from mechet.proof_program import execute_proof


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def expected_precursor(row: dict) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("core_precursor") or metadata.get("derived_precursor") or "")


def generate(model, tokenizer, messages, *, max_new_tokens, temperature, top_p):
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    if torch.cuda.is_available():
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    return tokenizer.decode(output[encoded["input_ids"].shape[1] :], skip_special_tokens=False)


def repair_messages(product: str, proof: str, feedback: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair the MECH_PROOF v1 program using the deterministic failure "
                "certificate. Preserve all unaffected lines and return only one "
                "complete <proof> block."
            ),
        },
        {
            "role": "user",
            "content": f"TARGET: {product}\nINVALID_PROOF:\n{proof}\nCERTIFICATE:\n{feedback}",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actor-adapter", type=Path, required=True)
    parser.add_argument("--repair-adapter", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--samples-per-target", type=int, default=16)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repair-temperature", type=float, default=0.2)
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
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(args.actor_adapter),
        adapter_name="actor",
        is_trainable=False,
    )
    model.load_adapter(str(args.repair_adapter), adapter_name="repair", is_trainable=False)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle, torch.no_grad():
        for row in load_rows(args.data, args.limit):
            actor_messages = [dict(item) for item in row.get("messages") or [] if item.get("role") != "assistant"]
            product = ""
            for message in actor_messages:
                if message.get("role") == "user" and str(message.get("content") or "").startswith("TARGET:"):
                    product = str(message.get("content") or "").split("TARGET:", 1)[1].strip().splitlines()[0]
                    break
            candidates = []
            for sample_index in range(args.samples_per_target):
                model.set_adapter("actor")
                proof = generate(
                    model,
                    tokenizer,
                    actor_messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                history = [{"round": 0, "proof": proof, "execute_ok": execute_proof(proof).ok}]
                for repair_round in range(1, args.max_repairs + 1):
                    if execute_proof(proof).ok:
                        break
                    certificate = diagnose_proof(proof)
                    if certificate is None:
                        break
                    feedback = format_repair_feedback(certificate)
                    model.set_adapter("repair")
                    proof = generate(
                        model,
                        tokenizer,
                        repair_messages(product, proof, feedback),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.repair_temperature,
                        top_p=args.top_p,
                    )
                    history.append(
                        {
                            "round": repair_round,
                            "proof": proof,
                            "execute_ok": execute_proof(proof).ok,
                            "certificate": feedback,
                        }
                    )
                scored = score_hypothesis(
                    proof,
                    source_index=sample_index,
                    expected_precursor=expected_precursor(row) or None,
                    repaired=len(history) > 1,
                    metadata={"repair_history": history},
                )
                candidates.append(scored.to_dict())
            handle.write(json.dumps({
                "id": row.get("id"),
                "candidates": candidates,
                "config": {
                    "samples_per_target": args.samples_per_target,
                    "max_repairs": args.max_repairs,
                },
            }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
