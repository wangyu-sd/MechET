#!/usr/bin/env python3
"""Self-MechVR: on-policy RLVR with local MECH_ET verifier (GRPO / RLOO)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.chat_template import build_generation_prompt
from mechet.model import resolve_qwen_model_path
from mechet.rlvr import (
    compute_advantages,
    compute_mechvr_reward,
    extract_prompt_messages,
    policy_loss_from_advantages,
)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
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
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    model.train()
    for _name, param in model.named_parameters():
        if "lora" not in _name.lower():
            param.requires_grad = False
    return model, tokenizer


def _generate_completions(
    model,
    tokenizer,
    prompt_messages: list[dict[str, str]],
    *,
    group_size: int,
    max_new_tokens: int,
    max_input_tokens: int,
    temperature: float,
    top_p: float,
):
    import torch

    texts: list[str] = []
    prompt = build_generation_prompt(tokenizer, prompt_messages)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(group_size):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            texts.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False))
    if was_training:
        model.train()
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--sft-adapter", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config) if args.config else {}

    def _cfg(key, default=None):
        cli_map = {
            "train_file": args.train_file,
            "sft_adapter": args.sft_adapter,
            "output_dir": args.output_dir,
            "max_steps": args.max_steps,
            "limit_examples": args.limit_examples,
        }
        cli_val = cli_map.get(key.replace("-", "_"))
        if cli_val is not None:
            return cli_val
        return cfg.get(key, default)

    train_file = Path(_cfg("train_file", REPO / "data/mechet_sft/overfit32/train.jsonl"))
    if not train_file.is_absolute():
        train_file = REPO / train_file
    sft_adapter = _cfg("sft_adapter", REPO / "outputs/mechet_overfit32/adapter")
    if sft_adapter and not Path(sft_adapter).is_absolute():
        sft_adapter = REPO / sft_adapter
    output_dir = Path(_cfg("output_dir", REPO / "outputs/mechet_rlvr"))
    if not output_dir.is_absolute():
        output_dir = REPO / output_dir

    max_steps = int(_cfg("max_steps", 20))
    group_size = int(_cfg("group_size", 4))
    prompts_per_step = int(_cfg("prompts_per_step", 1))
    learning_rate = float(_cfg("learning_rate", 1e-5))
    max_new_tokens = int(_cfg("max_new_tokens", 2048))
    max_input_tokens = int(_cfg("max_input_tokens", 4096))
    max_seq_length = int(_cfg("max_seq_length", 8192))
    temperature = float(_cfg("temperature", 0.9))
    top_p = float(_cfg("top_p", 0.95))
    advantage_method = str(_cfg("advantage_method", "grpo"))
    gate_penalty = float(_cfg("gate_penalty", -1.0))
    use_4bit = bool(_cfg("use_qlora", True))
    seed = int(_cfg("seed", 11))
    save_steps = int(_cfg("save_steps", 10))
    limit = int(args.limit_examples or _cfg("limit_examples", 0) or 0)

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    rows = _load_rows(train_file, limit)
    if not rows:
        print(json.dumps({"error": f"empty train file: {train_file}"}))
        return 1

    base = args.model_path or resolve_qwen_model_path() or os.environ.get("QWEN_MODEL_PATH") or ""
    if not base or not Path(base).exists():
        print(json.dumps({"error": "missing base model; set QWEN_MODEL_PATH"}))
        return 1

    adapter_path = str(sft_adapter) if sft_adapter and Path(sft_adapter).exists() else None
    if not adapter_path and not args.dry_run:
        print(json.dumps({"error": f"missing SFT adapter: {sft_adapter}"}))
        return 1

    if args.dry_run:
        sample = rows[0]
        fake = str((sample.get("messages") or [{}])[-1].get("content") or "")
        scored = compute_mechvr_reward(sample, fake, gate_penalty=gate_penalty)
        adv = compute_advantages([scored["rlvr_total"]] * group_size, method=advantage_method)
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "train_file": str(train_file),
                    "n_rows": len(rows),
                    "sample_gate_ok": scored.get("gate_ok"),
                    "sample_rlvr_total": scored.get("rlvr_total"),
                    "sample_advantages": adv,
                },
                indent=2,
            )
        )
        return 0

    import torch
    from torch.optim import AdamW

    model, tokenizer = _build_model(base, adapter_path, use_4bit=use_4bit)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "rlvr_steps.jsonl"
    metrics_path = output_dir / "rlvr_metrics.json"

    step_logs: list[dict] = []
    for step in range(max_steps):
        batch_rows = random.sample(rows, k=min(prompts_per_step, len(rows)))
        step_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        rewards_all: list[float] = []
        gate_pass = 0
        n_rollouts = 0

        for row in batch_rows:
            _, prompt_messages = extract_prompt_messages(row)
            completions = _generate_completions(
                model,
                tokenizer,
                prompt_messages,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            scored = [compute_mechvr_reward(row, text, gate_penalty=gate_penalty) for text in completions]
            rewards = [float(s["rlvr_total"]) for s in scored]
            advantages = compute_advantages(rewards, method=advantage_method)
            rewards_all.extend(rewards)
            gate_pass += sum(1 for s in scored if s.get("gate_ok"))
            n_rollouts += len(completions)

            loss, _pstats = policy_loss_from_advantages(
                model,
                tokenizer,
                prompt_messages,
                completions,
                advantages,
                max_length=max_seq_length,
            )
            step_loss = step_loss + loss

        step_loss = step_loss / max(len(batch_rows), 1)
        optimizer.zero_grad()
        if step_loss.requires_grad:
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

        record = {
            "step": step + 1,
            "loss": float(step_loss.detach().cpu()),
            "reward_mean": sum(rewards_all) / max(len(rewards_all), 1),
            "gate_pass_rate": gate_pass / max(n_rollouts, 1),
            "n_rollouts": n_rollouts,
            "prompts_per_step": len(batch_rows),
            "group_size": group_size,
        }
        step_logs.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))

        if save_steps and (step + 1) % save_steps == 0:
            ckpt = output_dir / f"adapter_step{step + 1}"
            model.save_pretrained(ckpt)

    final_adapter = output_dir / "adapter"
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    summary = {
        "status": "completed",
        "train_file": str(train_file),
        "sft_adapter": adapter_path,
        "output_dir": str(output_dir),
        "max_steps": max_steps,
        "group_size": group_size,
        "advantage_method": advantage_method,
        "steps": step_logs,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "adapter": str(final_adapter)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
