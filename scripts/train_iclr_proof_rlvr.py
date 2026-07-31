#!/usr/bin/env python3
"""On-policy ICLR proof RLVR with structural-endpoint rewards."""
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
sys.path.insert(0, str(REPO / "scripts"))

from mechet.iclr_rewards import compute_core_proof_reward
from mechet.model import resolve_qwen_model_path
from mechet.rlvr import compute_advantages, extract_prompt_messages, policy_loss_from_advantages
from train_mechet_rlvr import _build_model, _generate_completions, _load_rows, _load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = _load_yaml(args.config)

    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPO / path

    train_file = resolve_path(str(cfg["train_file"]))
    adapter = resolve_path(str(cfg["sft_adapter"]))
    output_dir = resolve_path(str(cfg["output_dir"]))
    rows = _load_rows(train_file, int(cfg.get("limit_examples", 0) or 0))
    if not rows:
        raise ValueError(f"empty training file: {train_file}")

    group_size = int(cfg.get("group_size", 4))
    prompts_per_step = int(cfg.get("prompts_per_step", 2))
    max_steps = int(cfg.get("max_steps", 100))
    max_new_tokens = int(cfg.get("max_new_tokens", 2048))
    max_input_tokens = int(cfg.get("max_input_tokens", 4096))
    max_seq_length = int(cfg.get("max_seq_length", 4096))
    temperature = float(cfg.get("temperature", 0.85))
    top_p = float(cfg.get("top_p", 0.95))
    advantage_method = str(cfg.get("advantage_method", "rloo"))
    seed = int(cfg.get("seed", 11))
    reward_config = dict(cfg.get("rewards") or {})
    random.seed(seed)

    if args.dry_run:
        sample = rows[0]
        gold = str((sample.get("messages") or [{}])[-1].get("content") or "")
        scored = compute_core_proof_reward(sample, gold, config=reward_config)
        print(json.dumps({"status": "dry_run", "reward": scored}, indent=2))
        return 0

    base = args.model_path or resolve_qwen_model_path() or os.environ.get("QWEN_MODEL_PATH") or ""
    if not base or not Path(base).exists():
        raise FileNotFoundError("set QWEN_MODEL_PATH or --model-path")
    if not adapter.exists():
        raise FileNotFoundError(f"missing SFT adapter: {adapter}")

    import torch
    from torch.optim import AdamW

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, tokenizer = _build_model(base, str(adapter), use_4bit=bool(cfg.get("use_qlora", True)))
    optimizer = AdamW([param for param in model.parameters() if param.requires_grad], lr=float(cfg.get("learning_rate", 5e-6)))
    output_dir.mkdir(parents=True, exist_ok=True)
    step_path = output_dir / "rlvr_steps.jsonl"
    logs: list[dict] = []

    for step in range(max_steps):
        batch = random.sample(rows, k=min(prompts_per_step, len(rows)))
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        rewards_all: list[float] = []
        execute = endpoint = composition = rollouts = effective_groups = 0
        for row in batch:
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
            scored = [compute_core_proof_reward(row, text, config=reward_config) for text in completions]
            rewards = [float(item["rlvr_total"]) for item in scored]
            advantages = compute_advantages(rewards, method=advantage_method)
            effective_groups += int(any(abs(value) > 1e-8 for value in advantages))
            loss, _ = policy_loss_from_advantages(
                model,
                tokenizer,
                prompt_messages,
                completions,
                advantages,
                max_length=max_seq_length,
            )
            total_loss = total_loss + loss
            rewards_all.extend(rewards)
            execute += sum(bool(item.get("execute_ok")) for item in scored)
            endpoint += sum(bool(item.get("endpoint_core_exact")) for item in scored)
            composition += sum(bool(item.get("composition_match")) for item in scored)
            rollouts += len(scored)

        total_loss = total_loss / max(len(batch), 1)
        optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        record = {
            "step": step + 1,
            "loss": float(total_loss.detach().cpu()),
            "reward_mean": sum(rewards_all) / max(len(rewards_all), 1),
            "execute_rate": execute / max(rollouts, 1),
            "endpoint_core_exact_rate": endpoint / max(rollouts, 1),
            "composition_match_rate": composition / max(rollouts, 1),
            "effective_group_rate": effective_groups / max(len(batch), 1),
            "n_rollouts": rollouts,
        }
        logs.append(record)
        with step_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        save_steps = int(cfg.get("save_steps", 25))
        if save_steps and (step + 1) % save_steps == 0:
            model.save_pretrained(output_dir / f"adapter_step{step + 1}")

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    summary = {
        "status": "completed",
        "objective": "group_relative_reinforce",
        "advantage_method": advantage_method,
        "reward_config": reward_config,
        "steps": logs,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "rlvr_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
