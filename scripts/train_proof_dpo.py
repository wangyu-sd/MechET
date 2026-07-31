#!/usr/bin/env python3
"""Verifier-DPO for executable proofs versus deterministic failures.

Reference log-ratios are cached from the frozen initial SFT adapter before any
updates, avoiding a second full model copy.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mechet.model import resolve_qwen_model_path
from mechet.rlvr import completion_log_probs, normalize_sequence_log_prob
from train_mechet_rlvr import _build_model, _load_yaml


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def validate_row(row: dict) -> None:
    if not row.get("prompt_messages"):
        raise ValueError(f"row {row.get('id')} missing prompt_messages")
    if not str(row.get("chosen") or "").strip():
        raise ValueError(f"row {row.get('id')} missing chosen")
    if not str(row.get("rejected") or "").strip():
        raise ValueError(f"row {row.get('id')} missing rejected")


def sequence_logp(model, tokenizer, prompt, completion, max_length, length_normalize):
    total, n_tokens = completion_log_probs(
        model,
        tokenizer,
        prompt,
        completion,
        max_length=max_length,
    )
    return normalize_sequence_log_prob(total, n_tokens, length_normalize=length_normalize)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = _load_yaml(args.config)

    def path(value: str) -> Path:
        item = Path(value)
        return item if item.is_absolute() else REPO / item

    train_file = path(str(cfg["train_file"]))
    adapter = path(str(cfg["sft_adapter"]))
    output_dir = path(str(cfg["output_dir"]))
    rows = load_rows(train_file, args.limit or int(cfg.get("limit_examples", 0) or 0))
    for row in rows:
        validate_row(row)
    if not rows:
        raise ValueError("empty DPO dataset")
    if args.dry_run:
        verdicts = sorted({str(row.get("rejected_verdict") or "") for row in rows})
        print(json.dumps({
            "status": "dry_run",
            "n_rows": len(rows),
            "rejected_verdicts": verdicts,
            "policy": "executable preferred only over formally invalid",
        }, indent=2))
        return 0

    base = args.model_path or resolve_qwen_model_path() or os.environ.get("QWEN_MODEL_PATH") or ""
    if not base or not Path(base).exists():
        raise FileNotFoundError("set QWEN_MODEL_PATH or --model-path")
    if not adapter.exists():
        raise FileNotFoundError(f"missing SFT adapter: {adapter}")

    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW

    seed = int(cfg.get("seed", 11))
    random.seed(seed)
    torch.manual_seed(seed)
    model, tokenizer = _build_model(base, str(adapter), use_4bit=bool(cfg.get("use_qlora", True)))
    max_length = int(cfg.get("max_seq_length", 4096))
    length_normalize = bool(cfg.get("length_normalize", True))
    beta = float(cfg.get("beta", 0.1))

    # Cache the frozen initial-policy margins before updating the adapter.
    model.eval()
    reference_margins: list[float] = []
    with torch.no_grad():
        for row in rows:
            prompt = list(row["prompt_messages"])
            chosen = sequence_logp(model, tokenizer, prompt, str(row["chosen"]), max_length, length_normalize)
            rejected = sequence_logp(model, tokenizer, prompt, str(row["rejected"]), max_length, length_normalize)
            reference_margins.append(float((chosen - rejected).detach().cpu()))
    model.train()

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.get("learning_rate", 5e-6)),
    )
    max_steps = int(cfg.get("max_steps", 200))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    save_steps = int(cfg.get("save_steps", 50))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "dpo_steps.jsonl"
    optimizer.zero_grad()
    for step in range(max_steps):
        index = step % len(rows)
        row = rows[index]
        prompt = list(row["prompt_messages"])
        chosen = sequence_logp(model, tokenizer, prompt, str(row["chosen"]), max_length, length_normalize)
        rejected = sequence_logp(model, tokenizer, prompt, str(row["rejected"]), max_length, length_normalize)
        policy_margin = chosen - rejected
        reference_margin = torch.tensor(reference_margins[index], device=policy_margin.device, dtype=policy_margin.dtype)
        logit = beta * (policy_margin - reference_margin)
        loss = -F.logsigmoid(logit) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
        record = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()) * grad_accum,
            "policy_margin": float(policy_margin.detach().cpu()),
            "reference_margin": reference_margins[index],
            "preference_accuracy": float(policy_margin.detach().cpu()) > 0,
            "row_id": row.get("id"),
        }
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if save_steps and (step + 1) % save_steps == 0:
            model.save_pretrained(output_dir / f"adapter_step{step + 1}")

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "dpo_manifest.json").write_text(json.dumps({
        "status": "completed",
        "objective": "verifier_dpo",
        "reference": "cached_initial_sft_adapter",
        "beta": beta,
        "n_rows": len(rows),
        "max_steps": max_steps,
        "length_normalize": length_normalize,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
