#!/usr/bin/env python3
"""Train a certificate-conditioned proof repair adapter with span-weighted CE."""
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

from mechet.chat_template import apply_mechet_chat_template
from mechet.model import resolve_qwen_model_path
from train_mechet_rlvr import _build_model, _load_yaml


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def find_subsequence(sequence: list[int], query: list[int], start: int = 0) -> list[tuple[int, int]]:
    if not query:
        return []
    found: list[tuple[int, int]] = []
    for index in range(start, len(sequence) - len(query) + 1):
        if sequence[index : index + len(query)] == query:
            found.append((index, index + len(query)))
    return found


def encode_weighted_row(tokenizer, row: dict, max_length: int, changed_weight: float):
    messages = list(row.get("messages") or [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"row {row.get('id')} must end with assistant repair")
    prompt_messages = messages[:-1]
    completion = str(messages[-1].get("content") or "")
    prompt_text = apply_mechet_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = apply_mechet_chat_template(tokenizer, messages, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_len = min(len(prompt_ids), input_ids.shape[1])
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    weights = labels.ne(-100).float()
    token_values = input_ids[0].tolist()
    changed_lines = list((row.get("metadata") or {}).get("changed_lines") or [])
    for line in changed_lines:
        if not line or str(line).startswith("<DELETE:"):
            continue
        query = tokenizer(str(line).strip(), add_special_tokens=False)["input_ids"]
        for left, right in find_subsequence(token_values, query, start=prompt_len):
            weights[0, left:right] = float(changed_weight)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "weights": weights,
        "n_changed_lines": len(changed_lines),
        "completion": completion,
    }


def weighted_causal_loss(model, batch):
    import torch.nn.functional as F

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    weights = batch["weights"][:, 1:]
    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    weighted = token_loss * weights * valid.float()
    return weighted.sum() / (weights * valid.float()).sum().clamp_min(1.0)


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
    init_adapter = path(str(cfg["init_adapter"]))
    output_dir = path(str(cfg["output_dir"]))
    rows = load_rows(train_file, args.limit or int(cfg.get("limit_examples", 0) or 0))
    if not rows:
        raise ValueError("empty repair dataset")
    for row in rows:
        if str(row.get("task_type") or "") != "mech_proof_repair":
            raise ValueError(f"row {row.get('id')} is not a repair task")
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "n_rows": len(rows),
            "failure_codes": sorted({str((row.get('metadata') or {}).get('failure_code') or '') for row in rows}),
            "changed_span_weight": float(cfg.get("changed_span_weight", 4.0)),
        }, indent=2))
        return 0

    base = args.model_path or resolve_qwen_model_path() or os.environ.get("QWEN_MODEL_PATH") or ""
    if not base or not Path(base).exists():
        raise FileNotFoundError("set QWEN_MODEL_PATH or --model-path")
    if not init_adapter.exists():
        raise FileNotFoundError(f"missing init adapter: {init_adapter}")

    import torch
    from torch.optim import AdamW

    seed = int(cfg.get("seed", 11))
    random.seed(seed)
    torch.manual_seed(seed)
    model, tokenizer = _build_model(base, str(init_adapter), use_4bit=bool(cfg.get("use_qlora", True)))
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.get("learning_rate", 5e-6)),
    )
    max_steps = int(cfg.get("max_steps", 200))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    max_length = int(cfg.get("max_seq_length", 4096))
    changed_weight = float(cfg.get("changed_span_weight", 4.0))
    save_steps = int(cfg.get("save_steps", 50))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "repair_steps.jsonl"
    optimizer.zero_grad()
    for step in range(max_steps):
        row = rows[step % len(rows)]
        batch = encode_weighted_row(tokenizer, row, max_length, changed_weight)
        device = next(model.parameters()).device
        for key in ("input_ids", "attention_mask", "labels", "weights"):
            batch[key] = batch[key].to(device)
        loss = weighted_causal_loss(model, batch) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
        record = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()) * grad_accum,
            "row_id": row.get("id"),
            "failure_code": (row.get("metadata") or {}).get("failure_code"),
            "n_changed_lines": batch["n_changed_lines"],
        }
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if save_steps and (step + 1) % save_steps == 0:
            model.save_pretrained(output_dir / f"adapter_step{step + 1}")
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "repair_manifest.json").write_text(json.dumps({
        "status": "completed",
        "objective": "changed_span_weighted_ce",
        "changed_span_weight": changed_weight,
        "n_rows": len(rows),
        "max_steps": max_steps,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
