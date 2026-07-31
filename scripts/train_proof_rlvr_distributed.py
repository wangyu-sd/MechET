#!/usr/bin/env python3
"""Staged proof RLVR: vLLM rollout, CPU verification, and LoRA learning.

The three stages are file-backed so rollout workers, verifier workers, and the
learner can run on different machines. One iteration remains on-policy when the
rollout adapter and learner initialization identify the same checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mechet.model import resolve_qwen_model_path
from mechet.proof_rlvr import group_diagnostics, score_proof_group
from mechet.rlvr import compute_advantages, policy_loss_from_advantages
from train_mechet_rlvr import _build_model, _load_yaml


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_messages(row: dict) -> list[dict]:
    return [dict(item) for item in row.get("messages") or [] if item.get("role") != "assistant"]


def rollout_stage(cfg: dict, model_path: str, adapter: Path, data: Path, output: Path) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("rollout mode requires vllm") from exc
    from transformers import AutoTokenizer

    rows = load_jsonl(data)
    limit = int(cfg.get("limit_examples", 0) or 0)
    if limit:
        rows = rows[:limit]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    prompts = [
        tokenizer.apply_chat_template(prompt_messages(row), tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    sampling = SamplingParams(
        n=int(cfg.get("group_size", 8)),
        temperature=float(cfg.get("temperature", 0.9)),
        top_p=float(cfg.get("top_p", 0.95)),
        max_tokens=int(cfg.get("max_new_tokens", 2048)),
    )
    llm = LLM(
        model=model_path,
        enable_lora=True,
        tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
        max_model_len=int(cfg.get("max_model_len", 8192)),
    )
    try:
        from vllm.lora.request import LoRARequest
        request = LoRARequest("proof_policy", 1, str(adapter))
        outputs = llm.generate(prompts, sampling, lora_request=request)
    except Exception:
        # Older vLLM versions may accept the adapter through model construction
        # only; fail explicitly rather than silently rolling out the base model.
        raise RuntimeError("vLLM LoRA loading failed; verify the installed vLLM API")
    result_rows = []
    for row, generated in zip(rows, outputs):
        result_rows.append({
            "id": row.get("id"),
            "row": row,
            "prompt_messages": prompt_messages(row),
            "policy_adapter": str(adapter),
            "completions": [item.text for item in generated.outputs],
        })
    write_jsonl(output, result_rows)


def _score_one(payload):
    group, reward_config = payload
    scored = score_proof_group(group["row"], group["completions"], config=reward_config)
    rewards = [float(item["rlvr_total"]) for item in scored]
    return {
        **group,
        "scored": scored,
        "rewards": rewards,
        "diagnostics": group_diagnostics(scored),
    }


def score_stage(cfg: dict, rollout_file: Path, output: Path) -> None:
    groups = load_jsonl(rollout_file)
    workers = int(cfg.get("verifier_workers", max(os.cpu_count() or 1, 1)))
    payloads = [(group, dict(cfg.get("rewards") or {})) for group in groups]
    if workers <= 1:
        scored = [_score_one(item) for item in payloads]
    else:
        with Pool(processes=workers) as pool:
            scored = list(pool.map(_score_one, payloads))
    write_jsonl(output, scored)


def train_stage(cfg: dict, model_path: str, adapter: Path, scored_file: Path, output_dir: Path) -> None:
    import torch
    from torch.optim import AdamW

    groups = load_jsonl(scored_file)
    if not groups:
        raise ValueError("empty scored rollout file")
    # Guard against accidental off-policy mixing.
    expected_adapter = str(adapter)
    if any(str(group.get("policy_adapter") or "") != expected_adapter for group in groups):
        raise ValueError("rollout adapter does not match learner initialization")
    model, tokenizer = _build_model(model_path, str(adapter), use_4bit=bool(cfg.get("use_qlora", True)))
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.get("learning_rate", 5e-6)),
    )
    advantage_method = str(cfg.get("advantage_method", "rloo"))
    max_length = int(cfg.get("max_seq_length", 4096))
    max_steps = min(int(cfg.get("learner_steps", len(groups))), len(groups))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "learner_steps.jsonl"
    for step, group in enumerate(groups[:max_steps]):
        rewards = [float(value) for value in group.get("rewards") or []]
        advantages = compute_advantages(rewards, method=advantage_method)
        loss, stats = policy_loss_from_advantages(
            model,
            tokenizer,
            list(group["prompt_messages"]),
            list(group["completions"]),
            advantages,
            max_length=max_length,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        record = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "reward_mean": sum(rewards) / max(len(rewards), 1),
            "effective_group": any(abs(value) > 1e-8 for value in advantages),
            "group_diagnostics": group.get("diagnostics") or {},
            "policy_stats": stats,
        }
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["rollout", "score", "train"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--model-path", default="")
    args = parser.parse_args()
    cfg = _load_yaml(args.config)
    base = args.model_path or resolve_qwen_model_path() or os.environ.get("QWEN_MODEL_PATH") or ""
    if args.mode in {"rollout", "train"} and (not base or not Path(base).exists()):
        raise FileNotFoundError("set QWEN_MODEL_PATH or --model-path")
    if args.mode in {"rollout", "train"} and (args.adapter is None or not args.adapter.exists()):
        raise FileNotFoundError("--adapter is required for rollout/train")
    if args.mode == "rollout":
        rollout_stage(cfg, base, args.adapter, args.input, args.output)
    elif args.mode == "score":
        score_stage(cfg, args.input, args.output)
    else:
        train_stage(cfg, base, args.adapter, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
