#!/usr/bin/env python3
"""Train the trace-owned inverse actor through an explicit TRL tool facade."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from agent_model_init import build_trainable_model, lineage_report
from train_inverse_agent_trl import build_rows, load_yaml
from mechet.agent_env import AgentEnvConfig
from mechet.trl_environments import TraceOwnedTRLEnvironment


TRACE_SYSTEM_SUFFIX = """
Use the stateful electron-flow tools for every chemical claim. Import mapped
fragments only with import_fragment. Free-form proof submission is disabled.
Call finish_trace after the committed actions reach the intended precursor; the
environment replays declared moves, compiles MECH_PROOF v1, and derives the
endpoint. Abstain when the available evidence is insufficient."""


def augment_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        value = dict(row)
        messages = [dict(item) for item in row.get("prompt") or []]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                str(messages[0].get("content") or "").rstrip()
                + "\n"
                + TRACE_SYSTEM_SUFFIX.strip()
            )
        if len(messages) > 1 and messages[1].get("role") == "user":
            messages[1]["content"] = (
                str(messages[1].get("content") or "").rstrip()
                + "\nThe final precursor must come from finish_trace."
            )
        value["prompt"] = messages
        output.append(value)
    return output


def environment_config(cfg: dict[str, Any]) -> AgentEnvConfig:
    return AgentEnvConfig(**dict(cfg.get("environment") or {}))


def dry_run_report(
    cfg: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    env_cfg = environment_config(cfg)
    env = TraceOwnedTRLEnvironment(config=env_cfg)
    first = rows[0]
    observation = json.loads(env.reset(**first))
    inventory = json.loads(env.inspect_state())
    training = dict(cfg.get("training") or {})
    public_tools = sorted(
        name
        for name in dir(env)
        if not name.startswith("_") and name not in {"reset", "get_reward"}
    )
    return {
        "model_name_or_path": cfg.get("model_name_or_path"),
        "train_file": cfg.get("train_file"),
        "n_rows": len(rows),
        "first_target": first["target_smiles"],
        "environment": env_cfg.__dict__,
        "environment_class": type(env).__name__,
        "public_model_tools": public_tools,
        "observation_ok": bool(observation),
        "trace_owned": observation.get("faithfulness_contract"),
        "n_sources": len(inventory.get("sources") or []),
        "n_sinks": len(inventory.get("sinks") or []),
        "forward_checkpoint": cfg.get("forward_checkpoint") or None,
        "max_tool_calling_iterations": int(
            training.get("max_tool_calling_iterations", 8)
        ),
        "num_generations": int(training.get("num_generations", 8)),
        "seed": int(training.get("seed", 17)),
        "data_seed": int(training.get("data_seed", training.get("seed", 17))),
        "checkpoint_lineage": lineage_report(cfg),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_file = Path(str(cfg.get("train_file") or ""))
    if not train_file.exists():
        raise FileNotFoundError(f"train_file does not exist: {train_file}")
    rows = augment_prompts(
        build_rows(
            train_file,
            limit=args.limit or int(cfg.get("limit_examples", 0) or 0),
        )
    )

    if args.dry_run:
        print(json.dumps(dry_run_report(cfg, rows), indent=2, ensure_ascii=False))
        return 0

    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install mechet[agent]") from exc

    training = dict(cfg.get("training") or {})
    env_cfg = environment_config(cfg)
    seed = int(training.get("seed", 17))
    data_seed = int(training.get("data_seed", seed))
    bf16 = bool(
        training.get(
            "bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )
    )
    fp16 = bool(
        training.get(
            "fp16", torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
        )
    )
    tf32 = bool(training.get("tf32", torch.cuda.is_available()))
    grpo_args = GRPOConfig(
        output_dir=str(cfg.get("output_dir") or "outputs/agent/inverse_trace_grpo"),
        learning_rate=float(training.get("learning_rate", 5e-6)),
        num_train_epochs=float(training.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(
            training.get("per_device_train_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 8)
        ),
        num_generations=int(training.get("num_generations", 8)),
        max_completion_length=int(training.get("max_completion_length", 2048)),
        max_tool_calling_iterations=int(
            training.get("max_tool_calling_iterations", 8)
        ),
        temperature=float(training.get("temperature", 0.9)),
        top_p=float(training.get("top_p", 0.95)),
        beta=float(training.get("beta", 0.02)),
        use_vllm=bool(training.get("use_vllm", False)),
        logging_steps=int(training.get("logging_steps", 1)),
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        seed=seed,
        data_seed=data_seed,
        bf16=bf16,
        fp16=fp16,
        tf32=tf32,
        gradient_checkpointing=bool(
            training.get("gradient_checkpointing", True)
        ),
        dataloader_num_workers=int(training.get("dataloader_num_workers", 2)),
        report_to=list(training.get("report_to") or []),
        chat_template_kwargs={
            "enable_thinking": bool(training.get("enable_thinking", True))
        },
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )
    model, peft_config = build_trainable_model(cfg, torch)
    factory = partial(
        TraceOwnedTRLEnvironment,
        config=env_cfg,
        forward_checkpoint=cfg.get("forward_checkpoint") or None,
        forward_device=str(cfg.get("forward_device") or "cpu"),
    )
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=Dataset.from_list(rows),
        environment_factory=factory,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    output = Path(grpo_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lineage = lineage_report(cfg)
    lineage["training_runtime"] = {
        "seed": seed,
        "data_seed": data_seed,
        "bf16": bf16,
        "fp16": fp16,
        "tf32": tf32,
        "gradient_checkpointing": bool(
            training.get("gradient_checkpointing", True)
        ),
        "max_tool_calls": env_cfg.max_tool_calls,
        "max_tool_calling_iterations": int(
            training.get("max_tool_calling_iterations", 8)
        ),
    }
    (output / "checkpoint_lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
