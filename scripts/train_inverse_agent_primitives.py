#!/usr/bin/env python3
"""Train the inverse actor with TRL and the mechanistic primitive library.

This is a matched alternative to ``train_inverse_agent_trl.py``. The only
method change is the environment: the actor can retrieve reviewed primitive
records and optionally receive a bounded soft primitive-support reward. The
executor remains the hard validity gate.
"""
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

from train_inverse_agent_trl import build_rows, load_yaml
from mechet.primitive_agent_env import PrimitiveAgentConfig, PrimitiveAugmentedAgentEnv

PRIMITIVE_SYSTEM_SUFFIX = """
A reviewed mechanistic primitive library is available through
retrieve_primitives. Use it to identify reusable motifs, bind generic roles to
current atom maps, inspect competing pathways, and obtain candidate E_MOVE sets.
Primitive records are soft guidance: never override tool execution, never claim
selectivity without explicit competitors, and abstain when evidence is weak."""


def augment_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        value = dict(row)
        messages = [dict(item) for item in row.get("prompt") or []]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = str(messages[0].get("content") or "").rstrip() + "\n" + PRIMITIVE_SYSTEM_SUFFIX.strip()
        value["prompt"] = messages
        output.append(value)
    return output


def environment_config(cfg: dict[str, Any]) -> PrimitiveAgentConfig:
    payload = dict(cfg.get("environment") or {})
    payload.setdefault("primitive_library_path", str(cfg.get("primitive_library_path") or REPO / "knowledge/primitives/core_polar_primitives.yaml"))
    payload.setdefault("primitive_source_registry_path", str(cfg.get("primitive_source_registry_path") or REPO / "knowledge/source_registry.yaml"))
    return PrimitiveAgentConfig(**payload)


def dry_run_report(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    env_cfg = environment_config(cfg)
    env = PrimitiveAugmentedAgentEnv(config=env_cfg)
    first = rows[0]
    observation = json.loads(env.reset(**first))
    inventory = json.loads(env.inspect_state())
    primitives = json.loads(env.retrieve_primitives())
    training = dict(cfg.get("training") or {})
    return {
        "model_name_or_path": cfg.get("model_name_or_path"),
        "train_file": cfg.get("train_file"),
        "n_rows": len(rows),
        "first_target": first["target_smiles"],
        "environment": env_cfg.__dict__,
        "observation_ok": bool(observation),
        "n_sources": len(inventory.get("sources") or []),
        "n_sinks": len(inventory.get("sinks") or []),
        "n_primitive_matches": len(primitives.get("matches") or []),
        "primitive_library": env.primitive_library.manifest(),
        "forward_checkpoint": cfg.get("forward_checkpoint") or None,
        "max_tool_calling_iterations": int(training.get("max_tool_calling_iterations", 8)),
        "num_generations": int(training.get("num_generations", 8)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    train_file = Path(str(cfg.get("train_file") or ""))
    if not train_file.exists(): raise FileNotFoundError(f"train_file does not exist: {train_file}")
    rows = augment_prompts(build_rows(train_file, limit=args.limit or int(cfg.get("limit_examples", 0) or 0)))
    if args.dry_run:
        print(json.dumps(dry_run_report(cfg, rows), indent=2, ensure_ascii=False)); return 0
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install mechet[agent,knowledge]") from exc
    model_name = str(cfg.get("model_name_or_path") or "")
    if not model_name: raise ValueError("model_name_or_path is required")
    training, lora = dict(cfg.get("training") or {}), dict(cfg.get("lora") or {})
    env_cfg = environment_config(cfg)
    grpo_args = GRPOConfig(
        output_dir=str(cfg.get("output_dir") or "outputs/agent/inverse_primitives_trl_grpo"),
        learning_rate=float(training.get("learning_rate", 5e-6)),
        num_train_epochs=float(training.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        num_generations=int(training.get("num_generations", 8)),
        max_completion_length=int(training.get("max_completion_length", 2048)),
        max_tool_calling_iterations=int(training.get("max_tool_calling_iterations", 8)),
        temperature=float(training.get("temperature", 0.9)),
        top_p=float(training.get("top_p", 0.95)),
        beta=float(training.get("beta", 0.02)),
        use_vllm=bool(training.get("use_vllm", False)),
        logging_steps=int(training.get("logging_steps", 1)),
        save_steps=int(training.get("save_steps", 100)),
        bf16=bool(training.get("bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())),
        fp16=bool(training.get("fp16", torch.cuda.is_available() and not torch.cuda.is_bf16_supported())),
        report_to=list(training.get("report_to") or []),
        chat_template_kwargs={"enable_thinking": bool(training.get("enable_thinking", True))},
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
    )
    factory = partial(
        PrimitiveAugmentedAgentEnv,
        config=env_cfg,
        forward_checkpoint=cfg.get("forward_checkpoint") or None,
        forward_device=str(cfg.get("forward_device") or "cpu"),
    )
    trainer = GRPOTrainer(model=model_name, args=grpo_args, train_dataset=Dataset.from_list(rows), environment_factory=factory, peft_config=peft_config)
    trainer.train(); trainer.save_model(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
