#!/usr/bin/env python3
"""Train the inverse actor with TRL's stateful agentic GRPO interface.

The chemistry environment remains framework-neutral in ``mechet.agent_env``.
This entrypoint is the recommended small-scale reference adapter. Use ``--dry-run``
to validate data, reward settings, and tool schemas without importing TRL.
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

from mechet.agent_env import AgentEnvConfig, MechETAgentEnv


SYSTEM_PROMPT = """You are MechET, an inverse electron-flow reasoning agent.
Use the chemistry tools to inspect atom-mapped states and test explicit electron
moves. The final precursor must be derived by submitting one executable
MECH_PROOF v1 program. Never invent atom maps that are absent from the state or
imports, and abstain when chemical support is insufficient."""


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("install PyYAML or mechet[agent]") from exc
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def target_from_row(row: dict[str, Any]) -> str:
    for key in ("target_smiles", "target", "product", "products"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    metadata = row.get("metadata") or {}
    for key in ("target_smiles", "product", "products"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    for message in row.get("messages") or []:
        content = str(message.get("content") or "")
        for line in content.splitlines():
            if line.strip().upper().startswith("TARGET:"):
                return line.split(":", 1)[1].strip()
    return ""


def expected_from_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(
        row.get("expected_precursor")
        or metadata.get("core_precursor")
        or metadata.get("derived_precursor")
        or metadata.get("initial_reactants")
        or ""
    ).strip()


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    target = target_from_row(row)
    if not target:
        raise ValueError("training row has no target product")
    competitors = row.get("competitor_products") or (row.get("metadata") or {}).get(
        "competitor_products"
    ) or []
    conditions = row.get("conditions") or (row.get("metadata") or {}).get("conditions")
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TARGET: {target}\n"
                    "Reason backward with explicit electron-flow tools and submit "
                    "an executable inverse proof."
                ),
            },
        ],
        "target_smiles": target,
        "expected_precursor": expected_from_row(row),
        "competitor_products": competitors,
        "conditions": conditions,
        "source_id": str(row.get("id") or ""),
    }


def build_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    output = []
    rejected = 0
    for row in read_jsonl(path, limit=limit):
        try:
            output.append(normalize_row(row))
        except Exception:
            rejected += 1
    if not output:
        raise ValueError(f"no usable rows in {path}; rejected={rejected}")
    return output


def dry_run_report(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    env_cfg = AgentEnvConfig(**dict(cfg.get("environment") or {}))
    env = MechETAgentEnv(config=env_cfg)
    first = rows[0]
    observation = env.reset(**first)
    inventory = json.loads(env.inspect_state())
    return {
        "model_name_or_path": cfg.get("model_name_or_path"),
        "train_file": cfg.get("train_file"),
        "n_rows": len(rows),
        "first_target": first["target_smiles"],
        "environment": env_cfg.__dict__,
        "observation_ok": bool(observation),
        "n_sources": len(inventory.get("sources") or []),
        "n_sinks": len(inventory.get("sinks") or []),
        "forward_checkpoint": cfg.get("forward_checkpoint") or None,
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
    rows = build_rows(train_file, limit=args.limit or int(cfg.get("limit_examples", 0) or 0))

    if args.dry_run:
        print(json.dumps(dry_run_report(cfg, rows), indent=2, ensure_ascii=False))
        return 0

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "agent training requires mechet[agent] and a TRL release with environment_factory"
        ) from exc

    model_name = str(cfg.get("model_name_or_path") or "")
    if not model_name:
        raise ValueError("model_name_or_path is required")
    training = dict(cfg.get("training") or {})
    lora = dict(cfg.get("lora") or {})
    env_cfg = AgentEnvConfig(**dict(cfg.get("environment") or {}))
    forward_checkpoint = cfg.get("forward_checkpoint") or None
    forward_device = str(cfg.get("forward_device") or "cpu")

    grpo_args = GRPOConfig(
        output_dir=str(cfg.get("output_dir") or "outputs/agent/inverse_trl_grpo"),
        learning_rate=float(training.get("learning_rate", 5e-6)),
        num_train_epochs=float(training.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        num_generations=int(training.get("num_generations", 8)),
        max_completion_length=int(training.get("max_completion_length", 2048)),
        logging_steps=int(training.get("logging_steps", 1)),
        save_steps=int(training.get("save_steps", 100)),
        bf16=bool(training.get("bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())),
        fp16=bool(training.get("fp16", torch.cuda.is_available() and not torch.cuda.is_bf16_supported())),
        report_to=list(training.get("report_to") or []),
        chat_template_kwargs={
            "enable_thinking": bool(training.get("enable_thinking", True))
        },
    )
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
    )
    environment_factory = partial(
        MechETAgentEnv,
        config=env_cfg,
        forward_checkpoint=forward_checkpoint,
        forward_device=forward_device,
    )
    trainer = GRPOTrainer(
        model=model_name,
        args=grpo_args,
        train_dataset=Dataset.from_list(rows),
        environment_factory=environment_factory,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
