#!/usr/bin/env python3
"""Train evidence-conditioned trace-owned actors through explicit TRL facades."""
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
from train_inverse_agent_trace import augment_prompts
from train_inverse_agent_trl import build_rows, load_yaml
from mechet.anchor_trl_environment import AnchorTraceOwnedTRLEnvironment
from mechet.knowledge_agent_env import KnowledgeAgentConfig
from mechet.trl_environments import (
    TextbookAnchorTraceOwnedTRLEnvironment,
    TextbookTraceOwnedTRLEnvironment,
)


KNOWLEDGE_SUFFIX = """
Mechanistic evidence tools provide soft external evidence, never answers,
rewards, or chemical truth. Ground useful principles into mapped electron-flow
actions and rely on finish_trace for the only proof and precursor."""


def knowledge_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = augment_prompts(rows)
    for row in output:
        messages = row.get("prompt") or []
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                str(messages[0].get("content") or "").rstrip()
                + "\n"
                + KNOWLEDGE_SUFFIX.strip()
            )
    return output


def evidence_mode(cfg: dict[str, Any]) -> str:
    value = str(cfg.get("evidence_mode") or "textbook").strip().lower()
    if value not in {"textbook", "anchors", "combined"}:
        raise ValueError(f"unknown evidence_mode: {value}")
    return value


def environment_config(cfg: dict[str, Any]) -> KnowledgeAgentConfig:
    mode = evidence_mode(cfg)
    payload = dict(cfg.get("environment") or {})
    payload.setdefault(
        "textbook_corpus_path",
        str(
            cfg.get("textbook_corpus_path")
            or REPO / "knowledge/corpus/passages.jsonl"
        ),
    )
    payload.setdefault(
        "primitive_library_path",
        str(
            cfg.get("primitive_library_path")
            or REPO / "knowledge/primitives/core_polar_primitives.yaml"
        ),
    )
    payload.setdefault(
        "primitive_source_registry_path",
        str(
            cfg.get("primitive_source_registry_path")
            or REPO / "knowledge/source_registry.yaml"
        ),
    )
    payload["enable_structured_primitives"] = mode in {"anchors", "combined"}
    payload["require_textbook_corpus"] = mode in {"textbook", "combined"}
    return KnowledgeAgentConfig(**payload)


def environment_class(cfg: dict[str, Any]):
    mode = evidence_mode(cfg)
    if mode == "anchors":
        return AnchorTraceOwnedTRLEnvironment
    if mode == "combined":
        return TextbookAnchorTraceOwnedTRLEnvironment
    return TextbookTraceOwnedTRLEnvironment


def dry_run_report(
    cfg: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    mode = evidence_mode(cfg)
    env_cfg = environment_config(cfg)
    cls = environment_class(cfg)
    env = cls(config=env_cfg)
    first = rows[0]
    observation = json.loads(env.reset(**first))
    inventory = json.loads(env.inspect_state())
    textbook = (
        json.loads(env.retrieve_textbook_guidance())
        if mode in {"textbook", "combined"}
        else {"ok": False, "code": "TEXTBOOK_TOOL_NOT_EXPOSED"}
    )
    anchors = (
        json.loads(env.retrieve_primitives())
        if mode in {"anchors", "combined"}
        else {"ok": False, "code": "ANCHOR_TOOL_NOT_EXPOSED"}
    )
    training = dict(cfg.get("training") or {})
    public_tools = sorted(
        name
        for name in dir(env)
        if not name.startswith("_") and name not in {"reset", "get_reward"}
    )
    return {
        "model_name_or_path": cfg.get("model_name_or_path"),
        "train_file": cfg.get("train_file"),
        "evidence_mode": mode,
        "n_rows": len(rows),
        "first_target": first["target_smiles"],
        "environment": env_cfg.__dict__,
        "environment_class": cls.__name__,
        "public_model_tools": public_tools,
        "observation_ok": bool(observation),
        "trace_owned": observation.get("faithfulness_contract"),
        "knowledge": observation.get("knowledge"),
        "n_sources": len(inventory.get("sources") or []),
        "n_sinks": len(inventory.get("sinks") or []),
        "n_textbook_matches": len(textbook.get("matches") or []),
        "textbook_context_hash": (textbook.get("context") or {}).get(
            "context_sha256"
        ),
        "n_anchor_matches": len(anchors.get("matches") or []),
        "forward_checkpoint": cfg.get("forward_checkpoint") or None,
        "max_tool_calling_iterations": int(
            training.get("max_tool_calling_iterations", 10)
        ),
        "num_generations": int(training.get("num_generations", 8)),
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
    rows = knowledge_prompts(
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
        raise RuntimeError("install mechet[agent,knowledge]") from exc

    training = dict(cfg.get("training") or {})
    env_cfg = environment_config(cfg)
    cls = environment_class(cfg)
    grpo_args = GRPOConfig(
        output_dir=str(cfg.get("output_dir") or "outputs/agent/inverse_knowledge_grpo"),
        learning_rate=float(training.get("learning_rate", 5e-6)),
        num_train_epochs=float(training.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        num_generations=int(training.get("num_generations", 8)),
        max_completion_length=int(training.get("max_completion_length", 2048)),
        max_tool_calling_iterations=int(training.get("max_tool_calling_iterations", 10)),
        temperature=float(training.get("temperature", 0.9)),
        top_p=float(training.get("top_p", 0.95)),
        beta=float(training.get("beta", 0.02)),
        use_vllm=bool(training.get("use_vllm", False)),
        logging_steps=int(training.get("logging_steps", 1)),
        save_steps=int(training.get("save_steps", 100)),
        bf16=bool(
            training.get(
                "bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            )
        ),
        fp16=bool(
            training.get(
                "fp16", torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
            )
        ),
        report_to=list(training.get("report_to") or []),
        chat_template_kwargs={
            "enable_thinking": bool(training.get("enable_thinking", True))
        },
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )
    model, peft_config = build_trainable_model(cfg, torch)
    factory = partial(
        cls,
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
    (output / "checkpoint_lineage.json").write_text(
        json.dumps(lineage_report(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
