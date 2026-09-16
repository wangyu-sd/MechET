#!/usr/bin/env python3
"""GRPO refinement for fixed-template electron programs with strict rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from agent_model_init import build_trainable_model, lineage_report, path_sha256, validate_lineage
from mechet.assistant_masking import render_chat
from mechet.python_template_rlvr import score_template_rlvr_candidate, template_rlvr_rewards


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("RLVR config must be a mapping")
    return dict(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            row_id = str(source.get("id") or "")
            messages = [dict(item) for item in source.get("messages") or ()]
            if not row_id or row_id in seen:
                raise ValueError(f"missing or duplicate row id: {row_id}")
            if str(source.get("task_type") or "") != "python_template_slots_retro":
                raise ValueError(f"{row_id} has wrong task_type")
            if len(messages) != 3 or messages[-1].get("role") != "assistant":
                raise ValueError(f"{row_id} has an invalid supervision conversation")
            seen.add(row_id)
            rows.append(
                {
                    "id": row_id,
                    "prompt": messages[:-1],
                    "target_smiles": str(source.get("target_smiles") or ""),
                    "expected_precursor": str(source.get("expected_precursor") or ""),
                }
            )
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"empty RLVR data: {path}")
    return rows


def _validate_contract(cfg: dict[str, Any], train_file: Path) -> dict[str, Any]:
    manifest_path = Path(str(cfg["stable_id_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "python_template_slots_sft_v1":
        raise ValueError("RLVR requires python_template_slots_sft_v1")
    if manifest.get("input_contract") != "mapped_product_only_with_fixed_runtime_template":
        raise ValueError("RLVR input is not mapped-product-only")
    if manifest.get("output_contract") != "steps_slot_only_executor_derived_endpoint":
        raise ValueError("RLVR output contract is not executor-owned")
    split = dict(manifest["splits"]["train"])
    if int(split["accepted_rows"]) != int(cfg["expected_train_rows"]):
        raise ValueError("RLVR train row count differs from the frozen manifest")
    if _sha256(train_file) != str(split["output_sha256"]):
        raise ValueError("RLVR train-file hash differs from the frozen manifest")
    lineage = validate_lineage(cfg)
    expected_parent = str(cfg.get("initial_adapter_sha256") or "")
    if expected_parent.lower() == "auto":
        expected_parent = str(lineage.get("initial_adapter_sha256_declared") or "")
    if expected_parent != str(lineage.get("initial_adapter_sha256_actual") or ""):
        raise ValueError("RLVR parent adapter hash mismatch")
    return {"dataset_manifest": manifest, "adapter_lineage": lineage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    train_file = Path(str(cfg["train_file"]))
    limit = int(cfg.get("limit_examples") or 0)
    contract = _validate_contract(cfg, train_file)
    rows = _load_rows(train_file, limit=limit)
    id_digest = hashlib.sha256(
        "\n".join(str(row["id"]) for row in rows).encode("utf-8")
    ).hexdigest()
    gold_source = json.loads(train_file.open(encoding="utf-8").readline())
    gold = str(gold_source["messages"][-1]["content"])
    gold_score = score_template_rlvr_candidate(gold_source, gold)
    invalid_score = score_template_rlvr_candidate(gold_source, "not a STEPS list")
    preflight = {
        "artifact_type": "python_template_slots_rlvr_preflight_v1",
        "train_file": str(train_file),
        "train_file_sha256": _sha256(train_file),
        "selected_rows": len(rows),
        "selected_id_sha256": id_digest,
        "parent_adapter": cfg["initial_adapter_path"],
        "parent_adapter_sha256": contract["adapter_lineage"]["initial_adapter_sha256_actual"],
        "gold_reward": gold_score["reward"],
        "invalid_reward": invalid_score["reward"],
        "reward": {
            "parse_failure": -1.0,
            "executable_prefix": "fraction in [0,1]",
            "formal_execution_bonus": 2.0,
            "structural_precursor_bonus": 4.0,
            "teacher_path_matching": False,
        },
    }
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    training = dict(cfg.get("training") or {})
    repair = dict(cfg.get("repair") or {})
    if repair:
        import trl
        if trl.__version__ != "0.21.0":
            raise ValueError(f"Repair trainer is tested against TRL 0.21.0, got {trl.__version__}")
        if cfg.get("resume_from_checkpoint", False):
            raise ValueError("Repair pilot must start from the formal SFT, not the old RLVR run")
        if any(Path(cfg["output_dir"]).glob("checkpoint-*/trainer_state.json")):
            raise ValueError("Repair output already contains checkpoints; choose a new output directory")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    model, peft_config = build_trainable_model(cfg, torch)
    if repair:
        model.load_adapter(str(cfg["initial_adapter_path"]), adapter_name="sft_reference", is_trainable=False)
        model.set_adapter("default")
        for name, param in model.named_parameters():
            if ".sft_reference." in name:
                param.requires_grad_(False)
        assert any(p.requires_grad for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg["model_name_or_path"]),
        revision=contract["adapter_lineage"]["resolved_model_revision"],
        trust_remote_code=bool(training.get("trust_remote_code", True)),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Render once with the same explicit no-thinking contract used by SFT and
    # standalone inference. Passing plain prompt strings prevents TRL from
    # silently selecting Qwen's default thinking-mode chat-template branch.
    for row in rows:
        row["prompt"] = render_chat(
            tokenizer,
            list(row["prompt"]),
            add_generation_prompt=True,
        )

    grpo_args = GRPOConfig(
        output_dir=str(cfg["output_dir"]),
        learning_rate=float(training.get("learning_rate", 2e-6)),
        max_steps=int(training.get("max_steps", 100)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 2)),
        num_generations=int(training.get("num_generations", 8)),
        max_prompt_length=int(training.get("max_prompt_length", 2048)),
        max_completion_length=int(training.get("max_completion_length", 1536)),
        temperature=float(training.get("temperature", 0.9)),
        top_p=float(training.get("top_p", 0.95)),
        top_k=training.get("top_k", 50),
        repetition_penalty=float(training.get("repetition_penalty", 1.05)),
        beta=float(training.get("beta", 0.02)),
        epsilon=float(training.get("epsilon", 0.2)),
        loss_type=str(training.get("loss_type", "dr_grpo")),
        scale_rewards=training.get("scale_rewards", "group"),
        mask_truncated_completions=bool(training.get("mask_truncated_completions", False)),
        use_vllm=bool(training.get("use_vllm", False)),
        vllm_mode=str(training.get("vllm_mode", "colocate")),
        vllm_gpu_memory_utilization=float(training.get("vllm_gpu_memory_utilization", 0.45)),
        vllm_tensor_parallel_size=int(training.get("vllm_tensor_parallel_size", 1)),
        logging_steps=int(training.get("logging_steps", 1)),
        save_steps=int(training.get("save_steps", 25)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(training.get("lr_scheduler_type", "cosine")),
        seed=int(training.get("seed", 17)),
        data_seed=int(training.get("data_seed", 17)),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        tf32=bool(training.get("tf32", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=list(training.get("report_to") or ()),
        log_completions=bool(training.get("log_completions", False)),
        ddp_find_unused_parameters=False if repair else None,
    )
    trainer_class = GRPOTrainer
    extra = {}
    if repair:
        from python_repair_grpo import PythonRepairGRPOTrainer
        trainer_class = PythonRepairGRPOTrainer
        extra["repair_config"] = repair
    trainer = trainer_class(
        model=model,
        reward_funcs=template_rlvr_rewards,
        args=grpo_args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=peft_config,
        **extra,
    )
    train_result = trainer.train(
        resume_from_checkpoint=bool(cfg.get("resume_from_checkpoint", True))
        and any(Path(grpo_args.output_dir).glob("checkpoint-*/trainer_state.json"))
    )
    output = Path(grpo_args.output_dir)
    final_adapter = output / "adapter"
    trainer.save_model(str(final_adapter))
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        return 0
    tokenizer.save_pretrained(output / "tokenizer")
    manifest = {
        **preflight,
        "artifact_type": "python_template_slots_rlvr_adapter_v1",
        "adapter_path": str(final_adapter),
        "adapter_sha256": path_sha256(final_adapter),
        "repair": repair,
        "reference_policy": "frozen_initial_sft_adapter" if repair else "legacy_peft_disabled_base",
        "training": {
            "max_steps": grpo_args.max_steps,
            "num_generations": grpo_args.num_generations,
            "max_completion_length": grpo_args.max_completion_length,
            "loss_type": grpo_args.loss_type,
            "beta": grpo_args.beta,
            "seed": grpo_args.seed,
            "train_metrics": dict(train_result.metrics),
        },
    }
    (output / "rlvr_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
