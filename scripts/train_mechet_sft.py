#!/usr/bin/env python3
"""MechET LoRA/QLoRA SFT trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from mechet.collator import AssistantOnlyCollator, encode_assistant_only
from mechet.model import resolve_qwen_model_path
from mechet.tokenizer import save_tokenizer_audit


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _pick_dtype(torch, *, force_fp16: bool = False):
    if force_fp16:
        return torch.float16
    if not torch.cuda.is_available():
        return torch.float32
    name = torch.cuda.get_device_name(0).lower()
    if any(tag in name for tag in ("h20", "a100", "h100", "h800")):
        return torch.bfloat16
    return torch.float16


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--validation-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--use-qlora", action="store_true", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-examples", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config) if args.config else {}

    def _cfg(key, default=None):
        cli = getattr(args, key.replace("-", "_"), None)
        if cli is not None and cli is not default:
            if isinstance(cli, bool) or cli != parser.get_default(key.replace("-", "_")):
                return cli
        return cfg.get(key, default)

    train_file = Path(_cfg("train_file", REPO / "data/orbit_training/orbit_sft/train.jsonl"))
    if not train_file.is_absolute():
        train_file = REPO / train_file
    validation_file_cfg = _cfg("validation_file", None)
    validation_file = args.validation_file or (Path(validation_file_cfg) if validation_file_cfg else None)
    if validation_file is not None and not validation_file.is_absolute():
        validation_file = REPO / validation_file
    output_dir = Path(_cfg("output_dir", REPO / "outputs/mechet/sft_dry_run"))
    if not output_dir.is_absolute():
        output_dir = REPO / output_dir
    max_steps = int(_cfg("max_steps", 20))
    save_steps = int(_cfg("save_steps", 50))
    max_seq_length = int(_cfg("max_seq_length", 1024))
    grad_accum = int(_cfg("gradient_accumulation_steps", 8))
    use_qlora = bool(_cfg("use_qlora", False))
    force_fp16 = bool(_cfg("fp16", False))
    force_bf16 = bool(_cfg("bf16", False))
    attn_impl = _cfg("attention_implementation", None)
    target_modules = _cfg("target_modules", ["q_proj", "v_proj"])
    seed = int(_cfg("seed", 42) if args.seed is None else args.seed)

    model_path = args.model_path or resolve_qwen_model_path() or os.environ.get("QWEN_MODEL_PATH")
    report = {
        "status": "not_executed",
        "model_path": model_path,
        "output_dir": str(output_dir),
        "max_steps": max_steps,
        "config": str(args.config) if args.config else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_tokenizer_audit(REPO / "outputs/mechet/tokenizer_audit.json", model_path)

    if args.require_cuda or cfg.get("require_cuda"):
        import torch

        if not torch.cuda.is_available():
            report["reason"] = "CUDA required but unavailable"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dry_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 1

    if not model_path or not Path(model_path).exists():
        report["reason"] = "No local Qwen checkpoint (set QWEN_MODEL_PATH)"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dry_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        report["reason"] = f"Missing deps: {exc}"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dry_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    rows = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit_examples is not None:
        rows = rows[: max(0, args.limit_examples)]
    if not rows:
        report["reason"] = "empty train file"
        (output_dir / "dry_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_rows = []
    label_token_counts = []
    for row_index, row in enumerate(rows):
        try:
            encoded = encode_assistant_only(tokenizer, row["messages"], max_length=max_seq_length)
        except Exception as exc:
            raise ValueError(f"{train_file}:{row_index + 1}: failed to encode assistant-only labels: {exc}") from exc
        label_token_counts.append(sum(1 for value in encoded["labels"] if value != -100))
        encoded_rows.append(encoded)
    report["avg_label_tokens"] = sum(label_token_counts) / max(len(label_token_counts), 1)
    report["min_label_tokens"] = min(label_token_counts) if label_token_counts else 0
    report["train_examples"] = len(rows)
    report["dataset_hash"] = _file_hash(train_file)

    dtype = _pick_dtype(torch, force_fp16=force_fp16 or force_bf16 is False)
    if force_bf16:
        dtype = torch.bfloat16
    if force_fp16:
        dtype = torch.float16

    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "quantization_config": quant_config,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": False,
        "device_map": "auto" if quant_config is not None else None,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    if args.dry_run:
        from torch.utils.data import DataLoader

        ds = Dataset.from_list(encoded_rows[: min(4, len(encoded_rows))])
        batch = next(iter(DataLoader(ds, batch_size=min(2, len(ds)), collate_fn=AssistantOnlyCollator(tokenizer))))
        report.update(
            {
                "status": "dry_run_completed",
                "input_shape": list(batch["input_ids"].shape),
                "labels_shape": list(batch["labels"].shape),
                "non_masked_label_tokens": int((batch["labels"] != -100).sum().item()),
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dry_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if torch.cuda.is_available() and quant_config is None:
        model = model.to("cuda")

    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=int(_cfg("lora_r", 8)),
        lora_alpha=int(_cfg("lora_alpha", 16)),
        lora_dropout=float(_cfg("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()

    ds = Dataset.from_list(encoded_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": output_dir.name,
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "dataset_manifest_hash": report["dataset_hash"],
        "model_path": model_path,
        "hardware": {
            "platform": platform.platform(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        },
        "config_path": str(args.config) if args.config else None,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.config and args.config.exists():
        (output_dir / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=max_steps,
        per_device_train_batch_size=int(_cfg("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(_cfg("learning_rate", 2e-4)),
        logging_steps=5,
        save_steps=save_steps,
        report_to=[],
        fp16=dtype == torch.float16 and torch.cuda.is_available(),
        bf16=dtype == torch.bfloat16 and torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=seed,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    resume = args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=resume)
    report["train_loss"] = train_result.training_loss
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    report["status"] = "completed"
    report["dtype"] = str(dtype)
    report["use_qlora"] = use_qlora
    (output_dir / "training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "dry_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
