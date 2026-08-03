#!/usr/bin/env python3
"""Train matched tool-using SFT conditions from replay-verified trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("install PyYAML") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def read_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Tool-SFT dataset is empty")
    ids = set()
    tool_counts: dict[str, int] = {}
    trace_bound = 0
    for row in rows:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in ids:
            raise ValueError(f"invalid or duplicate row ID: {identifier}")
        ids.add(identifier)
        messages = row.get("messages") or []
        if not messages or not any(message.get("role") == "assistant" for message in messages):
            raise ValueError(f"row has no assistant messages: {identifier}")
        for message in messages:
            for call in message.get("tool_calls") or []:
                name = str((call.get("function") or {}).get("name") or "")
                tool_counts[name] = tool_counts.get(name, 0) + 1
        if (row.get("metadata") or {}).get("endpoint_source") == "environment_owned_trace":
            trace_bound += 1
    return {
        "n_rows": len(rows),
        "n_unique_ids": len(ids),
        "tool_calls": tool_counts,
        "trace_bound_rows": trace_bound,
        "trace_bound_rate": trace_bound / len(rows),
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
    rows = read_rows(
        train_file,
        limit=args.limit or int(cfg.get("limit_examples", 0) or 0),
    )
    report = {
        **validate_rows(rows),
        "train_file": str(train_file),
        "model_name_or_path": cfg.get("model_name_or_path"),
        "condition_name": cfg.get("condition_name"),
        "output_dir": cfg.get("output_dir"),
        "assistant_only_loss": bool(
            (cfg.get("training") or {}).get("assistant_only_loss", True)
        ),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Tool-SFT training requires mechet[agent]") from exc

    model_name = str(cfg.get("model_name_or_path") or "")
    if not model_name:
        raise ValueError("model_name_or_path is required")
    training = dict(cfg.get("training") or {})
    lora = dict(cfg.get("lora") or {})
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )

    rendered = []
    for row in rows:
        try:
            text = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            raise ValueError(
                f"chat template failed for {row.get('id')}; use a tool-capable tokenizer"
            ) from exc
        rendered.append({"text": text, "id": row["id"]})

    sft_args = SFTConfig(
        output_dir=str(cfg.get("output_dir") or "outputs/agent/tool_sft"),
        learning_rate=float(training.get("learning_rate", 2e-5)),
        num_train_epochs=float(training.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        max_length=int(training.get("max_length", 4096)),
        packing=bool(training.get("packing", False)),
        assistant_only_loss=bool(training.get("assistant_only_loss", True)),
        logging_steps=int(training.get("logging_steps", 1)),
        save_steps=int(training.get("save_steps", 100)),
        report_to=list(training.get("report_to") or []),
    )
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(
            lora.get("target_modules")
            or ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model_name,
        args=sft_args,
        train_dataset=Dataset.from_list(rendered),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    Path(sft_args.output_dir, "data_contract.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
