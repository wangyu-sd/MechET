#!/usr/bin/env python3
"""Train matched supervised conditions from replay-verified conversations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("install PyYAML") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def tool_names(messages: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    for message in messages:
        if message.get("role") == "tool" and message.get("name"):
            output.append(str(message["name"]))
        for call in message.get("tool_calls") or []:
            name = str((call.get("function") or {}).get("name") or "")
            if name:
                output.append(name)
    return output


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    require_trace_owned: bool,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Tool-SFT dataset is empty")
    ids = set()
    tool_counts: dict[str, int] = {}
    trace_bound = 0
    finish_trace_rows = 0
    assistant_messages = 0
    tool_result_messages = 0
    for row in rows:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in ids:
            raise ValueError(f"invalid or duplicate row ID: {identifier}")
        ids.add(identifier)
        messages = list(row.get("messages") or [])
        if not messages or not any(message.get("role") == "assistant" for message in messages):
            raise ValueError(f"row has no assistant messages: {identifier}")
        assistant_messages += sum(message.get("role") == "assistant" for message in messages)
        tool_result_messages += sum(message.get("role") == "tool" for message in messages)
        names = tool_names(messages)
        for name in names:
            tool_counts[name] = tool_counts.get(name, 0) + 1
        has_finish = "finish_trace" in names
        finish_trace_rows += int(has_finish)
        metadata = dict(row.get("metadata") or {})
        is_trace_bound = metadata.get("endpoint_source") == "environment_owned_trace"
        trace_bound += int(is_trace_bound)
        if require_trace_owned:
            if not has_finish:
                raise ValueError(f"trace-owned row lacks finish_trace: {identifier}")
            if not is_trace_bound:
                raise ValueError(
                    f"trace-owned row lacks endpoint_source=environment_owned_trace: {identifier}"
                )
            if metadata.get("executor_replayed") is not True:
                raise ValueError(f"trace-owned row was not executor replayed: {identifier}")
            if not metadata.get("trace_digest"):
                raise ValueError(f"trace-owned row lacks trace_digest: {identifier}")
    denominator = len(rows)
    return {
        "n_rows": denominator,
        "n_unique_ids": len(ids),
        "tool_calls_and_results": tool_counts,
        "assistant_messages": assistant_messages,
        "tool_result_messages": tool_result_messages,
        "trace_bound_rows": trace_bound,
        "trace_bound_rate": trace_bound / denominator,
        "finish_trace_rows": finish_trace_rows,
        "finish_trace_rate": finish_trace_rows / denominator,
        "require_trace_owned": require_trace_owned,
    }


def conversational_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep messages structured so TRL can construct assistant-token masks."""

    return [
        {
            "id": str(row["id"]),
            "messages": row["messages"],
        }
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="override training.max_steps; useful for a fixed tiny overfit smoke test",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_file = Path(str(cfg.get("train_file") or ""))
    if not train_file.exists():
        raise FileNotFoundError(f"train_file does not exist: {train_file}")
    rows = read_rows(
        train_file,
        limit=args.limit or int(cfg.get("limit_examples", 0) or 0),
    )
    contract = dict(cfg.get("contract") or {})
    require_trace_owned = bool(contract.get("require_trace_owned", True))
    training = dict(cfg.get("training") or {})
    configured_max_steps = int(training.get("max_steps", -1))
    max_steps = int(args.max_steps or configured_max_steps)
    report = {
        **validate_rows(rows, require_trace_owned=require_trace_owned),
        "scientific_hypothesis": cfg.get("scientific_hypothesis"),
        "train_file": str(train_file),
        "train_file_sha256": file_sha256(train_file),
        "model_name_or_path": cfg.get("model_name_or_path"),
        "condition_name": cfg.get("condition_name"),
        "output_dir": cfg.get("output_dir"),
        "dataset_format": "conversational_messages",
        "assistant_only_loss": bool(training.get("assistant_only_loss", True)),
        "max_steps": max_steps,
        "stable_id_manifest": contract.get("stable_id_manifest"),
        "terminal_tool": contract.get("terminal_tool"),
        "free_form_proof_submission": contract.get("free_form_proof_submission"),
        "real_overfit_smoke_test_required": bool(
            contract.get("real_overfit_smoke_test_required", False)
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
    lora = dict(cfg.get("lora") or {})
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer does not expose a conversational chat template")

    sft_args = SFTConfig(
        output_dir=str(cfg.get("output_dir") or "outputs/agent/tool_sft"),
        learning_rate=float(training.get("learning_rate", 2e-5)),
        num_train_epochs=float(training.get("num_train_epochs", 1.0)),
        max_steps=max_steps,
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
        train_dataset=Dataset.from_list(conversational_records(rows)),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    output = Path(sft_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_contract.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
