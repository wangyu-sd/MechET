#!/usr/bin/env python3
"""Train matched supervised conditions from executable conversations."""
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


def selected_directory_sha256(path: Path) -> str:
    """Hash saved adapter/model files without self-referential manifests."""

    digest = hashlib.sha256()
    excluded = {"adapter_manifest.json", "data_contract.json"}
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.name not in excluded
    )
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
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


def schema_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str((item.get("function") or {}).get("name") or "")
        for item in tools
        if str((item.get("function") or {}).get("name") or "")
    }


def validate_conversation(
    row: dict[str, Any], *, require_trace_owned: bool
) -> dict[str, int]:
    identifier = str(row.get("id") or "")
    messages = list(row.get("messages") or [])
    tools = list(row.get("tools") or [])
    allowed = schema_tool_names(tools)
    pending: dict[str, str] = {}
    calls = results = 0
    finish_trace = 0
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        for call in message.get("tool_calls") or []:
            if role != "assistant":
                raise ValueError(
                    f"tool call outside assistant message in {identifier}:{index}"
                )
            call_id = str(call.get("id") or "")
            function = dict(call.get("function") or {})
            name = str(function.get("name") or "")
            arguments = function.get("arguments")
            if not call_id or call_id in pending:
                raise ValueError(f"invalid/duplicate tool call id in {identifier}")
            if name not in allowed:
                raise ValueError(
                    f"tool {name!r} is absent from tools schema in {identifier}"
                )
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"tool arguments must be a JSON object in {identifier}:{name}"
                )
            pending[call_id] = name
            calls += 1
            finish_trace += int(name == "finish_trace")
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or "")
            expected = pending.pop(call_id, None)
            if expected is None:
                raise ValueError(
                    f"orphan tool result {call_id!r} in {identifier}:{index}"
                )
            if name != expected:
                raise ValueError(
                    f"tool result name mismatch in {identifier}: {name} != {expected}"
                )
            try:
                json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"tool result is not JSON in {identifier}:{name}"
                ) from exc
            results += 1
    if pending:
        raise ValueError(f"tool calls without results in {identifier}: {pending}")
    metadata = dict(row.get("metadata") or {})
    if require_trace_owned:
        if finish_trace != 1:
            raise ValueError(
                f"trace-owned row requires exactly one finish_trace: {identifier}"
            )
        if metadata.get("endpoint_source") != "environment_owned_trace":
            raise ValueError(
                f"trace-owned row lacks environment-owned endpoint: {identifier}"
            )
        if metadata.get("executor_replayed") is not True:
            raise ValueError(f"trace-owned row was not executor replayed: {identifier}")
        if not metadata.get("trace_digest"):
            raise ValueError(f"trace-owned row lacks trace_digest: {identifier}")
    elif calls or results:
        raise ValueError(f"direct condition contains tools: {identifier}")
    return {"tool_calls": calls, "tool_results": results, "finish_trace": finish_trace}


def validate_rows(
    rows: list[dict[str, Any]], *, require_trace_owned: bool
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Tool-SFT dataset is empty")
    ids: set[str] = set()
    tool_calls = tool_results = finish_rows = trace_bound = 0
    assistant_messages = 0
    for row in rows:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in ids:
            raise ValueError(f"invalid or duplicate row ID: {identifier}")
        ids.add(identifier)
        messages = list(row.get("messages") or [])
        if not messages or not any(
            message.get("role") == "assistant" for message in messages
        ):
            raise ValueError(f"row has no assistant messages: {identifier}")
        assistant_messages += sum(
            message.get("role") == "assistant" for message in messages
        )
        counts = validate_conversation(
            row, require_trace_owned=require_trace_owned
        )
        tool_calls += counts["tool_calls"]
        tool_results += counts["tool_results"]
        finish_rows += int(counts["finish_trace"] == 1)
        trace_bound += int(
            (row.get("metadata") or {}).get("endpoint_source")
            == "environment_owned_trace"
        )
    denominator = len(rows)
    return {
        "n_rows": denominator,
        "n_unique_ids": len(ids),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "assistant_messages": assistant_messages,
        "trace_bound_rows": trace_bound,
        "trace_bound_rate": trace_bound / denominator,
        "finish_trace_rows": finish_rows,
        "finish_trace_rate": finish_rows / denominator,
        "require_trace_owned": require_trace_owned,
        "conversation_schema_valid": True,
    }


def conversational_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve tools and messages for TRL's tool-calling chat template."""

    return [
        {
            "id": str(row["id"]),
            "messages": row["messages"],
            "tools": row.get("tools") or [],
        }
        for row in rows
    ]


def _assistant_mask(payload: dict[str, Any]) -> list[int]:
    for key, value in payload.items():
        lowered = str(key).lower()
        if "assistant" in lowered and "mask" in lowered:
            return [int(item) for item in value]
    return []


def tokenizer_audit(
    rows: list[dict[str, Any]], tokenizer, *, max_length: int
) -> dict[str, Any]:
    """Render every row using the real tokenizer and verify assistant masks."""

    lengths: list[int] = []
    supervised: list[int] = []
    zero_mask_ids: list[str] = []
    truncated = 0
    for row in rows:
        kwargs = {
            "conversation": row["messages"],
            "tools": row.get("tools") or None,
            "tokenize": True,
            "return_dict": True,
            "return_assistant_tokens_mask": True,
            "add_generation_prompt": False,
        }
        rendered = tokenizer.apply_chat_template(**kwargs)
        if not isinstance(rendered, dict):
            raise ValueError("chat template did not return a token dictionary")
        input_ids = list(rendered.get("input_ids") or [])
        mask = _assistant_mask(rendered)
        if not mask or len(mask) != len(input_ids):
            raise ValueError(
                "chat template does not expose a valid assistant-token mask"
            )
        n_supervised = sum(mask)
        if n_supervised <= 0:
            zero_mask_ids.append(str(row.get("id") or ""))
        lengths.append(len(input_ids))
        supervised.append(n_supervised)
        truncated += int(len(input_ids) > max_length)
    if zero_mask_ids:
        raise ValueError(f"rows with zero assistant mask: {zero_mask_ids[:10]}")
    return {
        "tokenizer_revision": getattr(tokenizer, "name_or_path", "unknown"),
        "n_tokenizer_audited_rows": len(rows),
        "total_input_tokens": sum(lengths),
        "total_supervised_tokens": sum(supervised),
        "max_input_tokens": max(lengths, default=0),
        "min_supervised_tokens": min(supervised, default=0),
        "truncation_count": truncated,
        "truncation_rate": truncated / max(len(rows), 1),
        "assistant_mask_valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="override training.max_steps for a fixed overfit smoke test",
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
        "artifact_type": "tool_sft_data_contract",
        "scientific_hypothesis": cfg.get("scientific_hypothesis"),
        "train_file": str(train_file),
        "train_file_sha256": file_sha256(train_file),
        "model_name_or_path": cfg.get("model_name_or_path"),
        "condition_name": cfg.get("condition_name"),
        "output_dir": cfg.get("output_dir"),
        "dataset_format": "conversational_messages_with_tools",
        "assistant_only_loss": bool(training.get("assistant_only_loss", True)),
        "max_steps": max_steps,
        "stable_id_manifest": contract.get("stable_id_manifest"),
        "validation_report": contract.get("validation_report"),
        "environment_revision": contract.get("environment_revision"),
        "executor_revision": contract.get("executor_revision"),
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
        revision=training.get("model_revision"),
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer does not expose a conversational chat template")
    report.update(
        tokenizer_audit(
            rows,
            tokenizer,
            max_length=int(training.get("max_length", 4096)),
        )
    )
    if report["truncation_count"]:
        raise ValueError(
            "Tool-SFT rows exceed max_length; increase the frozen budget or rebuild data"
        )

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
    records = conversational_records(rows)
    try:
        dataset = Dataset.from_list(records, on_mixed_types="use_json")
    except TypeError:
        dataset = Dataset.from_list(records)
    trainer = SFTTrainer(
        model=model_name,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    output = Path(sft_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_contract.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    adapter_hash = selected_directory_sha256(output)
    adapter_manifest = {
        "artifact_type": "trainable_peft_adapter",
        "adapter_path": str(output),
        "adapter_sha256": adapter_hash,
        "base_model": model_name,
        "condition_name": cfg.get("condition_name"),
        "scientific_hypothesis": cfg.get("scientific_hypothesis"),
        "data_contract": str(output / "data_contract.json"),
        "train_file_sha256": report["train_file_sha256"],
        "environment_revision": report.get("environment_revision"),
        "executor_revision": report.get("executor_revision"),
    }
    (output / "adapter_manifest.json").write_text(
        json.dumps(adapter_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
