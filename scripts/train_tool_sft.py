#!/usr/bin/env python3
"""Train matched supervised conditions from executable conversations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("install PyYAML") from exc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from mechet.collator import AssistantOnlyCollator


def load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_directory_sha256(path: Path) -> str:
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
    per_row_calls: list[int] = []
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
        per_row_calls.append(counts["tool_calls"])
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
        "max_tool_calls_per_row": max(per_row_calls, default=0),
        "mean_tool_calls_per_row": tool_calls / denominator,
        "assistant_messages": assistant_messages,
        "trace_bound_rows": trace_bound,
        "trace_bound_rate": trace_bound / denominator,
        "finish_trace_rows": finish_rows,
        "finish_trace_rate": finish_rows / denominator,
        "require_trace_owned": require_trace_owned,
        "conversation_schema_valid": True,
    }


def conversational_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row["id"]),
            "messages": row["messages"],
            "tools": row.get("tools") or [],
        }
        for row in rows
    ]


def _flatten_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("tokenizer did not return a list of token ids")
    while len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if any(isinstance(item, list) for item in value):
        raise ValueError("tokenizer returned nested token ids")
    return [int(item) for item in value]


def _render_chat_text(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    attempts = [
        {"enable_thinking": False, **kwargs},
        dict(kwargs),
    ]
    if tools:
        without_tools = dict(kwargs)
        without_tools.pop("tools", None)
        attempts.extend(
            [
                {"enable_thinking": False, **without_tools},
                without_tools,
            ]
        )
    last_error: TypeError | None = None
    for attempt in attempts:
        try:
            return tokenizer.apply_chat_template(messages, **attempt)
        except TypeError as exc:
            last_error = exc
        try:
            return tokenizer.apply_chat_template(conversation=messages, **attempt)
        except TypeError as exc:
            last_error = exc
    raise last_error or TypeError("chat template rendering failed")


def _tokenize_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return _flatten_token_ids(encoded["input_ids"])


def _encode_row(
    tokenizer,
    row: dict[str, Any],
    *,
    max_length: int,
) -> tuple[dict[str, list[int]], int]:
    messages = list(row.get("messages") or [])
    if not messages:
        raise ValueError("row has no messages")
    tools = list(row.get("tools") or [])
    rendered = _render_chat_text(
        tokenizer,
        messages,
        tools=tools or None,
        add_generation_prompt=False,
    )
    input_ids = _tokenize_text(tokenizer, rendered)
    full_length = len(input_ids)
    mask = [0] * full_length
    assistant_found = False
    for index, message in enumerate(messages):
        if str(message.get("role") or "") != "assistant":
            continue
        assistant_found = True
        prefix_text = _render_chat_text(
            tokenizer,
            messages[:index],
            tools=tools or None,
            add_generation_prompt=True,
        )
        assistant_text = _render_chat_text(
            tokenizer,
            messages[: index + 1],
            tools=tools or None,
            add_generation_prompt=False,
        )
        start = len(_tokenize_text(tokenizer, prefix_text))
        end = len(_tokenize_text(tokenizer, assistant_text))
        start = max(0, min(start, full_length))
        end = max(start, min(end, full_length))
        for token_index in range(start, end):
            mask[token_index] = 1
    if not assistant_found:
        raise ValueError("row has no assistant messages")
    if full_length > max_length:
        input_ids = input_ids[-max_length:]
        mask = mask[-max_length:]
    labels = [
        token_id if supervised else -100
        for token_id, supervised in zip(input_ids, mask)
    ]
    if not any(value != -100 for value in labels):
        raise ValueError("truncation removed all assistant labels")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }, full_length


def tokenize_rows(
    rows: list[dict[str, Any]], tokenizer, *, max_length: int
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    encoded_rows: list[dict[str, list[int]]] = []
    lengths: list[int] = []
    supervised: list[int] = []
    zero_mask_ids: list[str] = []
    truncated = 0
    for row in rows:
        encoded, raw_length = _encode_row(tokenizer, row, max_length=max_length)
        input_ids = list(encoded.get("input_ids") or [])
        mask = [int(value != -100) for value in list(encoded.get("labels") or [])]
        n_supervised = sum(mask)
        if n_supervised <= 0:
            zero_mask_ids.append(str(row.get("id") or ""))
        lengths.append(len(input_ids))
        supervised.append(n_supervised)
        truncated += int(raw_length > max_length)
        encoded_rows.append(encoded)
    if zero_mask_ids:
        raise ValueError(f"rows with zero assistant mask: {zero_mask_ids[:10]}")
    audit = {
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
        "tokenizer_revision": str(
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            or getattr(tokenizer, "init_kwargs", {}).get("revision")
            or "unresolved"
        ),
        "n_tokenizer_audited_rows": len(encoded_rows),
        "total_input_tokens": sum(lengths),
        "total_supervised_tokens": sum(supervised),
        "max_input_tokens": max(lengths, default=0),
        "min_supervised_tokens": min(supervised, default=0),
        "truncation_count": truncated,
        "truncation_rate": truncated / max(len(rows), 1),
        "assistant_mask_valid": True,
    }
    return encoded_rows, audit


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
    seed = int(training.get("seed", 17))
    data_seed = int(training.get("data_seed", seed))
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
        "seed": seed,
        "data_seed": data_seed,
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
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from peft import get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Tool-SFT training requires mechet[agent]") from exc

    model_name = str(cfg.get("model_name_or_path") or "")
    if not model_name:
        raise ValueError("model_name_or_path is required")
    requested_revision = str(training.get("model_revision") or "").strip() or None
    trust_remote_code = bool(training.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=requested_revision,
        trust_remote_code=trust_remote_code,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer does not expose a conversational chat template")
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_rows, audit = tokenize_rows(
        rows,
        tokenizer,
        max_length=int(training.get("max_length", 4096)),
    )
    report.update(audit)
    if report["truncation_count"]:
        raise ValueError(
            "Tool-SFT rows exceed max_length; increase the frozen budget or rebuild data"
        )
    resolved_revision = str(
        requested_revision or report.get("tokenizer_revision") or "unresolved"
    )
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
    gradient_checkpointing = bool(training.get("gradient_checkpointing", True))
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=None if resolved_revision == "unresolved" else resolved_revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    if gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    lora = dict(cfg.get("lora") or {})
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
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(cfg.get("output_dir") or "outputs/agent/tool_sft"),
        learning_rate=float(training.get("learning_rate", 2e-5)),
        num_train_epochs=float(training.get("num_train_epochs", 1.0)),
        max_steps=max_steps,
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        logging_steps=int(training.get("logging_steps", 1)),
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        seed=seed,
        data_seed=data_seed,
        bf16=bf16,
        fp16=fp16,
        tf32=tf32,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=int(training.get("dataloader_num_workers", 2)),
        optim=str(training.get("optim", "adamw_torch_fused")),
        report_to=list(training.get("report_to") or []),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    try:
        dataset = Dataset.from_list(encoded_rows, on_mixed_types="use_json")
    except TypeError:
        dataset = Dataset.from_list(encoded_rows)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model()
    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report["base_model_revision"] = resolved_revision
    report["training_runtime"] = {
        "seed": seed,
        "data_seed": data_seed,
        "bf16": bf16,
        "fp16": fp16,
        "tf32": tf32,
        "gradient_checkpointing": gradient_checkpointing,
        "group_by_length": bool(training.get("group_by_length", True)),
    }
    (output / "data_contract.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    adapter_hash = selected_directory_sha256(output)
    adapter_manifest = {
        "artifact_type": "trainable_peft_adapter",
        "adapter_path": str(output),
        "adapter_sha256": adapter_hash,
        "base_model": model_name,
        "base_model_revision": resolved_revision,
        "tokenizer_revision": report.get("tokenizer_revision"),
        "condition_name": cfg.get("condition_name"),
        "scientific_hypothesis": cfg.get("scientific_hypothesis"),
        "data_contract": str(output / "data_contract.json"),
        "train_file_sha256": report["train_file_sha256"],
        "environment_revision": report.get("environment_revision"),
        "executor_revision": report.get("executor_revision"),
        "seed": seed,
        "data_seed": data_seed,
    }
    (output / "adapter_manifest.json").write_text(
        json.dumps(adapter_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
