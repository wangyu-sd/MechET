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

from mechet.assistant_masking import (
    encode_assistant_only_conversation,
    percentile_nearest_rank,
)
from mechet.collator import AssistantOnlyCollator
from mechet.model_revision import (
    is_immutable_revision,
    resolve_loaded_model_revision,
)


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


def tokenize_rows(
    rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    """Render each conversation once and supervise only final ChatML assistant spans."""

    encoded_rows: list[dict[str, list[int]]] = []
    lengths: list[int] = []
    supervised: list[int] = []
    assistant_turns: list[int] = []
    over_budget_ids: list[str] = []
    mask_methods: set[str] = set()
    for row in rows:
        encoded, audit = encode_assistant_only_conversation(
            tokenizer,
            row,
            max_length=max_length,
        )
        identifier = str(row.get("id") or "")
        lengths.append(int(audit["raw_length"]))
        supervised.append(int(audit["supervised_tokens"]))
        assistant_turns.append(int(audit["assistant_turns"]))
        mask_methods.add(str(audit["mask_method"]))
        if bool(audit["exceeds_max_length"]):
            over_budget_ids.append(identifier)
        encoded_rows.append(encoded)
    if any(value <= 0 for value in supervised):
        raise ValueError("one or more rows have zero assistant supervision")
    audit = {
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
        "n_tokenizer_audited_rows": len(encoded_rows),
        "total_input_tokens": sum(lengths),
        "total_supervised_tokens": sum(supervised),
        "max_input_tokens": max(lengths, default=0),
        "p50_input_tokens": percentile_nearest_rank(lengths, 0.50),
        "p95_input_tokens": percentile_nearest_rank(lengths, 0.95),
        "p99_input_tokens": percentile_nearest_rank(lengths, 0.99),
        "min_supervised_tokens": min(supervised, default=0),
        "max_assistant_turns": max(assistant_turns, default=0),
        "configured_max_length": int(max_length),
        "truncation_count": len(over_budget_ids),
        "truncation_rate": len(over_budget_ids) / max(len(rows), 1),
        "over_budget_ids": over_budget_ids[:50],
        "assistant_mask_valid": True,
        "assistant_mask_methods": sorted(mask_methods),
        "zero_truncation_required": True,
    }
    return encoded_rows, audit


def _training_argument_kwargs(
    TrainingArguments: Any,
    *,
    cfg: dict[str, Any],
    training: dict[str, Any],
    max_steps: int,
    seed: int,
    data_seed: int,
    bf16: bool,
    fp16: bool,
    tf32: bool,
    gradient_checkpointing: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "output_dir": str(cfg.get("output_dir") or "outputs/agent/tool_sft"),
        "learning_rate": float(training.get("learning_rate", 2e-5)),
        "num_train_epochs": float(training.get("num_train_epochs", 1.0)),
        "max_steps": max_steps,
        "per_device_train_batch_size": int(
            training.get("per_device_train_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 8)
        ),
        "logging_steps": int(training.get("logging_steps", 1)),
        "save_steps": int(training.get("save_steps", 100)),
        "save_total_limit": int(training.get("save_total_limit", 2)),
        "seed": seed,
        "data_seed": data_seed,
        "bf16": bf16,
        "fp16": fp16,
        "tf32": tf32,
        "gradient_checkpointing": gradient_checkpointing,
        "dataloader_num_workers": int(training.get("dataloader_num_workers", 2)),
        "optim": str(training.get("optim", "adamw_torch_fused")),
        "report_to": list(training.get("report_to") or []),
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    }
    fields = dict(getattr(TrainingArguments, "__dataclass_fields__", {}) or {})
    requested_grouping = bool(training.get("group_by_length", True))
    grouping = {
        "requested": requested_grouping,
        "api_field": None,
        "applied_value": None,
    }
    if "train_sampling_strategy" in fields:
        value = "group_by_length" if requested_grouping else "random"
        kwargs["train_sampling_strategy"] = value
        grouping.update({"api_field": "train_sampling_strategy", "applied_value": value})
    elif "group_by_length" in fields:
        kwargs["group_by_length"] = requested_grouping
        grouping.update(
            {"api_field": "group_by_length", "applied_value": requested_grouping}
        )
    return kwargs, grouping


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
    if bool(training.get("packing", False)):
        raise ValueError(
            "packing=true is unsupported by the pretokenized assistant-only Trainer path"
        )
    if not bool(training.get("assistant_only_loss", True)):
        raise ValueError(
            "train_tool_sft.py requires assistant_only_loss=true; labels are explicitly masked"
        )
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
        "dataset_format": "pretokenized_chatml_with_explicit_assistant_labels",
        "assistant_only_loss": True,
        "packing": False,
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
        from peft import LoraConfig, get_peft_model
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

    revision_info = resolve_loaded_model_revision(
        model_name_or_path=model_name,
        requested_revision=requested_revision,
        tokenizer=tokenizer,
    )
    resolved_revision = str(revision_info.get("resolved_model_revision") or "")
    report.update(revision_info)
    report["model_revision_frozen"] = is_immutable_revision(resolved_revision)
    if not report["model_revision_frozen"] and not Path(model_name).exists():
        raise ValueError("remote Tool-SFT training requires an immutable model revision")

    encoded_rows, audit = tokenize_rows(
        rows,
        tokenizer,
        max_length=int(training.get("max_length", 12288)),
    )
    report.update(audit)
    if report["truncation_count"]:
        raise ValueError(
            "Tool-SFT rows exceed max_length; increase the same frozen max_length "
            "for every matched condition or rebuild the data. "
            f"max={report['max_input_tokens']} budget={report['configured_max_length']} "
            f"examples={report['over_budget_ids'][:10]}"
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
        revision=resolved_revision or requested_revision,
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
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_kwargs, grouping = _training_argument_kwargs(
        TrainingArguments,
        cfg=cfg,
        training=training,
        max_steps=max_steps,
        seed=seed,
        data_seed=data_seed,
        bf16=bf16,
        fp16=fp16,
        tf32=tf32,
        gradient_checkpointing=gradient_checkpointing,
    )
    training_args = TrainingArguments(**training_kwargs)
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
    report["base_model_revision"] = resolved_revision or None
    report["training_runtime"] = {
        "seed": seed,
        "data_seed": data_seed,
        "bf16": bf16,
        "fp16": fp16,
        "tf32": tf32,
        "gradient_checkpointing": gradient_checkpointing,
        "length_grouping": grouping,
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
        "requested_model_revision": revision_info.get("requested_model_revision"),
        "base_model_revision": resolved_revision or None,
        "tokenizer_revision": revision_info.get("tokenizer_revision"),
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
