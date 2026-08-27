#!/usr/bin/env python3
"""Train matched supervised conditions from executable conversations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
sys.path.insert(0, str(REPO / "scripts"))

from agent_model_init import validate_lineage
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


def enforce_artifact_status(
    train_file: Path, contract: dict[str, Any]
) -> dict[str, Any] | None:
    """Fail closed when a dataset sidecar marks an artifact unsafe for training."""

    status_path = train_file.parent / "ARTIFACT_STATUS.json"
    if not status_path.is_file():
        return None
    status = dict(json.loads(status_path.read_text(encoding="utf-8")))
    if bool(status.get("training_allowed")):
        return status
    override = bool(contract.get("allow_deprecated_artifact", False)) or os.environ.get(
        "MECHET_ALLOW_DEPRECATED_ARTIFACT", ""
    ).strip().lower() in {"1", "true", "yes"}
    if not override:
        artifact_id = str(status.get("artifact_id") or train_file.parent)
        reason = str(status.get("reason") or "artifact is not approved for training")
        raise ValueError(
            f"ARTIFACT_TRAINING_FORBIDDEN:{artifact_id}:{reason}; "
            "inspect ARTIFACT_STATUS.json and the dataset registry"
        )
    return status


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


def validate_initial_adapter_config(
    cfg: dict[str, Any], lineage: dict[str, Any]
) -> dict[str, Any] | None:
    """Fail closed when a warm-start adapter is absent or LoRA-incompatible."""

    adapter_value = str(cfg.get("initial_adapter_path") or "").strip()
    if not adapter_value:
        return None
    adapter_path = Path(adapter_value)
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"initial adapter config does not exist: {adapter_config_path}"
        )
    adapter_cfg = dict(
        json.loads(adapter_config_path.read_text(encoding="utf-8"))
    )
    requested = dict(cfg.get("lora") or {})
    expected_targets = set(
        requested.get("target_modules")
        or ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    observed_targets = set(adapter_cfg.get("target_modules") or [])
    checks = {
        "r": (int(adapter_cfg.get("r", 0)), int(requested.get("r", 16))),
        "alpha": (
            int(adapter_cfg.get("lora_alpha", 0)),
            int(requested.get("alpha", 32)),
        ),
        "target_modules": (observed_targets, expected_targets),
        "task_type": (str(adapter_cfg.get("task_type") or ""), "CAUSAL_LM"),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in checks.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"initial adapter LoRA config mismatch: {mismatches}")
    return {
        **lineage,
        "adapter_config": str(adapter_config_path),
        "adapter_lora_r": checks["r"][0],
        "adapter_lora_alpha": checks["alpha"][0],
        "adapter_target_modules": sorted(observed_targets),
        "adapter_task_type": checks["task_type"][0],
    }


def resolve_cached_arrow_files(
    cache_dir: Path, manifest_paths: list[str]
) -> list[str]:
    """Bind manifest shard names to the selected cache directory.

    Pretokenization manifests are portable artifacts, but older manifests
    record paths relative to the repository.  A node-local cache override must
    not silently mmap those original Ceph paths again.
    """

    resolved: list[str] = []
    for recorded in manifest_paths:
        path = cache_dir / Path(recorded).name
        if not path.is_file():
            raise FileNotFoundError(f"cached Arrow shard does not exist: {path}")
        resolved.append(str(path))
    return resolved


def resolve_resume_checkpoint(value: str | None, output_dir: Path) -> Path | None:
    """Resolve an explicit checkpoint or the newest checkpoint in output_dir.

    Resume is opt-in.  This prevents a benchmark output directory from silently
    inheriting unrelated optimizer state while still making interrupted Taiji
    jobs straightforward to continue.
    """

    requested = str(value or "").strip()
    if not requested or requested.lower() in {"0", "false", "none", "no"}:
        return None
    if requested.lower() in {"1", "true", "auto", "latest"}:
        candidates: list[tuple[int, Path]] = []
        for path in output_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if (path / "trainer_state.json").is_file():
                candidates.append((step, path))
        if not candidates:
            raise FileNotFoundError(
                f"no resumable checkpoint exists under {output_dir}"
            )
        return max(candidates, key=lambda item: item[0])[1]
    checkpoint = Path(requested).expanduser()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    if not (checkpoint / "trainer_state.json").is_file():
        raise FileNotFoundError(
            f"resume checkpoint has no trainer_state.json: {checkpoint}"
        )
    return checkpoint


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
    row: dict[str, Any],
    *,
    require_trace_owned: bool,
    allow_upstream_endpoint_fallback: bool = False,
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
    if metadata.get("upstream_endpoint_fallback") is True:
        if not allow_upstream_endpoint_fallback:
            raise ValueError(f"endpoint fallback is not allowed: {identifier}")
        if calls or results:
            raise ValueError(f"endpoint fallback must not claim tool replay: {identifier}")
        if metadata.get("endpoint_source") != "upstream_frozen_endpoint_fallback":
            raise ValueError(f"endpoint fallback provenance is invalid: {identifier}")
        return {"tool_calls": 0, "tool_results": 0, "finish_trace": 0}
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
    rows: list[dict[str, Any]],
    *,
    require_trace_owned: bool,
    allow_upstream_endpoint_fallback: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Tool-SFT dataset is empty")
    ids: set[str] = set()
    tool_calls = tool_results = finish_rows = trace_bound = 0
    assistant_messages = upstream_endpoint_fallback_rows = 0
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
            row,
            require_trace_owned=require_trace_owned,
            allow_upstream_endpoint_fallback=allow_upstream_endpoint_fallback,
        )
        upstream_endpoint_fallback_rows += int(
            (row.get("metadata") or {}).get("upstream_endpoint_fallback") is True
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
        "upstream_endpoint_fallback_rows": upstream_endpoint_fallback_rows,
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
    has_validation: bool = False,
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
    if has_validation:
        eval_strategy = str(training.get("eval_strategy", "epoch"))
        if "eval_strategy" in fields:
            kwargs["eval_strategy"] = eval_strategy
        elif "evaluation_strategy" in fields:
            kwargs["evaluation_strategy"] = eval_strategy
        kwargs["per_device_eval_batch_size"] = int(
            training.get("per_device_eval_batch_size", 1)
        )
        if "eval_accumulation_steps" in fields:
            kwargs["eval_accumulation_steps"] = int(
                training.get("eval_accumulation_steps", 8)
            )
    if gradient_checkpointing and "gradient_checkpointing_kwargs" in fields:
        kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False,
        }
    for field, default in (
        ("torch_compile", False),
        ("include_tokens_per_second", True),
        ("include_num_input_tokens_seen", True),
    ):
        if field in fields:
            kwargs[field] = bool(training.get(field, default))
    for field in ("torch_compile_backend", "torch_compile_mode"):
        value = str(training.get(field) or "").strip()
        if value and field in fields:
            kwargs[field] = value
    use_liger_kernel = bool(training.get("use_liger_kernel", False))
    if use_liger_kernel:
        if "use_liger_kernel" not in fields:
            raise RuntimeError(
                "use_liger_kernel=true requires a Transformers version that "
                "exposes TrainingArguments.use_liger_kernel"
            )
        kwargs["use_liger_kernel"] = True
        if "liger_kernel_config" in fields:
            kwargs["liger_kernel_config"] = dict(
                training.get("liger_kernel_config") or {}
            )
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
        "--resume-from-checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "resume optimizer/scheduler/RNG state from a checkpoint path; "
            "without a path, select the latest checkpoint in output_dir"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="override training.max_steps for a fixed overfit smoke test",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help="override training.num_train_epochs without editing the frozen YAML",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    initial_adapter_override = os.environ.get(
        "MECHET_INITIAL_ADAPTER_PATH", ""
    ).strip()
    if initial_adapter_override:
        cfg["initial_adapter_path"] = initial_adapter_override
        cfg["initial_adapter_manifest"] = str(
            Path(initial_adapter_override) / "adapter_manifest.json"
        )
    output_override = os.environ.get("MECHET_OUTPUT_DIR", "").strip()
    if output_override:
        cfg["output_dir"] = output_override
    cache_override = os.environ.get("MECHET_PRETOKENIZED_CACHE_DIR", "").strip()
    if cache_override:
        cfg["pretokenized_cache_dir"] = cache_override
    resume_request = (
        args.resume_from_checkpoint
        if args.resume_from_checkpoint is not None
        else os.environ.get(
            "MECHET_RESUME_FROM_CHECKPOINT",
            str((cfg.get("training") or {}).get("resume_from_checkpoint") or ""),
        )
    )
    resume_checkpoint = resolve_resume_checkpoint(
        resume_request,
        Path(str(cfg.get("output_dir") or "outputs/agent/tool_sft")),
    )
    initial_adapter_value = str(cfg.get("initial_adapter_path") or "").strip()
    initial_adapter_lineage = None
    if initial_adapter_value or bool(cfg.get("require_initial_adapter", False)):
        initial_adapter_lineage = validate_initial_adapter_config(
            cfg, validate_lineage(cfg)
        )
    train_file = Path(str(cfg.get("train_file") or ""))
    if not train_file.exists():
        raise FileNotFoundError(f"train_file does not exist: {train_file}")
    cache_value = str(cfg.get("pretokenized_cache_dir") or "").strip()
    cache_dir = Path(cache_value) if cache_value else None
    cache_manifest_path = cache_dir / "manifest.json" if cache_dir else None
    cache_manifest = (
        json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest_path is not None and cache_manifest_path.exists()
        else None
    )
    if cache_dir is not None and cache_manifest is None and not args.dry_run:
        raise FileNotFoundError(
            f"pretokenized cache manifest does not exist: {cache_manifest_path}"
        )
    if cache_manifest is not None and (
        args.limit or int(cfg.get("limit_examples", 0) or 0)
    ):
        raise ValueError("pretokenized cache cannot be combined with a row limit")
    rows = (
        []
        if cache_manifest is not None
        else read_rows(
            train_file,
            limit=args.limit or int(cfg.get("limit_examples", 0) or 0),
        )
    )
    validation_file_value = str(cfg.get("validation_file") or "").strip()
    validation_file = Path(validation_file_value) if validation_file_value else None
    if validation_file is not None and not validation_file.exists():
        raise FileNotFoundError(
            f"validation_file does not exist: {validation_file}"
        )
    contract = dict(cfg.get("contract") or {})
    artifact_status = enforce_artifact_status(train_file, contract)
    require_trace_owned = bool(contract.get("require_trace_owned", True))
    expected_fallback_rows = int(
        contract.get("expected_upstream_endpoint_fallback_rows", 0) or 0
    )
    training = dict(cfg.get("training") or {})
    if args.num_train_epochs is not None:
        if args.num_train_epochs <= 0:
            raise ValueError("--num-train-epochs must be positive")
        training["num_train_epochs"] = float(args.num_train_epochs)
    validation_limit = int(training.get("validation_limit", 0) or 0)
    if args.limit and not validation_limit:
        validation_limit = args.limit
    validation_rows = (
        []
        if cache_manifest is not None
        else (
            read_rows(validation_file, limit=validation_limit)
            if validation_file is not None
            else []
        )
    )
    if bool(training.get("packing", False)):
        raise ValueError(
            "packing=true is unsupported by the pretokenized assistant-only Trainer path"
        )
    if not bool(training.get("assistant_only_loss", True)):
        raise ValueError(
            "train_tool_sft.py requires assistant_only_loss=true; labels are explicitly masked"
        )
    configured_max_steps = int(training.get("max_steps", -1))
    env_max_steps = int(os.environ.get("MECHET_MAX_STEPS", "0") or 0)
    max_steps = int(args.max_steps or env_max_steps or configured_max_steps)
    seed = int(training.get("seed", 17))
    data_seed = int(training.get("data_seed", seed))
    validated_train = (
        dict(cache_manifest["splits"]["train"])
        if cache_manifest is not None
        else validate_rows(
            rows,
            require_trace_owned=require_trace_owned,
            allow_upstream_endpoint_fallback=expected_fallback_rows > 0,
        )
    )
    validated_validation = (
        dict(cache_manifest["splits"]["validation"])
        if cache_manifest is not None
        else (
            validate_rows(
                validation_rows,
                require_trace_owned=require_trace_owned,
                allow_upstream_endpoint_fallback=expected_fallback_rows > 0,
            )
            if validation_rows
            else None
        )
    )
    report = {
        **validated_train,
        "artifact_type": "tool_sft_data_contract",
        "scientific_hypothesis": cfg.get("scientific_hypothesis"),
        "train_file": str(train_file),
        "train_file_sha256": (
            cache_manifest["sources"]["train"]["sha256"]
            if cache_manifest is not None
            else file_sha256(train_file)
        ),
        "validation_file": str(validation_file) if validation_file else None,
        "validation_file_sha256": (
            (
                cache_manifest["sources"]["validation"]["sha256"]
                if cache_manifest is not None
                else file_sha256(validation_file)
            )
            if validation_file
            else None
        ),
        "validation": validated_validation,
        "pretokenized_cache_manifest": (
            str(cache_manifest_path) if cache_manifest is not None else None
        ),
        "model_name_or_path": cfg.get("model_name_or_path"),
        "condition_name": cfg.get("condition_name"),
        "output_dir": cfg.get("output_dir"),
        "dataset_format": "pretokenized_chatml_with_explicit_assistant_labels",
        "assistant_only_loss": True,
        "packing": False,
        "max_steps": max_steps,
        "num_train_epochs": float(training.get("num_train_epochs", 1.0)),
        "seed": seed,
        "data_seed": data_seed,
        "stable_id_manifest": contract.get("stable_id_manifest"),
        "artifact_status": artifact_status,
        "validation_report": contract.get("validation_report"),
        "environment_revision": contract.get("environment_revision"),
        "executor_revision": contract.get("executor_revision"),
        "terminal_tool": contract.get("terminal_tool"),
        "free_form_proof_submission": contract.get("free_form_proof_submission"),
        "real_overfit_smoke_test_required": bool(
            contract.get("real_overfit_smoke_test_required", False)
        ),
        "initial_adapter_lineage": initial_adapter_lineage,
    }
    if report["upstream_endpoint_fallback_rows"] != expected_fallback_rows:
        raise ValueError(
            "upstream endpoint fallback count mismatch: "
            f"{report['upstream_endpoint_fallback_rows']} != {expected_fallback_rows}"
        )
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    try:
        import torch
        from datasets import Dataset, concatenate_datasets
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
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

    if cache_manifest is not None:
        if str(cache_manifest.get("model_name_or_path")) != model_name:
            raise ValueError("pretokenized cache model_name_or_path mismatch")
        if int(cache_manifest.get("max_length") or 0) != int(
            training.get("max_length", 12288)
        ):
            raise ValueError("pretokenized cache max_length mismatch")
        encoded_rows = []
        audit = dict(cache_manifest["splits"]["train"])
        audit["arrow_files"] = resolve_cached_arrow_files(
            cache_dir, list(audit["arrow_files"])
        )
    else:
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

    encoded_validation_rows: list[dict[str, list[int]]] = []
    if cache_manifest is not None:
        validation_audit = dict(cache_manifest["splits"]["validation"])
        validation_audit["arrow_files"] = resolve_cached_arrow_files(
            cache_dir, list(validation_audit["arrow_files"])
        )
        report["validation_tokenizer_audit"] = validation_audit
    elif validation_rows:
        encoded_validation_rows, validation_audit = tokenize_rows(
            validation_rows,
            tokenizer,
            max_length=int(training.get("max_length", 12288)),
        )
        report["validation_tokenizer_audit"] = validation_audit
        if validation_audit["truncation_count"]:
            raise ValueError(
                "validation Tool-SFT rows exceed max_length; "
                f"max={validation_audit['max_input_tokens']} "
                f"budget={validation_audit['configured_max_length']} "
                f"examples={validation_audit['over_budget_ids'][:10]}"
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
    use_qlora = bool(training.get("qlora", False))
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype or torch.float16,
        )
        if use_qlora
        else None
    )
    attention_implementation = str(
        training.get("attention_implementation") or ""
    ).strip()
    if attention_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "attention_implementation=flash_attention_2 requires the "
                "optional flash-attn package; use sdpa for the PyTorch Flash-SDPA "
                "backend when no compatible wheel is installed"
            ) from exc
    if bool(training.get("require_flash_sdp", False)):
        if attention_implementation != "sdpa":
            raise ValueError("require_flash_sdp=true requires attention_implementation=sdpa")
        if not torch.cuda.is_available() or not torch.backends.cuda.flash_sdp_enabled():
            raise RuntimeError("PyTorch Flash-SDPA is unavailable on this CUDA runtime")
    if attention_implementation == "mechet_xformers_causal":
        if int(training.get("per_device_train_batch_size", 1)) != 1:
            raise ValueError(
                "mechet_xformers_causal requires per_device_train_batch_size=1"
            )
        if int(training.get("per_device_eval_batch_size", 1)) != 1:
            raise ValueError(
                "mechet_xformers_causal requires per_device_eval_batch_size=1"
            )
        from mechet.xformers_attention import register_xformers_attention

        register_xformers_attention()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=resolved_revision or requested_revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map={"": local_rank} if use_qlora else None,
        attn_implementation=attention_implementation or None,
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
    if use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if initial_adapter_value:
        model = PeftModel.from_pretrained(
            model,
            initial_adapter_value,
            is_trainable=True,
        )
    else:
        model = get_peft_model(model, peft_config)
    if gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
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
        has_validation=validated_validation is not None,
    )
    training_args = TrainingArguments(**training_kwargs)
    if cache_manifest is not None:
        dataset = concatenate_datasets(
            [Dataset.from_file(path) for path in audit["arrow_files"]]
        )
        validation_dataset = concatenate_datasets(
            [
                Dataset.from_file(path)
                for path in validation_audit["arrow_files"]
            ]
        )
    else:
        try:
            dataset = Dataset.from_list(encoded_rows, on_mixed_types="use_json")
        except TypeError:
            dataset = Dataset.from_list(encoded_rows)
        try:
            validation_dataset = (
                Dataset.from_list(encoded_validation_rows, on_mixed_types="use_json")
                if encoded_validation_rows
                else None
            )
        except TypeError:
            validation_dataset = (
                Dataset.from_list(encoded_validation_rows)
                if encoded_validation_rows
                else None
            )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
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
        "qlora": use_qlora,
        "quantization": "bnb_nf4_double_quant" if use_qlora else None,
        "attention_implementation": attention_implementation or None,
        "require_flash_sdp": bool(training.get("require_flash_sdp", False)),
        "resume_from_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "torch_compile": bool(training.get("torch_compile", False)),
        "torch_compile_backend": training.get("torch_compile_backend"),
        "torch_compile_mode": training.get("torch_compile_mode"),
        "use_liger_kernel": bool(training.get("use_liger_kernel", False)),
        "liger_kernel_config": dict(training.get("liger_kernel_config") or {}),
        "length_grouping": grouping,
        "initial_adapter_path": initial_adapter_value or None,
        "initial_adapter_sha256": (
            initial_adapter_lineage.get("initial_adapter_sha256_actual")
            if initial_adapter_lineage
            else None
        ),
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
        "initial_adapter_path": initial_adapter_value or None,
        "initial_adapter_sha256": (
            initial_adapter_lineage.get("initial_adapter_sha256_actual")
            if initial_adapter_lineage
            else None
        ),
        "initial_adapter_condition_name": (
            initial_adapter_lineage.get("adapter_condition_name")
            if initial_adapter_lineage
            else None
        ),
    }
    (output / "adapter_manifest.json").write_text(
        json.dumps(adapter_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
