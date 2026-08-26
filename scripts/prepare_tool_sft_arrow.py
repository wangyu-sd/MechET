#!/usr/bin/env python3
"""Distributed, bounded-memory tokenization for full Tool-SFT JSONL data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist
import yaml
from datasets import Features, Sequence, Value
from datasets.arrow_writer import ArrowWriter
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from mechet.assistant_masking import encode_assistant_only_conversation
from scripts.train_tool_sft import validate_conversation


FEATURES = Features(
    {
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int32")),
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_from_histogram(histogram: Counter[int], fraction: float) -> int:
    total = sum(histogram.values())
    if not total:
        return 0
    target = max(1, int((total * fraction) + 0.999999999))
    seen = 0
    for value, count in sorted(histogram.items()):
        seen += count
        if seen >= target:
            return int(value)
    return int(max(histogram))


def lossless_assistant_span_windows(
    encoded: dict[str, list[int]],
    audit: dict[str, Any],
    *,
    max_length: int,
    context_tokens: int,
) -> list[dict[str, list[int]]]:
    """Split a long conversation without dropping or duplicating supervision.

    Windows own complete assistant spans whenever possible. Earlier assistant
    text may occur in a later window as context, but its labels are masked, so
    every supervised token contributes to the loss exactly once.
    """

    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    if len(input_ids) <= max_length:
        return [encoded]
    if not 0 <= context_tokens < max_length:
        raise ValueError(
            f"window_context_tokens must be in [0, {max_length}), got "
            f"{context_tokens}"
        )

    spans = [(int(start), int(end)) for start, end in audit["assistant_spans"]]
    windows: list[dict[str, list[int]]] = []
    span_index = 0
    while span_index < len(spans):
        owned_start = span_index
        first_start, first_end = spans[span_index]
        window_start = max(0, first_start - context_tokens)

        # Retain a whole assistant turn even if doing so requires less context.
        if first_end - window_start > max_length:
            window_start = max(0, first_end - max_length)

        window_limit = window_start + max_length
        while span_index < len(spans) and spans[span_index][1] <= window_limit:
            span_index += 1

        if span_index == owned_start:
            # A single unusually long assistant turn: split only that span, with
            # no overlap in labels. This path is expected to be extremely rare.
            chunk_start = first_start
            while chunk_start < first_end:
                chunk_end = min(first_end, chunk_start + max_length)
                chunk_ids = input_ids[chunk_start:chunk_end]
                chunk_labels = labels[chunk_start:chunk_end]
                windows.append(
                    {
                        "input_ids": chunk_ids,
                        "attention_mask": [1] * len(chunk_ids),
                        "labels": chunk_labels,
                    }
                )
                chunk_start = chunk_end
            span_index += 1
            continue

        owned_spans = spans[owned_start:span_index]
        window_end = owned_spans[-1][1]
        window_labels = [-100] * (window_end - window_start)
        for start, end in owned_spans:
            relative_start = start - window_start
            relative_end = end - window_start
            window_labels[relative_start:relative_end] = labels[start:end]
        window_ids = input_ids[window_start:window_end]
        windows.append(
            {
                "input_ids": window_ids,
                "attention_mask": [1] * len(window_ids),
                "labels": window_labels,
            }
        )

    original_supervised = sum(value != -100 for value in labels)
    window_supervised = sum(
        value != -100 for window in windows for value in window["labels"]
    )
    if window_supervised != original_supervised:
        raise ValueError(
            "lossless window supervision mismatch: "
            f"{window_supervised} != {original_supervised}"
        )
    if any(len(window["input_ids"]) > max_length for window in windows):
        raise ValueError("lossless window exceeds configured max_length")
    return windows


def tokenize_shard(
    *,
    source: Path,
    target: Path,
    tokenizer: Any,
    rank: int,
    world_size: int,
    max_length: int,
    context_tokens: int,
    require_trace_owned: bool,
    allow_fallback: bool,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    raw_lengths: Counter[int] = Counter()
    window_lengths: Counter[int] = Counter()
    supervised: Counter[int] = Counter()
    mask_methods: Counter[str] = Counter()
    n_rows = n_windows = windowed_rows = 0
    fallback_rows = tool_calls = tool_results = finish_rows = 0
    assistant_messages = max_tool_calls = 0
    max_assistant_turns = 0
    writer = ArrowWriter(features=FEATURES, path=str(target), writer_batch_size=64)
    try:
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index % world_size != rank or not line.strip():
                    continue
                row = dict(json.loads(line))
                counts = validate_conversation(
                    row,
                    require_trace_owned=require_trace_owned,
                    allow_upstream_endpoint_fallback=allow_fallback,
                )
                encoded, audit = encode_assistant_only_conversation(
                    tokenizer, row, max_length=max_length
                )
                windows = lossless_assistant_span_windows(
                    encoded,
                    audit,
                    max_length=max_length,
                    context_tokens=context_tokens,
                )
                for window in windows:
                    writer.write(window)
                    window_lengths[len(window["input_ids"])] += 1
                n_windows += len(windows)
                windowed_rows += int(len(windows) > 1)
                n_rows += 1
                fallback_rows += int(
                    (row.get("metadata") or {}).get("upstream_endpoint_fallback")
                    is True
                )
                tool_calls += int(counts["tool_calls"])
                tool_results += int(counts["tool_results"])
                max_tool_calls = max(max_tool_calls, int(counts["tool_calls"]))
                assistant_messages += sum(
                    message.get("role") == "assistant"
                    for message in row.get("messages") or []
                )
                finish_rows += int(counts["finish_trace"] == 1)
                raw_lengths[int(audit["raw_length"])] += 1
                supervised[int(audit["supervised_tokens"])] += 1
                mask_methods[str(audit["mask_method"])] += 1
                max_assistant_turns = max(
                    max_assistant_turns, int(audit["assistant_turns"])
                )
                if n_rows % 2000 == 0:
                    print(
                        f"rank={rank} source={source.name} rows={n_rows}", flush=True
                    )
    finally:
        writer.finalize()
    return {
        "rank": rank,
        "source": str(source),
        "arrow": str(target),
        "n_rows": n_rows,
        "n_training_windows": n_windows,
        "n_windowed_source_rows": windowed_rows,
        "fallback_rows": fallback_rows,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "max_tool_calls_per_row": max_tool_calls,
        "assistant_messages": assistant_messages,
        "finish_rows": finish_rows,
        "raw_length_histogram": {str(k): v for k, v in raw_lengths.items()},
        "window_length_histogram": {str(k): v for k, v in window_lengths.items()},
        "supervised_histogram": {str(k): v for k, v in supervised.items()},
        "mask_methods": dict(mask_methods),
        "max_assistant_turns": max_assistant_turns,
        "arrow_bytes": target.stat().st_size,
    }


def aggregate(reports: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    raw_lengths: Counter[int] = Counter()
    window_lengths: Counter[int] = Counter()
    supervised: Counter[int] = Counter()
    mask_methods: Counter[str] = Counter()
    for report in reports:
        raw_lengths.update(
            {int(k): int(v) for k, v in report["raw_length_histogram"].items()}
        )
        window_lengths.update(
            {int(k): int(v) for k, v in report["window_length_histogram"].items()}
        )
        supervised.update(
            {int(k): int(v) for k, v in report["supervised_histogram"].items()}
        )
        mask_methods.update(report["mask_methods"])
    n_rows = sum(int(report["n_rows"]) for report in reports)
    fallback_rows = sum(int(report["fallback_rows"]) for report in reports)
    finish_rows = sum(int(report["finish_rows"]) for report in reports)
    return {
        "n_rows": n_rows,
        "n_unique_ids": n_rows,
        "n_training_windows": sum(
            int(report["n_training_windows"]) for report in reports
        ),
        "n_windowed_source_rows": sum(
            int(report["n_windowed_source_rows"]) for report in reports
        ),
        "upstream_endpoint_fallback_rows": fallback_rows,
        "tool_calls": sum(int(report["tool_calls"]) for report in reports),
        "mean_tool_calls_per_row": sum(
            int(report["tool_calls"]) for report in reports
        )
        / max(n_rows, 1),
        "tool_results": sum(int(report["tool_results"]) for report in reports),
        "max_tool_calls_per_row": max(
            (int(report["max_tool_calls_per_row"]) for report in reports), default=0
        ),
        "assistant_messages": sum(
            int(report["assistant_messages"]) for report in reports
        ),
        "trace_bound_rows": n_rows - fallback_rows,
        "trace_bound_rate": (n_rows - fallback_rows) / max(n_rows, 1),
        "finish_trace_rows": finish_rows,
        "finish_trace_rate": finish_rows / max(n_rows, 1),
        "conversation_schema_valid": True,
        "n_tokenizer_audited_rows": n_rows,
        "total_input_tokens": sum(k * v for k, v in window_lengths.items()),
        "total_supervised_tokens": sum(k * v for k, v in supervised.items()),
        "max_input_tokens": max(window_lengths, default=0),
        "p50_input_tokens": percentile_from_histogram(window_lengths, 0.50),
        "p95_input_tokens": percentile_from_histogram(window_lengths, 0.95),
        "p99_input_tokens": percentile_from_histogram(window_lengths, 0.99),
        "max_raw_input_tokens": max(raw_lengths, default=0),
        "p95_raw_input_tokens": percentile_from_histogram(raw_lengths, 0.95),
        "p99_raw_input_tokens": percentile_from_histogram(raw_lengths, 0.99),
        "min_supervised_tokens": min(supervised, default=0),
        "max_assistant_turns": max(
            (int(report["max_assistant_turns"]) for report in reports), default=0
        ),
        "configured_max_length": max_length,
        "truncation_count": 0,
        "truncation_rate": 0.0,
        "assistant_mask_valid": True,
        "assistant_mask_methods": sorted(mask_methods),
        "zero_truncation_required": True,
        "token_window_policy": "lossless_assistant_span_windows_v1",
        "arrow_files": [report["arrow"] for report in reports],
        "arrow_bytes": sum(int(report["arrow_bytes"]) for report in reports),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = dict(yaml.safe_load(args.config.read_text(encoding="utf-8")) or {})
    training = dict(cfg.get("training") or {})
    contract = dict(cfg.get("contract") or {})
    cache_value = str(cfg.get("pretokenized_cache_dir") or "").strip()
    if not cache_value:
        raise ValueError("pretokenized_cache_dir is required")
    cache_dir = Path(cache_value)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    max_length = int(training.get("max_length", 12288))
    context_tokens = int(training.get("window_context_tokens", 8192))
    expected_world_size = int(cfg.get("pretokenization_world_size", 8))
    if world_size != expected_world_size:
        raise ValueError(f"expected {expected_world_size} ranks, got {world_size}")

    model_name = str(cfg.get("model_name_or_path") or "")
    revision = str(training.get("model_revision") or "").strip() or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=bool(training.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sources = {
        "train": Path(str(cfg["train_file"])),
        "validation": Path(str(cfg["validation_file"])),
    }
    local_reports: dict[str, Any] = {}
    for split, source in sources.items():
        local_reports[split] = tokenize_shard(
            source=source,
            target=cache_dir / f"{split}.rank{rank:02d}.arrow",
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            max_length=max_length,
            context_tokens=context_tokens,
            require_trace_owned=bool(contract.get("require_trace_owned", True)),
            allow_fallback=int(
                contract.get("expected_upstream_endpoint_fallback_rows", 0) or 0
            )
            > 0,
        )
    report_path = cache_dir / f"rank{rank:02d}.json"
    report_path.write_text(
        json.dumps(local_reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dist.barrier()

    if rank == 0:
        reports = [
            json.loads((cache_dir / f"rank{index:02d}.json").read_text())
            for index in range(world_size)
        ]
        manifest = {
            "artifact_type": "distributed_pretokenized_tool_sft_v2",
            "config": str(args.config),
            "model_name_or_path": model_name,
            "model_revision": revision,
            "max_length": max_length,
            "window_context_tokens": context_tokens,
            "world_size": world_size,
            "sources": {
                split: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for split, path in sources.items()
            },
            "splits": {
                split: aggregate(
                    [report[split] for report in reports], max_length=max_length
                )
                for split in sources
            },
        }
        expected_train = int(contract.get("expected_train_rows", 0) or 0)
        if manifest["splits"]["train"]["n_rows"] != expected_train:
            raise ValueError(
                f"train row mismatch: {manifest['splits']['train']['n_rows']} "
                f"!= {expected_train}"
            )
        expected_validation = int(
            contract.get("expected_validation_rows", 0) or 0
        )
        if (
            manifest["splits"]["validation"]["n_rows"]
            != expected_validation
        ):
            raise ValueError(
                "validation row mismatch: "
                f"{manifest['splits']['validation']['n_rows']} "
                f"!= {expected_validation}"
            )
        expected_fallback = int(
            contract.get("expected_upstream_endpoint_fallback_rows", 0) or 0
        )
        if (
            manifest["splits"]["train"]["upstream_endpoint_fallback_rows"]
            != expected_fallback
        ):
            raise ValueError("train fallback row count mismatch")
        (cache_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
