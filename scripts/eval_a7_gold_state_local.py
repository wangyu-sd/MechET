#!/usr/bin/env python3
"""Evaluate A7's next event at authoritative gold states (F-oracle, K=1)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.a7_rescue import (
    assistant_event_targets,
    audit_gold_event,
    local_prediction_metrics,
)
from mechet.agent_inference import parse_tool_calls
from mechet.assistant_masking import render_chat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def event_key(identifier: str, event_index: int) -> str:
    return f"{identifier}::event-{event_index:03d}"


def distributed_coordinates() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not 0 <= rank < world:
        raise ValueError("invalid rank/world-size")
    return rank, world, local_rank


def nf4_available() -> tuple[bool, str]:
    """Return whether the optional bitsandbytes NF4 backend is installed."""
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        return False, "bitsandbytes-not-installed"
    return True, f"bitsandbytes-{version}"


def collect_tasks(rows: list[dict]) -> list[dict]:
    tasks = []
    for row in rows:
        steps = list(row["metadata"]["trace_plan"]["steps"])
        targets = assistant_event_targets(row)
        if len(steps) != len(targets):
            raise ValueError(f"{row['id']}: step/message count mismatch")
        stratum = "short" if len(steps) <= 2 else "medium" if len(steps) <= 4 else "long"
        for event_index, (message_index, supervised_moves) in enumerate(targets):
            step = steps[event_index]
            tasks.append(
                {
                    "key": event_key(row["id"], event_index),
                    "id": row["id"],
                    "event_index": event_index,
                    "event_depth": event_index + 1,
                    "stratum": stratum,
                    "messages": row["messages"][:message_index],
                    "tools": row.get("tools") or [],
                    "state_before": step["state_before"],
                    "state_after": step["state_after"],
                    "gold_moves": supervised_moves,
                    "gold_gate": audit_gold_event(step),
                }
            )
    return tasks


def prediction_call(text: str, tokenizer) -> tuple[str, dict, str]:
    try:
        calls = parse_tool_calls(text, tokenizer=tokenizer)
    except Exception as exc:
        return "", {}, f"PARSE_ERROR:{exc}"
    if len(calls) != 1:
        return "", {}, f"EXPECTED_ONE_TOOL_CALL:observed={len(calls)}"
    return calls[0].name, calls[0].arguments, ""


def run(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rank, world, local_rank = distributed_coordinates()
    torch.cuda.set_device(local_rank)
    rows = read_jsonl(args.data)
    tasks = collect_tasks(rows)
    selected = [task for index, task in enumerate(tasks) if index % world == rank]
    if args.limit_events:
        selected = selected[: args.limit_events]
    args.output.mkdir(parents=True, exist_ok=True)
    shard = args.output / f"events.shard-{rank:02d}-of-{world:02d}.jsonl"
    completed = {row["key"] for row in read_jsonl(shard)} if shard.exists() else set()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(local_rank)[0] >= 8 else torch.float16
    has_nf4, load_backend = nf4_available()
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": {"": local_rank},
        "attn_implementation": "sdpa",
    }
    if has_nf4:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        load_backend = f"{str(dtype).removeprefix('torch.')}-unquantized ({load_backend})"
    print(
        f"[meteor-a7-local] rank={rank}/{world} model_load={load_backend}",
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    device = next(model.parameters()).device
    print(
        f"[meteor-a7-local] rank={rank}/{world} gpu={torch.cuda.get_device_name(local_rank)} tasks={len(selected)}",
        flush=True,
    )
    with shard.open("a", encoding="utf-8") as sink:
        for ordinal, task in enumerate(selected, 1):
            if task["key"] in completed:
                continue
            started = time.time()
            prompt = render_chat(
                tokenizer,
                task["messages"],
                tools=task["tools"],
                add_generation_prompt=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_length = int(encoded["input_ids"].shape[1])
            if input_length + args.max_new_tokens > args.max_context:
                generated_text = ""
                name, arguments, generation_error = "", {}, "CONTEXT_BUDGET_EXCEEDED"
                generated_tokens = 0
            else:
                encoded = {key: value.to(device) for key, value in encoded.items()}
                try:
                    with torch.inference_mode():
                        output = model.generate(
                            **encoded,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    suffix = output[0, input_length:]
                    generated_tokens = int(suffix.numel())
                    generated_text = tokenizer.decode(suffix, skip_special_tokens=False)
                    name, arguments, generation_error = prediction_call(
                        generated_text, tokenizer
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    generated_text = ""
                    name, arguments, generation_error = "", {}, "CUDA_OOM"
                    generated_tokens = 0
            metrics = local_prediction_metrics(
                state_before=task["state_before"],
                state_after=task["state_after"],
                gold_moves=task["gold_moves"],
                predicted_name=name,
                predicted_arguments=arguments,
            )
            record = {
                "key": task["key"],
                "id": task["id"],
                "event_index": task["event_index"],
                "event_depth": task["event_depth"],
                "stratum": task["stratum"],
                "gold_gate": task["gold_gate"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_tokens": input_length,
                "generated_tokens": generated_tokens,
                "generated_text": generated_text,
                "predicted_name": name,
                "predicted_arguments": arguments,
                "generation_error": generation_error,
                "seconds": time.time() - started,
                **metrics,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[meteor-a7-local] rank={rank} done={ordinal}/{len(selected)} key={task['key']} "
                f"exact={metrics['event_exact']} execute={metrics['formal_execute']} "
                f"successor={metrics['successor_exact']} seconds={record['seconds']:.1f}",
                flush=True,
            )
    return 0


def aggregate(args: argparse.Namespace) -> int:
    source_rows = read_jsonl(args.data)
    tasks = collect_tasks(source_rows)
    expected = {task["key"] for task in tasks}
    rows = []
    for path in sorted(args.output.glob("events.shard-*-of-*.jsonl")):
        rows.extend(read_jsonl(path))
    observed = [row["key"] for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate event predictions")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    metric_names = (
        "well_formed_event",
        "event_exact",
        "formal_execute",
        "successor_exact",
        "source_site_exact",
        "sink_site_exact",
        "source_type_exact",
        "sink_type_exact",
    )

    def summary(group: list[dict]) -> dict:
        return {
            "n": len(group),
            **{name: sum(bool(row.get(name)) for row in group) for name in metric_names},
        }

    by_stratum = {
        name: summary([row for row in rows if row["stratum"] == name])
        for name in ("short", "medium", "long")
    }
    by_depth = {
        str(depth): summary([row for row in rows if row["event_depth"] == depth])
        for depth in sorted({row["event_depth"] for row in rows})
    }
    report = {
        "artifact_type": "a7_gold_state_local_k1_v1",
        "condition": "F-oracle_gold_prefix_greedy_next_event",
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "adapter": str(args.adapter),
        "adapter_model_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "model": str(args.model),
        "planned_events": len(expected),
        "completed_events": len(rows),
        "missing_events": len(missing),
        "extra_events": len(extra),
        "complete": not missing and not extra,
        "overall": summary(rows),
        "by_stratum": by_stratum,
        "by_depth": by_depth,
        "generation_errors": dict(Counter(row["generation_error"] for row in rows if row["generation_error"])),
        "execution_errors": dict(Counter(row["execution_error"] for row in rows if row["execution_error"])),
        "claim_boundary": "Gold-prefix F-oracle diagnostic on validation; not product-only free rollout and not test accuracy.",
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--limit-events", type=int, default=0)
    args = parser.parse_args()
    return run(args) if args.command == "run" else aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
