#!/usr/bin/env python3
"""Matched per-state vLLM eager/CUDA-graph/structured-decoding benchmark.

This measures proposal compilation and throughput on fixed, oracle-provided
states; it is not an autonomous product-start endpoint evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.electron_pointer import (
    UnsupportedPointerEvent,
    action_pointer_targets,
    candidate_keys,
    parse_pointer_example,
)
from mechet.vnext_structured_actions import parse_structured_action, structured_action_schema
from scripts.eval_natural_language_event_local import prediction_call


MODES = {
    "eager_prefix": (True, True, "none"),
    "graph_no_prefix": (False, False, "none"),
    "graph_prefix": (False, True, "none"),
    "graph_schema": (False, True, "schema"),
    "graph_inventory": (False, True, "inventory"),
}


def fixed_examples(path: Path, *, count: int, seed: int, rank: int, world: int):
    selected = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            try:
                example = parse_pointer_example(row)
            except UnsupportedPointerEvent:
                continue
            if example is None:
                continue
            digest = hashlib.sha256(f"{seed}:{example.row_id}".encode()).digest()
            selected.append((digest, example))
    selected.sort(key=lambda pair: pair[0])
    if count > len(selected):
        raise ValueError(f"requested {count} examples; only {len(selected)} event decisions")
    return [example for index, (_, example) in enumerate(selected[:count]) if index % world == rank]


def structural_handle_ok(name: str, arguments: dict[str, Any], example) -> bool:
    if name != "apply_electron_flow":
        return name in {"import_fragments", "finish_trace"}
    try:
        sources, sinks = action_pointer_targets(arguments)
        source = set(candidate_keys(len(example.atom_names), example.bonds, source=True))
        sink = set(candidate_keys(len(example.atom_names), example.bonds, source=False))
        return all(site in source for site in sources) and all(site in sink for site in sinks)
    except (KeyError, TypeError, ValueError):
        return False


def shard_assignment() -> tuple[int, int]:
    """Require one visible GPU for each independent vLLM benchmark worker."""
    rank = int(os.environ.get("VNEXT_RANK", "0"))
    world = int(os.environ.get("VNEXT_WORLD_SIZE", "1"))
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"invalid vNext shard assignment: rank={rank}, world={world}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if world > 1 and (not visible or len(visible.split(",")) != 1):
        raise ValueError("distributed vLLM benchmark requires exactly one visible GPU per worker")
    return rank, world


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = p.parse_args()
    if args.count < 1 or args.max_new_tokens < 1:
        raise ValueError("positive count/token budgets required")
    rank, world = shard_assignment()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams
    import vllm

    if vllm.__version__ != "0.8.5":
        raise RuntimeError(f"validated benchmark uses vLLM 0.8.5, not {vllm.__version__}")
    examples = fixed_examples(args.data, count=args.count, seed=args.seed, rank=rank, world=world)
    args.output.mkdir(parents=True, exist_ok=True)
    lora = LoRARequest("pr71_stage2", 1, str(args.adapter.resolve()))
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in args.modes:
        eager, prefix_cache, guidance = MODES[mode]
        print(json.dumps({"phase": "load", "rank": rank, "mode": mode, "examples": len(examples)}), flush=True)
        engine = LLM(
            model=args.model, revision=args.revision, tokenizer=args.model,
            trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=1,
            gpu_memory_utilization=0.84, max_model_len=4096, max_num_seqs=16,
            enable_prefix_caching=prefix_cache, enforce_eager=eager,
            enable_lora=True, max_lora_rank=16, seed=args.seed + rank,
        )
        tokenizer = engine.get_tokenizer()
        prompt_ids = [
            tokenizer.encode(
                render_qwen_sft_tool_prefix(tokenizer, example.messages, tools=example.tools),
                add_special_tokens=False,
            )
            for example in examples
        ]
        if any(len(ids) + args.max_new_tokens > 4096 for ids in prompt_ids):
            raise ValueError("benchmark prompt exceeds 4096-token context")
        params = []
        for example in examples:
            guided = None
            if guidance != "none":
                guided = GuidedDecodingParams.from_optional(
                    json=structured_action_schema(
                        example if guidance == "inventory" else None,
                        inventory_handles=guidance == "inventory",
                    ),
                    backend="xgrammar:no-fallback",
                )
            params.append(SamplingParams(
                n=1, temperature=0.0, max_tokens=args.max_new_tokens,
                guided_decoding=guided,
            ))
        start = time.monotonic()
        generated = engine.generate(
            [{"prompt_token_ids": ids} for ids in prompt_ids], params,
            lora_request=lora, use_tqdm=False,
        )
        wall = time.monotonic() - start
        records = []
        for example, result in zip(examples, generated, strict=True):
            output = result.outputs[0]
            raw = output.text
            try:
                if guidance == "none":
                    name, arguments, error = prediction_call(raw, tokenizer)
                    if error:
                        raise ValueError(str(error))
                else:
                    name, arguments = parse_structured_action(raw)
                parsed = True
                handle_ok = structural_handle_ok(name, arguments, example)
                signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                parsed, handle_ok, signature = False, False, ""
                error = f"{type(exc).__name__}:{exc}"
            records.append({
                "id": example.row_id, "mode": mode, "parsed": parsed,
                "handle_ok": handle_ok, "signature": signature,
                "output_tokens": len(output.token_ids), "raw": raw,
                "error": "" if parsed else str(error),
            })
        by_mode[mode] = records
        path = args.output / f"{mode}.rank{rank:02d}.jsonl"
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        baseline = by_mode.get("eager_prefix")
        exact = None if baseline is None else sum(
            bool(a["signature"] and a["signature"] == b["signature"])
            for a, b in zip(baseline, records, strict=True)
        ) / max(len(records), 1)
        report = {
            "rank": rank, "mode": mode, "n_states": len(records), "wall_seconds": wall,
            "states_per_second": len(records) / max(wall, 1e-9),
            "output_tokens_per_second": sum(r["output_tokens"] for r in records) / max(wall, 1e-9),
            "parse_success": sum(r["parsed"] for r in records) / max(len(records), 1),
            "handle_valid": sum(r["handle_ok"] for r in records) / max(len(records), 1),
            "exact_action_vs_eager_prefix": exact,
            "scope": "oracle_state_engineering_benchmark_not_product_start_accuracy",
        }
        (args.output / f"{mode}.rank{rank:02d}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        del engine
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
