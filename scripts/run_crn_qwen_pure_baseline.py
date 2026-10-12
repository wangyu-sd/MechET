#!/usr/bin/env python3
"""Resumable, sharded pure-Qwen control for the PR83 FlowER comparison.

This calls the unchanged historical Runtime/rollout with matched-v2 greedy
decoding. It never loads the CRN model, and does not alter its search policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.crn_graph_bridge import file_sha256
from scripts.run_crn_qwen_hybrid import completed_ids, selected_shard
from scripts.run_natural_language_value_search import (
    Runtime, read_selected, rollout, validate_v2_adapter_manifest,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-reactions", type=int, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--model-revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    p.add_argument("--policy-adapter", type=Path, required=True)
    p.add_argument("--compact-history", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.sample_reactions < 1 or not args.source_sha256:
        raise ValueError("pure-Qwen control requires a pinned nonempty source")
    validate_v2_adapter_manifest(
        args.policy_adapter, compact_history=args.compact_history,
        expected_model=args.model, expected_revision=args.model_revision,
    )
    args.matched_v2 = True
    args.vnext_v2_prefix = False
    args.legacy_dual_prompt = False
    args.product_only_remap = True
    args.reject_target_retained_finish = True
    args.branching = args.early_beam = args.late_beam = 1
    args.early_depth = 2
    args.max_decisions = 40
    args.max_imports = 32
    args.max_context = 4096
    args.max_new_tokens = 384
    args.value_adapter = ""
    args.value_kind = "successor_pn"
    args.value_weight = 0.0
    args.pointer_head = ""
    args.pointer_weight = 0.0
    args.planning_sample = False
    args.raw_model_nll = False
    args.no_4bit = True
    args.dtype = "bfloat16"
    return args


def main() -> int:
    args = prepare_args(parser().parse_args())
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world < 1 or not 0 <= rank < world:
        raise ValueError("invalid distributed rank/world")
    import torch

    if torch.cuda.device_count() <= local_rank:
        raise RuntimeError("one CUDA device is required per local rank")
    if file_sha256(args.data) != args.source_sha256:
        raise ValueError("pure-Qwen source SHA-256 mismatch")
    rows = read_selected(args.data, args.sample_reactions, args.seed)
    if len(rows) != args.sample_reactions:
        raise ValueError("source has fewer rows than the frozen evaluation count")
    selected = selected_shard(rows, rank, world)
    expected_ids = {str(row["id"]) for row in selected}
    if len(expected_ids) != len(selected):
        raise ValueError("duplicate selected IDs in pure-Qwen shard")
    adapter_sha = file_sha256(args.policy_adapter / "adapter_model.safetensors")
    contract = {
        "source": str(args.data.resolve()),
        "source_sha256": args.source_sha256,
        "sample_reactions": args.sample_reactions,
        "seed": args.seed,
        "rank": rank,
        "world": world,
        "model": args.model,
        "model_revision": args.model_revision,
        "policy_adapter_sha256": adapter_sha,
        "compact_history": args.compact_history,
        "matched_v2": True,
        "product_only_remap": True,
        "greedy": True,
        "max_new_tokens": 384,
        "max_context": 4096,
        "max_decisions": 40,
        "max_imports": 32,
        "code_sha256": {name: file_sha256(ROOT / name) for name in (
            "scripts/run_crn_qwen_pure_baseline.py",
            "scripts/run_natural_language_value_search.py",
            "src/mechet/endpoints.py",
        )},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path = args.output / f"contract.shard-{rank:02d}-of-{world:02d}.json"
    results_path = args.output / f"results.shard-{rank:02d}-of-{world:02d}.jsonl"
    if args.resume:
        if contract_path.exists():
            if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
                raise ValueError("pure-Qwen resume contract differs from the saved shard")
        elif results_path.exists():
            raise FileExistsError("cannot resume results without a saved contract")
    elif contract_path.exists() or results_path.exists():
        raise FileExistsError("pure-Qwen shard exists; pass --resume only for an identical run")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    done = completed_ids(results_path, expected_ids)
    runtime = Runtime(args, local_rank)
    started = time.perf_counter()
    with results_path.open("a", encoding="utf-8") as sink:
        for row in selected:
            identifier = str(row["id"])
            if identifier in done:
                continue
            row_started = time.perf_counter()
            result = rollout(runtime, row, args)
            result["wall_seconds"] = time.perf_counter() - row_started
            result["policy"] = "pure_qwen3_0_6b_matched_v2_greedy"
            result["qwen_adapter_sha256"] = adapter_sha
            sink.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            sink.flush()
            done.add(identifier)
            print(json.dumps({
                "stage": "pure-qwen-pr83-control", "rank": rank,
                "done": len(done), "total": len(selected), "id": identifier,
                "top1_structural_exact": bool(result["top1_structural_exact"]),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }), flush=True)
    if len(done) != len(selected):
        raise RuntimeError("pure-Qwen shard did not cover all selected reactions")
    print(json.dumps({"stage": "pure-qwen-pr83-control-shard-complete",
                      "rank": rank, "world": world, "evaluated": len(done)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
