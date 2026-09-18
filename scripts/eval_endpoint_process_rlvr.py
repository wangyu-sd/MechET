#!/usr/bin/env python3
"""Compare parent SFT and process-RL adapters on product-start K=1 monitor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mechet.endpoint_process_rl_env import EndpointProcessRLEnv, ProcessRewardConfig
from mechet.chemical_runtime import require_endpoint_process_rdkit
from mechet.endpoint_process_rollout import run_rollout_group
from mechet.model import resolve_qwen_model_path
from train_endpoint_process_rlvr import (
    activate_adapter,
    all_reduce_values,
    distributed_context,
    load_actor,
    read_jsonl,
)


def evaluate_one(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    config: ProcessRewardConfig,
    max_proposals: int,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    env = EndpointProcessRLEnv(
        row, prefix_events=0, start_horizon="product", reward_config=config
    )
    rollout = run_rollout_group(
        model,
        tokenizer,
        [env],
        max_proposals=max_proposals,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=seed,
    )[0]
    summary = env.summary().to_dict()
    return {
        **summary,
        "generated_tokens": sum(len(span.completion_token_ids) for span in rollout.spans),
        "sampled_events": [credit.to_dict() for credit in env.credits],
    }


def local_totals(records: list[dict[str, Any]], method: str) -> dict[str, float]:
    totals = {
        "n": 0.0,
        "endpoint_exact": 0.0,
        "formal_terminal": 0.0,
        "explicit_finish": 0.0,
        "rejected_proposals": 0.0,
        "generated_tokens": 0.0,
        "committed_events": 0.0,
    }
    for record in records:
        value = record[method]
        totals["n"] += 1
        totals["endpoint_exact"] += int(value["endpoint_exact"])
        totals["formal_terminal"] += int(value["formal_terminal"])
        totals["explicit_finish"] += int(value["explicit_finish"])
        totals["rejected_proposals"] += float(value["rejected_proposals"])
        totals["generated_tokens"] += float(value["generated_tokens"])
        totals["committed_events"] += float(value["generated_committed_events"])
    return totals


def rates(totals: Mapping[str, float]) -> dict[str, float]:
    n = max(float(totals["n"]), 1.0)
    return {
        "n": float(totals["n"]),
        "endpoint_pass_at_1": float(totals["endpoint_exact"]) / n,
        "formal_terminal_rate": float(totals["formal_terminal"]) / n,
        "explicit_finish_rate": float(totals["explicit_finish"]) / n,
        "invalid_actions_per_target": float(totals["rejected_proposals"]) / n,
        "mean_generated_tokens": float(totals["generated_tokens"]) / n,
        "mean_trajectory_length": float(totals["committed_events"]) / n,
    }


def promotion_decision(
    parent_metrics: Mapping[str, float], actor_metrics: Mapping[str, float]
) -> bool:
    return bool(
        actor_metrics["endpoint_pass_at_1"] > 0.05
        and actor_metrics["endpoint_pass_at_1"]
        > parent_metrics["endpoint_pass_at_1"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-file", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--actor-adapter", type=Path, required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-proposals", type=int, default=24)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    runtime_version = require_endpoint_process_rdkit()
    rank, world_size, local_rank = distributed_context()
    if rank == 0:
        print(json.dumps({"type": "runtime", "rdkit": runtime_version}), flush=True)
    rows = read_jsonl(args.monitor_file)
    base_model = args.base_model or resolve_qwen_model_path() or ""
    if not base_model:
        raise ValueError("base model is required")
    model, tokenizer, revision = load_actor(
        base_model=base_model,
        parent_adapter=args.parent_adapter,
        actor_adapter=args.actor_adapter,
        local_rank=local_rank,
        use_4bit=not args.no_4bit,
    )
    reward_config = ProcessRewardConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index % world_size != rank:
            continue
        activate_adapter(model, "reference", trainable=False)
        parent = evaluate_one(
            model,
            tokenizer,
            row,
            config=reward_config,
            max_proposals=args.max_proposals,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + index,
        )
        activate_adapter(model, "default", trainable=False)
        actor = evaluate_one(
            model,
            tokenizer,
            row,
            config=reward_config,
            max_proposals=args.max_proposals,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + index,
        )
        record = {"id": row["id"], "parent": parent, "actor": actor}
        records.append(record)
        print(
            f"[endpoint-monitor] rank={rank} {len(records)} id={row['id']} "
            f"parent={int(parent['endpoint_exact'])} actor={int(actor['endpoint_exact'])}",
            flush=True,
        )
    prediction_path = args.output_dir / f"predictions.shard-{rank:03d}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    parent_totals = all_reduce_values(local_totals(records, "parent"), world_size=world_size)
    actor_totals = all_reduce_values(local_totals(records, "actor"), world_size=world_size)
    parent_metrics = rates(parent_totals)
    actor_metrics = rates(actor_totals)
    if rank == 0:
        summary = {
            "artifact_type": "endpoint_process_rlvr_monitor_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "product_only": True,
            "k": 1,
            "search_enabled": False,
            "gold_used_for_generation": False,
            "parent_adapter": str(args.parent_adapter),
            "actor_adapter": str(args.actor_adapter),
            "base_model_revision": revision,
            "parent": parent_metrics,
            "actor": actor_metrics,
            "endpoint_pass_absolute_gain": (
                actor_metrics["endpoint_pass_at_1"]
                - parent_metrics["endpoint_pass_at_1"]
            ),
            "promotion": promotion_decision(parent_metrics, actor_metrics),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if world_size > 1:
        import torch

        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
