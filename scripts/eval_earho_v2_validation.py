#!/usr/bin/env python3
"""Evaluate a pinned EARHO policy on the full executable-view validation split.

This is product-start evaluation, not a test-set or full 31k evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_anchor_branch_rl import read_rows, write_json, write_rows
from scripts.run_earho_v2 import (
    _attach_decisions,
    _evaluation_candidate_count,
    load_earho_config,
    validate_contract,
)
from scripts.run_natural_language_anchor_branch_rl import _sha256, run_workers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--critic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontier", type=int, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    cfg = load_earho_config(args.config)
    validate_contract(cfg)
    if args.frontier < 1:
        raise ValueError("frontier must be positive")
    for adapter in (args.actor, args.critic):
        if adapter is not None and not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"incomplete adapter: {adapter}")

    args.output.mkdir(parents=True, exist_ok=True)
    monitor = args.output / "validation_full.jsonl"
    plan_path = args.output / "plan.json"
    inputs = {
        "artifact_type": "earho_v2_full_executable_view_validation",
        "source_valid_sha256": _sha256(Path(cfg["validation_file"])),
        "history_valid_sha256": _sha256(Path(cfg["history_validation_file"])),
        "actor": str(args.actor),
        "actor_sha256": _sha256(args.actor / "adapter_model.safetensors"),
        "critic": str(args.critic) if args.critic else None,
        "critic_sha256": _sha256(args.critic / "adapter_model.safetensors") if args.critic else None,
        "frontier": args.frontier,
        "beam_width": int(cfg["rollout"]["continuation_beam_width"]),
        "candidate_count_per_reaction": _evaluation_candidate_count(cfg),
        "test_used": False,
    }
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        if {key: plan.get(key) for key in inputs} != inputs:
            raise ValueError("validation output belongs to different pinned inputs")
        if _sha256(monitor) != plan["monitor_sha256"]:
            raise ValueError("full-validation monitor changed")
    else:
        source = read_rows(cfg["validation_file"])
        if len(source) != int(cfg["reaction_denominator"]["valid"]):
            raise ValueError("validation reaction denominator changed")
        rows = _attach_decisions(source, Path(cfg["history_validation_file"]))
        write_rows(monitor, rows)
        plan = dict(inputs, monitor_sha256=_sha256(monitor), groups=len(rows))
        write_json(plan_path, plan)
    if args.prepare_only:
        print(json.dumps(plan, ensure_ascii=False), flush=True)
        return 0

    eval_cfg = dict(cfg, value_adapter_path=str(args.critic) if args.critic else None)
    _, result = run_workers(
        eval_cfg, monitor, args.actor, args.output / "validation",
        frontier=args.frontier, round_index=-1, evaluation=True,
    )
    summary = {
        **plan,
        "groups": result["groups"],
        "candidates": result["candidates"],
        "candidate_execution_rate": result["candidate_execution_rate"],
        "candidate_endpoint_rate": result["candidate_endpoint_rate"],
        "group_pass_at_k": result["group_pass_at_k"],
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
