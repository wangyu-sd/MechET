#!/usr/bin/env python3
"""Matched product-start EARHO actor/critic smoke on a frozen validation monitor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_earho_v2 import validate_contract
from scripts.run_natural_language_anchor_branch_rl import _sha256, run_workers
from scripts.train_python_template_rlvr import _load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--critic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontier", type=int, required=True)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    validate_contract(cfg)
    campaign = args.campaign
    plan = json.loads((campaign / "plan.json").read_text())
    monitor = campaign / "validation_monitor.jsonl"
    expected_sha = plan["prepared_files"]["validation_monitor.jsonl"]
    if _sha256(monitor) != expected_sha:
        raise ValueError("frozen EARHO validation monitor SHA-256 mismatch")
    if plan["initial_adapter_model_sha256"] != cfg["initial_adapter_model_sha256"]:
        raise ValueError("smoke and campaign have different Stage-II parents")
    if not (args.actor / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("actor adapter is incomplete")
    if args.critic and not (args.critic / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("critic adapter is incomplete")
    if args.frontier < 1:
        raise ValueError("frontier must be positive")
    if args.beam_width is not None and args.beam_width < 1:
        raise ValueError("beam width must be positive")
    beam_width = int(args.beam_width or cfg["rollout"]["continuation_beam_width"])

    smoke = {
        "artifact_type": "earho_v2_matched_component_smoke",
        "actor": str(args.actor),
        "actor_sha256": _sha256(args.actor / "adapter_model.safetensors"),
        "critic": str(args.critic) if args.critic else None,
        "critic_sha256": _sha256(args.critic / "adapter_model.safetensors") if args.critic else None,
        "monitor": str(monitor),
        "monitor_sha256": expected_sha,
        "frontier": args.frontier,
        "beam_width": beam_width,
        "prompt_prefix_contract": "qwen_sft_aligned_no_think_v1",
        "candidate_count_per_reaction": 2,
        "test_used": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "smoke_plan.json"
    if marker.is_file() and json.loads(marker.read_text()) != smoke:
        raise ValueError("smoke output belongs to different pinned inputs")
    marker.write_text(json.dumps(smoke, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(smoke, ensure_ascii=False), flush=True)
        return 0
    eval_cfg = dict(cfg, value_adapter_path=str(args.critic) if args.critic else None)
    eval_cfg["rollout"] = dict(cfg["rollout"], continuation_beam_width=beam_width)
    _, result = run_workers(
        eval_cfg, monitor, args.actor, args.output / "validation",
        frontier=args.frontier, round_index=-1, evaluation=True,
    )
    summary = {
        **smoke,
        "groups": result["groups"],
        "candidates": result["candidates"],
        "candidate_execution_rate": result["candidate_execution_rate"],
        "candidate_endpoint_rate": result["candidate_endpoint_rate"],
        "group_pass_at_k": result["group_pass_at_k"],
    }
    (args.output / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
