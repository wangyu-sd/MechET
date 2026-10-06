#!/usr/bin/env python3
"""Run Syntheseus with fresh product-only MechET rollouts at every expansion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.run_natural_language_value_search import (
    Runtime, validate_matched_v2_args, validate_v2_adapter_manifest,
)
from scripts.run_syntheseus_search import file_sha256, read_smiles, run_planner
from mechet.online_syntheseus_adapter import OnlineMechETBackwardReactionModel


MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--stage", choices=("state", "trajectory"), required=True)
    result.add_argument("--targets", type=Path, required=True)
    result.add_argument("--inventory", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--algorithm", choices=("retro_star", "breadth_first"), default="retro_star")
    result.add_argument("--num-results", type=int, default=10)
    result.add_argument("--max-routes", type=int, default=25)
    result.add_argument("--reaction-model-calls", type=int, default=100)
    result.add_argument("--iterations", type=int, default=1000)
    result.add_argument("--time-limit-s", type=float, default=300.0)
    result.add_argument("--max-new-tokens", type=int, default=384)
    result.add_argument("--max-context", type=int, default=4096)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--no-4bit", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def policy_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=MODEL,
        model_revision=REVISION,
        policy_adapter=str(args.adapter),
        value_adapter="",
        value_kind="state_abc",
        pointer_head="",
        pointer_weight=0.0,
        value_weight=0.0,
        no_4bit=args.no_4bit,
        matched_v2=True,
        vnext_v2_prefix=False,
        legacy_dual_prompt=False,
        product_only_remap=True,
        reject_target_retained_finish=True,
        compact_history=args.stage == "trajectory",
        branching=1,
        early_beam=1,
        late_beam=1,
        early_depth=2,
        max_decisions=40,
        max_imports=32,
        max_new_tokens=args.max_new_tokens,
        max_context=args.max_context,
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if min(args.num_results, args.max_routes, args.reaction_model_calls, args.iterations) < 1:
        raise ValueError("all search/candidate count budgets must be positive")
    if args.time_limit_s <= 0 or args.max_new_tokens < 1 or args.max_context < 1:
        raise ValueError("time and token budgets must be positive")
    actor_args = policy_args(args)
    validate_matched_v2_args(actor_args)
    manifest = validate_v2_adapter_manifest(
        args.adapter, compact_history=actor_args.compact_history,
        expected_model=MODEL, expected_revision=REVISION,
    )
    adapter_weights = args.adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"adapter weights are missing: {adapter_weights}")
    targets = read_smiles(args.targets)
    inventory = read_smiles(args.inventory)
    if not targets or not inventory:
        raise ValueError("targets and inventory must both be nonempty")
    provenance = {
        "candidate_provider": "online_mechet_nl_reverse_et_v2",
        "policy_stage": args.stage,
        "model": MODEL,
        "model_revision": REVISION,
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": file_sha256(adapter_weights),
        "adapter_environment_revision": manifest["environment_revision"],
        "product_only_remap": True,
        "reference_precursor_visible_to_policy": False,
        "policy_decoding": "greedy_k1_stochastic_independent_episodes_k_gt_1",
        "targets_sha256": file_sha256(args.targets),
        "inventory_sha256": file_sha256(args.inventory),
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps({
            **provenance,
            "n_targets": len(targets), "n_inventory": len(inventory),
            "num_results": args.num_results,
            "reaction_model_calls": args.reaction_model_calls,
        }, indent=2))
        return 0
    runtime = Runtime(actor_args, local_rank=0)
    model = OnlineMechETBackwardReactionModel(
        runtime, actor_args, seed=args.seed, max_candidates=args.num_results,
    )
    run_planner(
        model=model, targets=targets, inventory=inventory, args=args,
        provenance=provenance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
