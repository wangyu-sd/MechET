#!/usr/bin/env python3
"""Matched EARHO actor update with within-anchor hard-negative balancing.

The rollout, critic training, verified replay and product-only validation are
identical to campaign round 1. Only which *negative-advantage* policy samples
receive a PPO update changes; all positive samples remain eligible.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.run_anchor_branch_rl import read_rows, run_train, write_json, write_rows
from scripts.run_earho_v2 import validate_contract
from scripts.run_natural_language_anchor_branch_rl import _sha256, run_workers
from scripts.train_python_template_rlvr import _load_yaml


def policy_likelihood(row: dict) -> float:
    values = [
        float(logp)
        for logp, mask in zip(row["old_logps"], row["loss_mask"], strict=True)
        if mask
    ]
    return sum(values) / max(len(values), 1)


def balance(rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["kind"] == "rl":
            groups[(str(row["id"]), str(row["anchor"]["state_hash"]))].append(row)
    original_positive = original_negative = selected_negative = 0
    changed: list[dict] = []
    eligible_negative_keys: set[tuple[str, str, int]] = set()
    for group_key, group in groups.items():
        positives = [row for row in group if float(row["advantage"]) > 0]
        negatives = [row for row in group if float(row["advantage"]) < 0]
        original_positive += len(positives)
        original_negative += len(negatives)
        # Penalize the likely, chemically distinct wrong successors first.
        distinct: dict[str, dict] = {}
        for row in negatives:
            fingerprint = str(row["action_fingerprint"])
            incumbent = distinct.get(fingerprint)
            if incumbent is None or policy_likelihood(row) > policy_likelihood(incumbent):
                distinct[fingerprint] = row
        ranked = sorted(
            distinct.values(),
            key=lambda row: (-policy_likelihood(row), int(row["candidate_index"])),
        )
        for row in ranked[: len(positives)]:
            eligible_negative_keys.add((*group_key, int(row["candidate_index"])))
            selected_negative += 1
    for row in rows:
        record = dict(row)
        if record["kind"] == "rl" and float(record["advantage"]) < 0:
            key = (
                str(record["id"]), str(record["anchor"]["state_hash"]),
                int(record["candidate_index"]),
            )
            if key not in eligible_negative_keys:
                record["advantage"] = 0.0
                record["update_eligible"] = False
        changed.append(record)
    return changed, {
        "groups": len(groups),
        "positive_policy_rows": original_positive,
        "original_negative_policy_rows": original_negative,
        "selected_negative_policy_rows": selected_negative,
        "verified_replay_rows": sum(row["kind"] == "verified_replay" for row in rows),
        "policy_rule": "retain_all_verified_positive_and_at_most_one_likely_distinct_negative_per_positive_per_anchor",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = _load_yaml(args.config)
    validate_contract(cfg)
    campaign = args.campaign
    source = campaign / "round01/training.jsonl"
    parent = campaign / "round00/actor_training/adapter"
    critic = campaign / "round01/successor_critic"
    monitor = campaign / "validation_monitor.jsonl"
    plan = json.loads((campaign / "plan.json").read_text())
    if _sha256(monitor) != plan["prepared_files"]["validation_monitor.jsonl"]:
        raise ValueError("frozen monitor changed")
    for adapter in (parent, critic):
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"incomplete adapter: {adapter}")
    if not (campaign / "round01/round_done.json").is_file():
        raise ValueError("source campaign round 1 is incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    training = args.output / "training.jsonl"
    manifest_path = args.output / "plan.json"
    inputs = {
        "artifact_type": "earho_v2_balanced_policy_smoke",
        "source_training_sha256": _sha256(source),
        "parent_actor_sha256": _sha256(parent / "adapter_model.safetensors"),
        "critic_sha256": _sha256(critic / "adapter_model.safetensors"),
        "monitor_sha256": _sha256(monitor),
        "seed": int(cfg["seed"]) + 1,
        "test_used": False,
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if {key: manifest.get(key) for key in inputs} != inputs:
            raise ValueError("smoke output belongs to different pinned inputs")
        if _sha256(training) != manifest["balanced_training_sha256"]:
            raise ValueError("balanced training artifact changed")
    else:
        changed, statistics = balance(read_rows(source))
        if statistics["positive_policy_rows"] == 0 or statistics["selected_negative_policy_rows"] == 0:
            raise ValueError("no balanced positive/negative policy signal")
        write_rows(training, changed)
        manifest = dict(inputs, **statistics, balanced_training_sha256=_sha256(training))
        write_json(manifest_path, manifest)
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
        return 0
    actor = run_train(
        cfg, training, parent, args.output / "actor_training", int(cfg["seed"]) + 1,
    )
    eval_cfg = dict(cfg, value_adapter_path=str(critic))
    _, result = run_workers(
        eval_cfg, monitor, actor, args.output / "validation",
        frontier=2, round_index=-1, evaluation=True,
    )
    summary = {
        **manifest,
        "actor": str(actor),
        "actor_sha256": _sha256(actor / "adapter_model.safetensors"),
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
