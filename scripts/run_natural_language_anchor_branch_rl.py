#!/usr/bin/env python3
"""Restartable driver for natural-language verified anchor-branch RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from run_anchor_branch_rl import (
    log,
    read_rows,
    run_train,
    summarize,
    write_json,
    write_rows,
)
from train_python_template_rlvr import _load_yaml
from mechet.anchor_branch_rl import update_frontier


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(cfg: dict) -> None:
    train = Path(cfg["train_file"])
    valid = Path(cfg["validation_file"])
    source_manifest = json.loads(Path(cfg["stable_id_manifest"]).read_text())
    if source_manifest.get("artifact_type") != "flower_strict_action_delta_trace_owned_tool_sft":
        raise ValueError("unexpected reaction source artifact")
    if source_manifest.get("observation_mode") != "action_delta_v1":
        raise ValueError("reaction source observation lineage changed")
    expected_rows = int(cfg["expected_train_rows"])
    split = source_manifest["splits"]["train"]
    if int(split["rows"]) != expected_rows:
        raise ValueError("train denominator changed")
    if str(split["sha256"]) != str(cfg["train_sha256"]):
        raise ValueError("declared train hash differs from source manifest")
    valid_split = source_manifest["splits"]["valid"]
    if str(valid_split["sha256"]) != str(cfg["validation_sha256"]):
        raise ValueError("declared validation hash differs from source manifest")
    if _sha256(train) != str(cfg["train_sha256"]):
        raise ValueError("train file hash mismatch")
    if _sha256(valid) != str(cfg["validation_sha256"]):
        raise ValueError("validation file hash mismatch")

    event_manifest = json.loads(Path(cfg["natural_language_manifest"]).read_text())
    if not event_manifest.get("training_allowed"):
        raise ValueError("natural-language event artifact is not training-enabled")
    if event_manifest.get("decision_contract") != "markov_tool_decision_v1":
        raise ValueError("natural-language decision contract changed")
    if event_manifest.get("model_visible_atom_maps") is not False:
        raise ValueError("model-visible mapping contract changed")
    event_train = event_manifest["splits"]["train"]
    if str(event_train["source_sha256"]) != str(cfg["train_sha256"]):
        raise ValueError("natural-language SFT source hash mismatch")

    adapter = Path(cfg["initial_adapter_path"])
    adapter_weights = adapter / "adapter_model.safetensors"
    if _sha256(adapter_weights) != str(cfg["initial_adapter_model_sha256"]):
        raise ValueError("natural-language SFT parent weights changed")
    adapter_manifest = json.loads(Path(cfg["initial_adapter_manifest"]).read_text())
    if adapter_manifest.get("environment_revision") != "natural_language_electron_event_v1":
        raise ValueError("parent adapter is not the natural-language event SFT")
    if adapter_manifest.get("base_model_revision") != cfg["model_revision"]:
        raise ValueError("base model revision mismatch")
    if "test_file" in cfg:
        raise ValueError("post-training config must not load test data")


def prepare(cfg: dict, output: Path) -> None:
    source = read_rows(cfg["train_file"])
    if len(source) != int(cfg["expected_train_rows"]):
        raise ValueError("source train row count changed")
    random.Random(int(cfg["seed"])).shuffle(source)
    count = int(cfg["rounds"]) * int(cfg["products_per_round"])
    if count > len(source):
        raise ValueError("not enough distinct training reactions")
    selected = source[:count]
    for round_index in range(int(cfg["rounds"])):
        begin = round_index * int(cfg["products_per_round"])
        end = begin + int(cfg["products_per_round"])
        write_rows(output / f"round{round_index:02d}/source.jsonl", selected[begin:end])
    validation = read_rows(cfg["validation_file"])
    random.Random(int(cfg["seed"])).shuffle(validation)
    monitor = validation[: int(cfg["validation_monitor_rows"])]
    write_rows(output / "validation_monitor.jsonl", monitor)
    ids = [str(row["id"]) for row in selected]
    write_json(
        output / "plan.json",
        {
            "artifact_type": "natural_language_verified_anchor_branch_rl_plan_v1",
            "algorithm": "executor_reset_same_state_successor_pooled_endpoint_reward_first_tool_call_credit",
            "source_rows": len(source),
            "selected_rows": len(selected),
            "selected_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "parent_adapter_model_sha256": cfg["initial_adapter_model_sha256"],
            "reference_suffix_visible_to_policy": False,
            "expected_precursor_visible_to_policy": False,
            "test_used": False,
            "config": cfg,
        },
    )
    log(stage="nl-anchor-prepare", selected=len(selected), validation=len(monitor))


def worker_command(cfg, data, adapter, path, rank, *, frontier, round_index, evaluation):
    rollout = cfg["rollout"]
    reward = cfg.get("reward") or {
        # Historical v1 contract: exact=1, wrong terminal=0, invalid=-penalty.
        "wrong_terminal_penalty": 0.0,
        "endpoint_similarity_weight": 0.0,
        "first_successor_progress_weight": 0.0,
        "nonexact_reward_ceiling": 0.0,
    }
    command = [
        sys.executable,
        "scripts/natural_language_anchor_branch_stage.py",
        "collect",
        "--data", str(data),
        "--output", str(path),
        "--model", cfg["model_snapshot"],
        "--adapter", str(adapter),
        "--rank", str(rank),
        "--world-size", "8",
        "--k", "2" if evaluation else str(cfg["candidates_per_product"]),
        "--seed", str((int(cfg["seed"]) + max(round_index, 0) * 1009) % (2**32)),
        "--round-index", str(round_index),
        "--frontier", str(frontier),
        "--full-episode-fraction", "1.0" if evaluation else str(cfg["curriculum"]["full_episode_fraction"]),
        "--invalid-penalty", str(cfg["invalid_penalty"]),
        "--wrong-terminal-penalty", str(reward["wrong_terminal_penalty"]),
        "--endpoint-similarity-weight", str(reward["endpoint_similarity_weight"]),
        "--first-successor-progress-weight", str(reward["first_successor_progress_weight"]),
        "--nonexact-reward-ceiling", str(reward["nonexact_reward_ceiling"]),
        "--temperature", str(rollout["temperature"]),
        "--max-new-tokens", str(rollout["max_new_tokens"]),
        "--max-context", str(rollout["max_context"]),
        "--max-decisions", str(rollout["max_decisions"]),
        "--max-imports", str(rollout["max_imports"]),
    ]
    if evaluation:
        command.extend(["--evaluation", "--full-only"])
    return command


def run_workers(cfg, data, adapter, output, *, frontier, round_index, evaluation):
    marker = output / "collection_done.json"
    shards = [output / f"rank{rank}.jsonl" for rank in range(8)]
    if marker.exists():
        if not all(path.is_file() for path in shards):
            raise ValueError("collection marker exists with missing shards")
        return shards, summarize(shards)
    output.mkdir(parents=True, exist_ok=True)
    for path in shards:
        if path.exists():
            path.rename(path.with_suffix(f".interrupted-{time.time_ns()}.jsonl"))
    runtime = os.environ.get("MECHET_ANCHOR_VLLM_RUNTIME", str(cfg["vllm_runtime"]))
    if not Path(runtime, ".mechet_vllm_runtime_complete").is_file():
        raise ValueError(f"incomplete vLLM runtime: {runtime}")
    workers = []
    try:
        for rank, path in enumerate(shards):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            env["PYTHONPATH"] = runtime + ":" + env.get("PYTHONPATH", "")
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            env["VLLM_CACHE_ROOT"] = f"/tmp/meteor-nl-anchor-vllm-{rank}"
            workers.append(
                subprocess.Popen(
                    worker_command(
                        cfg, data, adapter, path, rank,
                        frontier=frontier, round_index=round_index, evaluation=evaluation,
                    ),
                    env=env,
                )
            )
        while any(worker.poll() is None for worker in workers):
            if any(worker.poll() not in (None, 0) for worker in workers):
                raise RuntimeError("natural-language anchor worker failed")
            time.sleep(2)
        if any(worker.returncode for worker in workers):
            raise RuntimeError("natural-language anchor collection failed")
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
    summary = summarize(shards)
    public = {key: value for key, value in summary.items() if key != "group_summaries"}
    write_json(marker, {"adapter": str(adapter), "frontier": frontier, "round": round_index, "evaluation": evaluation, **public})
    log(stage="nl-anchor-collection-complete", **public)
    return shards, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = _load_yaml(Path(args.config))
    validate_contract(cfg)
    output = Path(cfg["output_dir"])
    plan = output / "plan.json"
    if plan.exists():
        if json.loads(plan.read_text())["config"] != cfg:
            raise ValueError("cannot change a prepared run in place")
    else:
        prepare(cfg, output)
    if args.prepare_only:
        return

    import torch
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    expected = str(cfg.get("expected_gpu_regex") or "A100")
    if len(names) != 8 or not all(re.search(expected, name) for name in names):
        raise ValueError(f"expected eight GPUs matching {expected}, got {names}")
    log(stage="nl-anchor-hardware", gpus=names)

    curriculum_path = output / "curriculum.json"
    curriculum = (
        json.loads(curriculum_path.read_text())
        if curriculum_path.exists()
        else {"frontier": int(cfg["curriculum"]["initial_frontier"]), "history": []}
    )
    write_json(curriculum_path, curriculum)
    adapter = Path(cfg["initial_adapter_path"])
    _, baseline = run_workers(
        cfg, output / "validation_monitor.jsonl", adapter,
        output / "baseline_validation", frontier=int(curriculum["frontier"]),
        round_index=-1, evaluation=True,
    )
    best_adapter = str(adapter)
    best_score = float(baseline["candidate_endpoint_rate"])
    log(stage="nl-anchor-baseline", endpoint_rate=best_score, adapter=str(adapter))

    for round_index in range(int(cfg["rounds"])):
        round_path = output / f"round{round_index:02d}"
        done = round_path / "round_done.json"
        if done.exists():
            record = json.loads(done.read_text())
            adapter = Path(record["adapter"])
            best_adapter = str(record["best_adapter"])
            best_score = float(record["best_validation_endpoint_rate"])
            continue
        shards, summary = run_workers(
            cfg, round_path / "source.jsonl", adapter, round_path / "rollouts",
            frontier=int(curriculum["frontier"]), round_index=round_index, evaluation=False,
        )
        training = round_path / "training.jsonl"
        write_rows(training, (row for shard in shards for row in read_rows(shard)))
        adapter = run_train(cfg, training, adapter, round_path / "training", int(cfg["seed"]) + round_index)
        _, validation = run_workers(
            cfg, output / "validation_monitor.jsonl", adapter, round_path / "validation",
            frontier=int(curriculum["frontier"]), round_index=-1, evaluation=True,
        )
        score = float(validation["candidate_endpoint_rate"])
        if score > best_score:
            best_score, best_adapter = score, str(adapter)
        next_frontier, decision = update_frontier(
            int(curriculum["frontier"]), summary["group_summaries"],
            promote_pass_rate=float(cfg["curriculum"]["promote_pass_at_k"]),
            min_effective_groups=int(cfg["curriculum"]["min_effective_groups"]),
            maximum=int(cfg["curriculum"]["maximum_frontier"]),
        )
        curriculum["frontier"] = next_frontier
        curriculum["history"].append({"round": round_index, **decision})
        write_json(curriculum_path, curriculum)
        write_json(done, {"adapter": str(adapter), "best_adapter": best_adapter, "best_validation_endpoint_rate": best_score, "validation": {key: value for key, value in validation.items() if key != "group_summaries"}, "curriculum": decision})
        log(stage="nl-anchor-round-complete", round=round_index, adapter=str(adapter), validation_endpoint_rate=score, **decision)
    write_json(output / "completed.json", {"latest_adapter": str(adapter), "best_adapter": best_adapter, "best_validation_endpoint_rate": best_score, "curriculum": curriculum, "test_used": False})
    log(stage="nl-anchor-all-rounds-complete", best_adapter=best_adapter, score=best_score)


if __name__ == "__main__":
    main()
