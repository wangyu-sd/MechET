#!/usr/bin/env python3
"""Restartable eight-GPU driver for verified anchor-branch RL."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from python_continual_stage import log, read_rows
from train_python_template_rlvr import _load_yaml, _validate_contract
from mechet.anchor_branch_rl import update_frontier


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_rows(path: Path, rows) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def summarize(paths: list[Path]) -> dict:
    records = [row for path in paths for row in read_rows(path) if row["kind"] == "rl"]
    if not records:
        raise ValueError("empty anchor rollout")
    groups = defaultdict(list)
    for row in records:
        groups[(row["id"], row["anchor"]["state_hash"])].append(row)
    group_summaries = []
    for rows in groups.values():
        group_summaries.append(
            {
                "id": rows[0]["id"],
                "horizon": rows[0]["anchor"]["horizon"],
                "is_full_episode": rows[0]["anchor"]["is_full_episode"],
                "effective": any(float(row["advantage"]) != 0.0 for row in rows),
                "endpoint_success": any(bool(row["score"]["correct"]) for row in rows),
                "unique_actions": len({row["action_fingerprint"] for row in rows}),
            }
        )
    candidates = len(records)
    return {
        "candidates": candidates,
        "groups": len(group_summaries),
        "candidate_endpoint_rate": sum(bool(row["score"]["correct"]) for row in records)
        / candidates,
        "candidate_execution_rate": sum(bool(row["score"]["formal_execute"]) for row in records)
        / candidates,
        "group_pass_at_k": sum(row["endpoint_success"] for row in group_summaries)
        / len(group_summaries),
        "effective_group_rate": sum(row["effective"] for row in group_summaries)
        / len(group_summaries),
        "full_episode_groups": sum(row["is_full_episode"] for row in group_summaries),
        "horizon_histogram": {
            str(horizon): sum(row["horizon"] == horizon for row in group_summaries)
            for horizon in sorted({row["horizon"] for row in group_summaries})
        },
        "group_summaries": group_summaries,
    }


def prepare(cfg: dict, output: Path) -> None:
    source = read_rows(cfg["train_file"])
    if len(source) != int(cfg["expected_train_rows"]):
        raise ValueError("source train row count changed")
    random.Random(cfg["seed"]).shuffle(source)
    count = int(cfg["rounds"]) * int(cfg["products_per_round"])
    if count > len(source):
        raise ValueError("not enough source reactions for distinct round allocation")
    selected = source[:count]
    for round_index in range(int(cfg["rounds"])):
        begin = round_index * int(cfg["products_per_round"])
        end = begin + int(cfg["products_per_round"])
        write_rows(output / f"round{round_index:02d}/source.jsonl", selected[begin:end])
    validation = read_rows(cfg["validation_file"])
    random.Random(cfg["seed"]).shuffle(validation)
    write_rows(
        output / "validation_monitor.jsonl",
        validation[: int(cfg["validation_monitor_rows"])],
    )
    ids = [str(row["id"]) for row in selected]
    write_json(
        output / "plan.json",
        {
            "artifact_type": "verified_anchor_branch_rl_plan_v1",
            "algorithm": "adaptive_suffix_resets_same_state_action_grouping_endpoint_verified_first_step_credit",
            "source_rows": len(source),
            "selected_rows": len(selected),
            "selected_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "initial_frontier": int(cfg["curriculum"]["initial_frontier"]),
            "reference_suffix_visible_to_rl_policy": False,
            "test_used": False,
            "config": cfg,
        },
    )
    log(stage="anchor-prepare", selected=len(selected), validation=len(validation))


def _worker_command(cfg, data, adapter, path, rank, *, frontier, round_index, evaluation):
    rollout = cfg["rollout"]
    command = [
        sys.executable,
        "scripts/anchor_branch_stage.py",
        "collect",
        "--data",
        str(data),
        "--output",
        str(path),
        "--model",
        cfg["model_snapshot"],
        "--adapter",
        str(adapter),
        "--rank",
        str(rank),
        "--world-size",
        "8",
        "--k",
        "1" if evaluation else str(cfg["candidates_per_product"]),
        "--seed",
        str((int(cfg["seed"]) + max(round_index, 0) * 1009) % (2**32)),
        "--round-index",
        str(round_index),
        "--frontier",
        str(frontier),
        "--full-episode-fraction",
        "1.0" if evaluation else str(cfg["curriculum"]["full_episode_fraction"]),
        "--replay-fraction",
        "0.0" if evaluation else str(cfg["reference_replay_fraction"]),
        "--invalid-penalty",
        str(cfg["invalid_penalty"]),
        "--temperature",
        str(rollout["temperature"]),
        "--max-new-tokens",
        str(rollout["max_new_tokens"]),
        "--max-context",
        str(rollout["max_context"]),
        "--batch-products",
        str(rollout["batch_products"]),
    ]
    if evaluation:
        command.extend(["--evaluation", "--full-only"])
    return command


def run_workers(cfg, data, adapter, output, *, frontier, round_index, evaluation):
    marker = output / "collection_done.json"
    shards = [output / f"rank{rank}.jsonl" for rank in range(8)]
    if marker.exists():
        if not all(path.is_file() for path in shards):
            raise ValueError("collection marker exists but one or more shards are missing")
        return shards, summarize(shards)
    output.mkdir(parents=True, exist_ok=True)
    for path in shards:
        if path.exists():
            path.rename(path.with_suffix(f".interrupted-{time.time_ns()}.jsonl"))
    workers = []
    vllm_runtime = os.environ.get(
        "MECHET_ANCHOR_VLLM_RUNTIME", str(cfg["vllm_runtime"])
    )
    if not Path(vllm_runtime, ".mechet_vllm_runtime_complete").is_file():
        raise ValueError(f"incomplete vLLM runtime: {vllm_runtime}")
    try:
        for rank, path in enumerate(shards):
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(rank)
            environment["PYTHONPATH"] = vllm_runtime + ":" + environment.get(
                "PYTHONPATH", ""
            )
            environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            environment["VLLM_CACHE_ROOT"] = f"/tmp/meteor-anchor-vllm-{rank}"
            workers.append(
                subprocess.Popen(
                    _worker_command(
                        cfg,
                        data,
                        adapter,
                        path,
                        rank,
                        frontier=frontier,
                        round_index=round_index,
                        evaluation=evaluation,
                    ),
                    env=environment,
                )
            )
        while any(worker.poll() is None for worker in workers):
            if any(worker.poll() not in (None, 0) for worker in workers):
                raise RuntimeError("anchor vLLM worker failed; inspect default POD logs")
            time.sleep(2)
        if any(worker.returncode for worker in workers):
            raise RuntimeError("anchor collection failed")
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
    write_json(
        marker,
        {
            "adapter": str(adapter),
            "frontier": frontier,
            "round": round_index,
            "evaluation": evaluation,
            **public,
        },
    )
    log(stage="anchor-collection-complete", **public)
    return shards, summary


def run_train(cfg, data, adapter, output, seed):
    if (output / "stage_done.json").exists():
        state = json.loads((output / "stage_done.json").read_text())
        return Path(state.get("adapter") or output / "adapter")
    rows = read_rows(data)
    has_rl_signal = any(
        row.get("kind") == "rl" and float(row.get("advantage") or 0.0) != 0.0
        for row in rows
    )
    has_replay = any(row.get("kind") != "rl" for row in rows)
    if not has_rl_signal and not has_replay:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "stage_done.json",
            {
                "adapter": str(adapter),
                "updates": 0,
                "skipped": "no_informative_anchor_advantage_or_verified_replay",
            },
        )
        log(stage="anchor-train-skip", rows=len(rows), adapter=str(adapter))
        return Path(adapter)
    command = [
        str(Path(sys.executable).with_name("torchrun")),
        "--standalone",
        "--nproc_per_node=8",
        "scripts/anchor_branch_stage.py",
        "train",
        "--data",
        str(data),
        "--output",
        str(output),
        "--model",
        cfg["model_snapshot"],
        "--adapter",
        str(adapter),
        "--reference",
        cfg["initial_adapter_path"],
        "--seed",
        str(seed),
    ]
    subprocess.run(command, check=True)
    if not (output / "adapter/adapter_model.safetensors").is_file():
        raise RuntimeError("anchor training returned without adapter weights")
    return output / "adapter"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = _load_yaml(Path(args.config))
    _validate_contract(cfg, Path(cfg["train_file"]))
    output = Path(cfg["output_dir"])
    plan = output / "plan.json"
    if plan.exists():
        if json.loads(plan.read_text())["config"] != cfg:
            raise ValueError("cannot change a prepared anchor run in place")
    else:
        prepare(cfg, output)
    if args.prepare_only:
        return

    import torch

    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    expected = str(cfg.get("expected_gpu_regex") or "A100|H20")
    if len(names) != 8 or not all(re.search(expected, name) for name in names):
        raise ValueError(f"expected eight GPUs matching {expected}, got {names}")
    log(stage="anchor-hardware", gpus=names)

    curriculum_path = output / "curriculum.json"
    if curriculum_path.exists():
        curriculum = json.loads(curriculum_path.read_text())
    else:
        curriculum = {
            "frontier": int(cfg["curriculum"]["initial_frontier"]),
            "history": [],
        }
        write_json(curriculum_path, curriculum)

    adapter = Path(cfg["initial_adapter_path"])
    _, baseline = run_workers(
        cfg,
        output / "validation_monitor.jsonl",
        adapter,
        output / "baseline_validation",
        frontier=int(curriculum["frontier"]),
        round_index=-1,
        evaluation=True,
    )
    best_adapter = str(adapter)
    best_score = float(baseline["candidate_endpoint_rate"])
    log(stage="anchor-baseline", endpoint_rate=best_score, adapter=str(adapter))
    for round_index in range(int(cfg["rounds"])):
        round_path = output / f"round{round_index:02d}"
        completed = round_path / "round_done.json"
        if completed.exists():
            record = json.loads(completed.read_text())
            adapter = Path(record["adapter"])
            best_adapter = record["best_adapter"]
            best_score = float(record["best_validation_endpoint_rate"])
            curriculum = json.loads(curriculum_path.read_text())
            continue
        shards, summary = run_workers(
            cfg,
            round_path / "source.jsonl",
            adapter,
            round_path / "rollouts",
            frontier=int(curriculum["frontier"]),
            round_index=round_index,
            evaluation=False,
        )
        training = round_path / "training.jsonl"
        write_rows(training, (row for shard in shards for row in read_rows(shard)))
        adapter = run_train(
            cfg,
            training,
            adapter,
            round_path / "training",
            int(cfg["seed"]) + round_index,
        )
        _, validation = run_workers(
            cfg,
            output / "validation_monitor.jsonl",
            adapter,
            round_path / "validation",
            frontier=int(curriculum["frontier"]),
            round_index=-1,
            evaluation=True,
        )
        validation_score = float(validation["candidate_endpoint_rate"])
        if validation_score > best_score:
            best_score = validation_score
            best_adapter = str(adapter)
        next_frontier, decision = update_frontier(
            int(curriculum["frontier"]),
            summary["group_summaries"],
            promote_pass_rate=float(cfg["curriculum"]["promote_pass_at_k"]),
            min_effective_groups=int(cfg["curriculum"]["min_effective_groups"]),
            maximum=int(cfg["curriculum"]["maximum_frontier"]),
        )
        curriculum["frontier"] = next_frontier
        curriculum["history"].append({"round": round_index, **decision})
        write_json(curriculum_path, curriculum)
        write_json(
            completed,
            {
                "adapter": str(adapter),
                "best_adapter": best_adapter,
                "best_validation_endpoint_rate": best_score,
                "validation": {
                    key: value
                    for key, value in validation.items()
                    if key != "group_summaries"
                },
                "curriculum": decision,
            },
        )
        log(
            stage="anchor-round-complete",
            round=round_index,
            adapter=str(adapter),
            validation_endpoint_rate=validation_score,
            **decision,
        )
    write_json(
        output / "completed.json",
        {
            "latest_adapter": str(adapter),
            "best_adapter": best_adapter,
            "best_validation_endpoint_rate": best_score,
            "curriculum": curriculum,
            "test_used": False,
        },
    )
    log(stage="anchor-all-rounds-complete", best_adapter=best_adapter, score=best_score)


if __name__ == "__main__":
    main()
