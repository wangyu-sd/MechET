#!/usr/bin/env python3
"""Submit the authorized full PR83 stages sequentially, with durable status.

The supervisor runs on the submission host, not inside a GPU POD. It submits
one job at a time and polls every five minutes. Private Taiji configuration is
captured in memory only; status files contain public task IDs and provenance.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/aaa/fionafyang/buddy1/whaleywang/MechET")
CLIENT = "/usr/local/bin/taiji_client"
STAGES = (
    ("pre", 8, "meteor_mechet_nl_v2_ckpt14000_finegrained_8a100_qy_20260919_01"),
    ("rl", 1, "meteor_mechet_nl_history_v2_stratified_smoke_1a100_qy_20260922_01"),
    ("post", 8, "meteor_mechet_nl_v2_ckpt14000_finegrained_8a100_qy_20260919_01"),
)


def read_common(raw: str) -> dict:
    start = raw.find("{")
    if start < 0:
        raise RuntimeError("Taiji returned no instance status")
    detail, _ = json.JSONDecoder().raw_decode(raw[start:])
    return dict(detail["common"])


def next_stage(current: int, common: dict) -> int:
    if common["state"] != "END":
        return current
    if common.get("is_succeed") is not True:
        raise RuntimeError(f"formal stage failed: {common.get('msg', 'unknown')}")
    return current + 1


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_stage(stage: str) -> None:
    if stage == "rl":
        root = DATA_ROOT / "outputs/agent/pr83_graph_earho_conditional_stage1_train257167_20261012"
        report = json.loads((root / "train_report.json").read_text())
        if report["algorithm"] != "graph_earho_conditional_endpoint_v4":
            raise RuntimeError("formal RL algorithm is not the corrected Graph-EARHO")
        if report["stats"]["processed"] != 257167 or report["stats"]["selected"] != 257167:
            raise RuntimeError("RL did not traverse the complete train universe")
        if sha(root / "graph_earho_policy_head.pt") != report["graph_earho_head_sha256"]:
            raise RuntimeError("completed graph head hash mismatch")
        return
    root = DATA_ROOT / "outputs/eval/pr83_formal_graph_earho_conditional_stage1_20261012"
    for split, expected in (("valid", 2890), ("test", 28967)):
        report = json.loads((root / split / f"graph_qwen_{stage}" / "summary.all-shards.json").read_text())
        if report["evaluated_strict_executable"] != expected:
            raise RuntimeError(f"incomplete formal {split} evaluation")
        if split == "test" and report.get("official_reaction_denominator") != 28971:
            raise RuntimeError("official FlowER test denominator changed")
        if not report["paired_baseline"]["baseline_contract_verified"]:
            raise RuntimeError("formal comparison has no authenticated baseline")


def persist(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--pre-config", type=Path,
                        help="use an available compatible application group for the first formal task")
    parser.add_argument("--rl-config", type=Path)
    parser.add_argument("--post-config", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.poll_seconds < 60:
        raise ValueError("formal supervisor must not poll more frequently than once a minute")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = args.state.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configs = [ROOT / "configs/taiji" / f"meteor_mechet_pr83_formal_{name}_{gpus}a100_qy_20261012_01.json"
               for name, gpus, _ in STAGES]
    if args.pre_config is not None:
        configs[0] = args.pre_config.resolve()
    if args.rl_config is not None:
        configs[1] = args.rl_config.resolve()
    if args.post_config is not None:
        configs[2] = args.post_config.resolve()
    hashes = [sha(path) for path in configs]
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state["config_sha256"] != hashes:
            raise RuntimeError("campaign config changed; do not reuse the old campaign")
    else:
        state = {"stage": 0, "config_sha256": hashes, "jobs": [], "status": "PREPARED"}
        persist(args.state, state)
    state["supervisor_pid"] = os.getpid()
    persist(args.state, state)
    try:
        while state["stage"] < len(STAGES):
            index = state["stage"]
            name, _, donor = STAGES[index]
            if len(state["jobs"]) == index:
                submitted = subprocess.run([
                    sys.executable, str(ROOT / "scripts/submit_taiji_with_donor_init.py"),
                    "submit", "--config", str(configs[index]), "--donor-task", donor,
                    "--client", CLIENT,
                ], capture_output=True, text=True)
                if submitted.returncode:
                    raise RuntimeError(f"formal {name} submission failed: {submitted.stderr[-1500:]}")
                found = re.search(r"instance_id:\s*([0-9a-f]{32})", submitted.stdout)
                if not found:
                    raise RuntimeError(f"formal {name} submission has no instance ID")
                config = json.loads(configs[index].read_text())
                state["jobs"].append({"stage": name, "task_flag": config["task_flag"],
                                      "instance_id": found.group(1)})
                state["status"] = "SUBMITTED"
                persist(args.state, state)
            job = state["jobs"][index]
            result = subprocess.run([CLIENT, "instance_detail", job["task_flag"], job["instance_id"]],
                                    capture_output=True, text=True)
            if result.returncode:
                print(json.dumps({"stage": name, "status": "STATUS_RETRY"}), flush=True)
                time.sleep(args.poll_seconds)
                continue
            common = read_common(result.stdout)
            state["status"] = common["state"]
            state["last_checked_unix"] = time.time()
            persist(args.state, state)
            print(json.dumps({"stage": name, "task_flag": job["task_flag"],
                              "instance_id": job["instance_id"], "status": common["state"]}), flush=True)
            following = next_stage(index, common)
            if following != index:
                seal_stage(name)
                state["stage"] = following
                persist(args.state, state)
                continue
            time.sleep(args.poll_seconds)
        state["status"] = "ALL_FORMAL_STAGES_COMPLETE"
        persist(args.state, state)
        print(json.dumps({"status": state["status"]}), flush=True)
    except Exception as exc:
        state["status"] = "NEEDS_REPAIR"
        state["error"] = f"{type(exc).__name__}:{exc}"
        persist(args.state, state)
        raise


if __name__ == "__main__":
    main()
