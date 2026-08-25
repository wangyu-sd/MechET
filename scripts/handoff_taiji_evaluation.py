#!/usr/bin/env python3
"""Safely hand an evaluation output directory from one Taiji task to another.

Candidate tasks wait behind a shared gate file.  The source task is stopped only
after one candidate owns a running pod; other candidates are stopped before the
gate is released, preventing concurrent writers from touching resume shards.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def run(client: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(client), *args], capture_output=True, text=True, check=False
    )


def state(client: Path, task: str, instance: str) -> str:
    result = run(client, "instance_detail", task, instance)
    match = re.search(r'"state"\s*:\s*"([^"]+)"', result.stdout)
    return match.group(1) if match else "UNKNOWN"


def write_status(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["checked_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--source-instance", required=True)
    parser.add_argument("--candidate", action="append", required=True,
                        help="TASK,INSTANCE")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    candidates = [tuple(item.split(",", 1)) for item in args.candidate]
    if not args.gate.exists():
        raise SystemExit(f"handoff gate does not exist: {args.gate}")

    while True:
        states = {task: state(args.client, task, instance)
                  for task, instance in candidates}
        write_status(args.status, phase="waiting_for_target", candidates=states,
                     source_state=state(args.client, args.source_task,
                                        args.source_instance))
        winner = next((item for item in candidates
                       if states[item[0]] == "TRAINING_RUNNING"), None)
        if winner:
            break
        if all(value == "END" for value in states.values()):
            write_status(args.status, phase="no_candidate_started",
                         candidates=states, source_preserved=True)
            return
        time.sleep(args.interval)

    winner_task, winner_instance = winner
    for task, instance in candidates:
        if task == winner_task:
            continue
        run(args.client, "stop", task)
        for _ in range(40):
            if state(args.client, task, instance) == "END":
                break
            time.sleep(3)

    stopped = run(args.client, "stop", args.source_task)
    if stopped.returncode != 0 or "[error]" in stopped.stdout.lower():
        write_status(args.status, phase="source_stop_failed", winner=winner_task,
                     source_preserved=True)
        return
    for _ in range(120):
        if state(args.client, args.source_task, args.source_instance) == "END":
            break
        time.sleep(3)
    else:
        write_status(args.status, phase="source_stop_timeout", winner=winner_task,
                     gate_preserved=True)
        return

    args.gate.unlink()
    write_status(args.status, phase="target_released", winner=winner_task,
                 winner_instance=winner_instance, source_stopped=True,
                 gate_released=True)


if __name__ == "__main__":
    main()
