#!/usr/bin/env python3
"""Persist verified Taiji evaluation status instead of trusting submit state."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Job:
    name: str
    task: str
    instance: str
    output_glob: str
    expected_rows: int


def parse_job(raw: str) -> Job:
    parts = raw.split(",", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "job must be NAME,TASK,INSTANCE,OUTPUT_GLOB,EXPECTED_ROWS"
        )
    return Job(parts[0], parts[1], parts[2], parts[3], int(parts[4]))


def instance_state(client: Path, job: Job) -> tuple[str, bool, str]:
    result = subprocess.run(
        [str(client), "instance_list", job.task],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if job.instance not in line:
            continue
        fields = [field.strip() for field in line.split("|")[1:-1]]
        if len(fields) >= 4:
            return fields[3], fields[2].lower() == "true", ""
    return "UNKNOWN", False, "instance not present in instance_list"


def output_rows(pattern: str) -> tuple[int, int]:
    files = [Path(path) for path in sorted(glob.glob(pattern))]
    rows = 0
    for path in files:
        with path.open("rb") as handle:
            rows += sum(1 for line in handle if line.strip())
    return rows, len(files)


def snapshot(client: Path, jobs: list[Job]) -> dict[str, object]:
    records = []
    for job in jobs:
        state, platform_success, error = instance_state(client, job)
        rows, files = output_rows(job.output_glob)
        complete_output = rows == job.expected_rows
        verified_running = state == "TRAINING_RUNNING" and rows > 0
        verified_success = state == "END" and platform_success and complete_output
        failed = state == "END" and not verified_success
        records.append(
            {
                "name": job.name,
                "task": job.task,
                "instance": job.instance,
                "state": state,
                "platform_success": platform_success,
                "verified_running": verified_running,
                "verified_success": verified_success,
                "failed": failed,
                "output_rows": rows,
                "expected_rows": job.expected_rows,
                "output_files": files,
                "progress": rows / job.expected_rows if job.expected_rows else 0.0,
                "error": error,
            }
        )
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "jobs": records,
        "all_verified_success": all(r["verified_success"] for r in records),
        "any_failed": any(r["failed"] for r in records),
    }


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", type=parse_job, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    previous = None
    while True:
        current = snapshot(args.client, args.job)
        write_atomic(args.status_file, current)
        signature = json.dumps(current["jobs"], sort_keys=True)
        if signature != previous:
            print(json.dumps(current, ensure_ascii=False), flush=True)
            previous = signature
        if args.once or current["all_verified_success"]:
            return
        time.sleep(max(args.interval, 30))


if __name__ == "__main__":
    main()
