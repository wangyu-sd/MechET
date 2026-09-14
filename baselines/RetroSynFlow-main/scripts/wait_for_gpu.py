#!/usr/bin/env python3
"""Wait until a selected GPU is genuinely idle before starting a long run."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def gpu_state(gpu: int) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, utilization = output.split(",")
    return int(memory.strip()), int(utilization.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument(
        "--ignore-skip-marker",
        action="store_true",
        help="wait even if a prior training run left a skip-wait marker",
    )
    args = parser.parse_args()

    skip_marker = (
        Path(__file__).resolve().parents[1]
        / "runtime"
        / f"skip_gpu_wait_gpu_{args.gpu}"
    )
    if skip_marker.is_file() and not args.ignore_skip_marker:
        print(
            f"GPU {args.gpu}: skip-wait marker found at {skip_marker}; "
            "starting the queued job immediately.",
            flush=True,
        )
        return 0

    ready_count = 0
    while ready_count < args.consecutive:
        try:
            memory, utilization = gpu_state(args.gpu)
            ready = (
                memory <= args.max_memory_mib
                and utilization <= args.max_utilization
            )
            ready_count = ready_count + 1 if ready else 0
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(
                f"{now} GPU {args.gpu}: memory={memory} MiB, "
                f"utilization={utilization}%, ready={ready_count}/{args.consecutive}",
                flush=True,
            )
        except Exception as error:
            ready_count = 0
            print(f"GPU query failed: {error}", flush=True)
        if ready_count < args.consecutive:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
