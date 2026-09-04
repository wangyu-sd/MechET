#!/usr/bin/env python3
"""Render and optionally submit a Taiji task with a proven private Ceph init.

The donor init command is recovered from a successful Taiji instance at runtime.
It is never printed or written into the repository.  The rendered task config is
stored under ``~/.taiji/rendered_configs`` with mode 0600.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIENT = REPO_ROOT / "artifacts/taiji_mount_bootstrap/taiji_client"
DEFAULT_RENDER_DIR = Path.home() / ".taiji/rendered_configs"
PLACEHOLDER = "REPLACE_WITH_PRIVATE_INIT_CMD_FROM_SUCCESSFUL_TASK"
REQUIRED_TASK_PREFIX = "meteor"
REQUIRED_LOG_WRAPPER = "scripts/taiji_run_with_heartbeat.sh"


def validate_submission_policy(config: dict[str, object], config_path: Path) -> None:
    """Enforce the permanent Taiji naming and live-log policy."""

    task_flag = str(config.get("task_flag") or "")
    readable_name = str(config.get("readable_name") or "")
    start_cmd = str(config.get("start_cmd") or "")
    if not task_flag.startswith(REQUIRED_TASK_PREFIX):
        raise ValueError(f"Taiji task_flag must start with 'meteor': {config_path}")
    if not readable_name.lower().startswith(REQUIRED_TASK_PREFIX):
        raise ValueError(f"Taiji readable_name must start with 'meteor': {config_path}")
    if REQUIRED_LOG_WRAPPER not in start_cmd:
        raise ValueError(f"Taiji start_cmd must use the heartbeat wrapper: {config_path}")
    if re.search(r"(?:^|[;&|\s])(?:[12]?>>?|&>)", start_cmd):
        raise ValueError(f"Taiji start_cmd must not redirect stdout/stderr: {config_path}")


def run(client: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(client), *args], capture_output=True, text=True, check=False
    )


def recover_init_cmd(client: Path, donor_task: str) -> str:
    listed = run(client, "instance_list", donor_task)
    if listed.returncode != 0:
        raise RuntimeError(f"could not list donor task instances: {donor_task}")
    successful_ids = []
    for line in listed.stdout.splitlines():
        if "true" not in line.lower():
            continue
        successful_ids.extend(re.findall(r"\b[0-9a-f]{32}\b", line, re.I))
    if not successful_ids:
        raise RuntimeError(f"donor task has no successful instance: {donor_task}")

    for instance_id in successful_ids:
        detail = run(client, "instance_detail", donor_task, instance_id)
        candidates: list[str] = []
        for match in re.finditer(
            r'"init_cmd"\s*:\s*"((?:\\.|[^"\\])*)"', detail.stdout
        ):
            try:
                candidates.append(json.loads('"' + match.group(1) + '"'))
            except json.JSONDecodeError:
                continue
        for command in sorted(candidates, key=len, reverse=True):
            if (
                "mount -t ceph" in command
                and "/aaa/fionafyang/buddy1" in command
                and "secret=" in command
            ):
                return command
    raise RuntimeError(f"donor task has no compatible private Ceph init: {donor_task}")


def render(config_path: Path, client: Path, donor_task: str) -> tuple[str, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_submission_policy(config, config_path)
    if config.get("init_cmd") != PLACEHOLDER:
        raise ValueError(
            f"committed config must use the private-init placeholder: {config_path}"
        )
    config["init_cmd"] = recover_init_cmd(client, donor_task)
    task_flag = str(config["task_flag"])
    DEFAULT_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(DEFAULT_RENDER_DIR, 0o700)
    output = DEFAULT_RENDER_DIR / f"{task_flag}.json"
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    os.chmod(output, 0o600)
    print(f"task_flag={task_flag}")
    print(f"init_cmd_source={donor_task}")
    print(f"rendered_config={output}")
    return task_flag, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("render", "submit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--donor-task", required=True)
    parser.add_argument("--client", type=Path, default=DEFAULT_CLIENT)
    args = parser.parse_args()

    task_flag, rendered = render(args.config, args.client, args.donor_task)
    if args.action == "render":
        return
    created = run(
        args.client, "create", "--type", "task", "--simple_config", str(rendered)
    )
    if created.returncode != 0 or "[error]" in created.stdout.lower():
        raise RuntimeError(f"Taiji task creation failed for {task_flag}")
    print(created.stdout.strip())
    started = run(args.client, "start", "--task_flag", task_flag)
    if started.returncode != 0 or "[error]" in started.stdout.lower():
        raise RuntimeError(f"Taiji task start failed for {task_flag}")
    print(started.stdout.strip())


if __name__ == "__main__":
    main()
