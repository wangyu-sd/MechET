"""Render smoke jobs and delegate submission to the validated Ceph helper."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess


PRIVATE_INIT_PLACEHOLDER = "REPLACE_WITH_PRIVATE_INIT_CMD_FROM_SUCCESSFUL_TASK"


def render_job(
    template: Path, output: Path, *, task_flag: str, readable_name: str,
    repo: Path, training_config: Path, gpu_name: str,
) -> dict:
    if not task_flag.startswith("meteor"):
        raise ValueError("all Taiji jobs must begin with meteor")
    if not template.is_file() or output.exists():
        raise FileExistsError("Taiji template missing or output already exists")
    job = json.loads(template.read_text())
    if str(job.get("GPUName", "")).lower() != gpu_name.lower():
        raise ValueError("template GPU does not match approved resource profile")
    if int(job.get("host_num", 0)) != 1 or int(job.get("host_gpu_num", 0)) != 1:
        raise ValueError("scientific smoke must use a validated single-GPU template")
    if not job.get("business_flag") or not job.get("location"):
        raise ValueError("template lacks application group or location")
    job["task_flag"] = task_flag
    job["readable_name"] = readable_name
    job["init_cmd"] = PRIVATE_INIT_PLACEHOLDER
    job["is_elasticity"] = False
    job["is_resource_waiting"] = True
    command = (
        f"cd {shlex.quote(str(repo))} && "
        "export PYTHONUNBUFFERED=1 TAIJI_HEARTBEAT_SECONDS=60 && "
        "bash scripts/taiji_run_with_heartbeat.sh "
        "/root/miniconda3/envs/meteor/bin/python -u scripts/train_tool_sft.py "
        f"--config {shlex.quote(str(training_config))}"
    )
    job["start_cmd"] = "bash -lc " + shlex.quote(command)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n")
    return {"task_flag": task_flag, "readable_name": readable_name,
            "business_flag": job["business_flag"], "location": job["location"],
            "GPUName": job["GPUName"], "template": str(template),
            "rendered_template": str(output), "start_cmd": job["start_cmd"]}


def submit(repo: Path, config: Path, donor_task: str, client: Path) -> str:
    helper = repo / "scripts/submit_taiji_with_donor_init.py"
    command = ["python", str(helper), "submit", "--config", str(config),
               "--donor-task", donor_task, "--client", str(client)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode or "[error]" in result.stdout.lower():
        raise RuntimeError("validated Taiji submission helper failed: " +
                           (result.stderr or result.stdout)[-1000:])
    return result.stdout
