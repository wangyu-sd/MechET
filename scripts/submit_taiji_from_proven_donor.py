#!/usr/bin/env python3
"""Submit a frozen Taiji template with a private init command from a proven job.

The mount command is never printed or committed.  This helper deliberately
requires an explicit --submit; the default only verifies the public contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def task_detail(flag: str) -> dict:
    result = subprocess.run(
        ["taiji_client", "task_detail", flag],
        check=True,
        capture_output=True,
        text=True,
    )
    offset = result.stdout.find("{")
    if offset < 0:
        raise ValueError("Taiji donor task detail has no JSON payload")
    return json.loads(result.stdout[offset:])


def require_successful_donor_instance(flag: str) -> None:
    result = subprocess.run(
        ["taiji_client", "instance_list", flag],
        check=True,
        capture_output=True,
        text=True,
    )
    if not re.search(r"\|\s*true\s*\|\s*END\s*\|", result.stdout):
        raise ValueError("donor has no completed successful Taiji instance")


def validate_template(template: dict, donor: dict) -> str:
    flag = str(template.get("task_flag") or "")
    name = str(template.get("readable_name") or "")
    if not flag.startswith("meteor") or not name.startswith("meteor"):
        raise ValueError("both task flag and readable name must start with meteor")
    if template.get("is_elasticity") is not False or template.get("host_gpu_num") != 8:
        raise ValueError("expected one ordinary eight-GPU task")
    if template.get("host_num") != 1 or template.get("GPUName") not in {"H20", "A100"}:
        raise ValueError("template is not one supported eight-GPU host")
    if template.get("business_flag") != donor.get("common", {}).get("business_flag"):
        raise ValueError("donor and template application groups differ")
    donor_resource = donor.get("designated_resource", {})
    for key in ("GPUName", "image_full_name"):
        if template.get(key) != donor_resource.get(key):
            raise ValueError(f"donor and template {key} differ")
    if template.get("is_mount_ceph") is not True:
        raise ValueError("Ceph mount is required")
    start = str(template.get("start_cmd") or "")
    if "taiji_run_with_heartbeat.sh" not in start or "TAIJI_MIRROR_PID1_STDOUT=1" not in start:
        raise ValueError("default POD heartbeat/stdout contract is missing")
    if ">" in start or "tee " in start:
        raise ValueError("start command must not redirect default POD logs")
    init = str(donor.get("job_config", {}).get("init_cmd") or "")
    if not init or "REPLACE_WITH" in init:
        raise ValueError("proven donor has no private init command")
    return init


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--donor-task", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    template = json.loads(args.template.read_text(encoding="utf-8"))
    donor = task_detail(args.donor_task)
    require_successful_donor_instance(args.donor_task)
    init = validate_template(template, donor)
    summary = {
        "task_flag": template["task_flag"],
        "business_flag": template["business_flag"],
        "gpu": f"{template['host_gpu_num']}x{template['GPUName']}",
        "donor_task": args.donor_task,
        "donor_init_sha256": hashlib.sha256(init.encode()).hexdigest(),
        "mode": "submit" if args.submit else "preflight_only",
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not args.submit:
        return

    with tempfile.TemporaryDirectory(prefix="meteor_reliable_taiji_") as directory:
        private_config = Path(directory) / "task.json"
        private_config.write_text(
            json.dumps({**template, "init_cmd": init}, ensure_ascii=False),
            encoding="utf-8",
        )
        private_config.chmod(0o600)
        created = subprocess.run(
            ["taiji_client", "create", "--type", "task", "--simple_config", str(private_config)],
            capture_output=True,
            text=True,
        )
        if created.returncode:
            raise RuntimeError(f"Taiji create failed (exit={created.returncode}); no start attempted")
        task_detail(template["task_flag"])
        print(f"created task {template['task_flag']}", flush=True)
        started = subprocess.run(
            ["taiji_client", "start", "--task_flag", template["task_flag"]],
            capture_output=True,
            text=True,
        )
        if started.returncode:
            raise RuntimeError(f"Taiji start failed (exit={started.returncode}); inspect task before retry")
        ids = re.findall(r"\b[0-9a-f]{32}\b", started.stdout)
        print(f"start requested: task={template['task_flag']} instance={ids[-1] if ids else 'inspect instance_list'}", flush=True)


if __name__ == "__main__":
    main()
