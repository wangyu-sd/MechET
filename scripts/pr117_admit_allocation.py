"""Release the waiting worker only after observing its actual Taiji allocation."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def verified_deployment(package):
    prepared = json.loads((package / "prepared.json").read_bytes())
    manifest = (package / "inputs.sha256").read_bytes()
    deployment = (package / "inputs/deployment.json").read_bytes()
    expected = f"{sha256(deployment).hexdigest()}  inputs/deployment.json"
    if sha256(manifest).hexdigest() != prepared["manifest_sha256"] or expected not in manifest.decode().splitlines():
        raise ValueError("Launch deployment differs from the sealed bundle")
    data = json.loads(deployment)
    shape = data["shape"]
    flag = data["task_flag"]
    if (flag != prepared["task_flag"] or not re.fullmatch(r"meteor_mechet_pr117_[a-z0-9_]+", flag)
            or (shape["host_num"], shape["host_gpu_num"], shape["GPUName"]) != (1, 1, "A100")):
        raise ValueError("Expected a registered PR117 one-A100 deployment")
    return data


def check_running_instance(instance, scheduler):
    if not re.fullmatch(r"[0-9a-f]{32}", instance):
        raise ValueError("Expected the exact Taiji instance ID")
    fields = [line.split("|") for line in scheduler.splitlines()]
    states = [parts[4].strip() for parts in fields if len(parts) > 4 and parts[2].strip() == instance]
    if states != ["TRAINING_RUNNING"]:
        raise ValueError("Expected exactly one running record for the submitted instance")


def verify_worker_observation(instance, ready, scheduler, probe):
    check_running_instance(instance, scheduler)
    hostname, uuid, model = (ready[key] for key in ("hostname", "gpu_uuid", "gpu_model"))
    lines = [line.strip() for line in probe.splitlines()]
    devices = [tuple(map(str.strip, line.split(",", 1))) for line in lines if line.startswith(uuid + ",")]
    if (instance not in hostname or lines.count(hostname) != 1
            or not re.fullmatch(r"GPU-[0-9a-f-]{36}", uuid) or "A100" not in model or devices != [(uuid, model)]):
        raise ValueError("Independent GPU identity does not match the admitted worker")
    return dict(hostname=hostname, gpu_uuid=uuid, gpu_model=model)


def release(package, deployment, client, instance):
    flag = deployment["task_flag"]
    if not re.fullmatch(r"[0-9a-f]{32}", instance):
        raise ValueError("Expected the exact Taiji instance ID")
    ready = json.loads((package / "worker_ready.json").read_bytes())
    scheduler = subprocess.check_output([client, "instance_list", "--vn", "1", flag], text=True, timeout=15)
    check_running_instance(instance, scheduler)
    probe = subprocess.check_output([client, "exec", "--tty=false", "--stderr=true", flag, instance,
        "bash", "-lc", "hostname; nvidia-smi --query-gpu=uuid,name --format=csv,noheader"], text=True, timeout=40)
    observed = verify_worker_observation(instance, ready, scheduler, probe)
    identity = dict(task_flag=flag, instance_id=instance, **observed)
    evidence = dict(identity, state="TRAINING_RUNNING", observed_at=datetime.now().astimezone().isoformat(),
                    platform_query_sha256=sha256(scheduler.encode()).hexdigest(),
                    independent_device_query_sha256=sha256(probe.encode()).hexdigest())
    receipt = package / "allocation_receipt.json"
    with tempfile.TemporaryDirectory(prefix=".allocation-observation-", dir=package) as stage:
        pending = Path(stage) / "receipt.json"
        pending.write_text(json.dumps(evidence, indent=2) + "\n")
        try:
            os.link(pending, receipt)
        except FileExistsError:
            earlier = json.loads(receipt.read_bytes())
            if any(earlier.get(key) != value for key, value in identity.items()):
                raise ValueError("An allocation receipt already binds this package to a different worker")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--client", default="taiji_client")
    parser.add_argument("--instance-id", help="An already submitted one-A100 task instance")
    args = parser.parse_args()
    package = args.package.resolve()
    if not package.is_relative_to("/aaa/fionafyang/buddy1"):
        parser.error("Use a sealed package on shared storage")
    deployment = verified_deployment(package)
    if args.instance_id is None:
        print(json.dumps(dict(task_flag=deployment["task_flag"], gpu=deployment["shape"]["GPUName"],
                              location=deployment["shape"]["location"], launched=False)))
        return
    print(json.dumps(release(package, deployment, args.client, args.instance_id)))


if __name__ == "__main__":
    main()
