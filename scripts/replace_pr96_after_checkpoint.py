#!/usr/bin/env python3
"""CPU-only, single-task handoff after a complete authenticated PR96 checkpoint.

Never kills an unsaved training run, never reserves waiting GPUs, never changes
the data/objective, and never reads or prints private Taiji instance details.
"""
import argparse
from datetime import datetime, timezone
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
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def instance_rows(table):
    rows = []
    for line in table.splitlines():
        fields = [part.strip() for part in line.split("|")][1:-1]
        if len(fields) == 6 and re.fullmatch(r"[0-9a-f]{32}", fields[1], re.I):
            rows.append({"instance_id": fields[1].lower(), "state": fields[3],
                         "success": fields[2].lower() == "true"})
    return rows


def sole_source_instance(table, expected_id):
    rows = instance_rows(table)
    matches = [row for row in rows if row["instance_id"] == expected_id]
    if len(matches) != 1 or any(row["state"] != "END" and row["instance_id"] != expected_id for row in rows):
        raise ValueError("PR96_HANDOFF_EXACT_SOURCE_INSTANCE_REQUIRED")
    return matches[0]


def find_complete_anchor(model_dir, expected_identity):
    from mechet.pr96_artifacts import _verify_checkpoint_components
    from mechet.pr96_performance import identity_sha256
    candidates = sorted(Path(model_dir).glob("checkpoint-*/pr96_checkpoint_manifest.json"),
                        key=lambda p: int(p.parent.name.split("-")[-1]), reverse=True)
    if not candidates:
        return None
    checkpoint = candidates[0].parent
    metadata = json.loads(candidates[0].read_text())
    if identity_sha256(metadata["lineage"]["training_identity"]) != expected_identity:
        raise ValueError("PR96_HANDOFF_SOURCE_IDENTITY_CHANGED")
    _verify_checkpoint_components(metadata, checkpoint)
    return checkpoint


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    import yaml
    from mechet.pr96_eval_runtime import sha
    from mechet.pr96_artifacts import verify_resume_checkpoint
    from mechet.pr96_performance import make_performance_resume_authorization
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    args = p.parse_args()
    control = json.loads(args.control.read_text())
    status = Path(control["status_file"])
    status.parent.mkdir(parents=True, exist_ok=True)
    lock = status.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def emit(phase, **extra):
        value = {"phase": phase, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "pid": os.getpid(), "source_task_flag": control["source_task_flag"], **extra}
        print(json.dumps(value), flush=True)
        atomic_json(status, value)
        with status.with_suffix(".jsonl").open("a") as out:
            out.write(json.dumps(value) + "\n")
    def cli(*options):
        result = subprocess.run([control["client"], *map(str, options)], capture_output=True,
                                text=True, timeout=180)
        if result.returncode or "[error]" in result.stdout.lower():
            raise RuntimeError("PR96_HANDOFF_TAIJI_COMMAND_FAILED:" + str(options[0]))
        return result.stdout
    try:
        for path, expected in control["frozen_files"].items():
            if sha(path) != expected:
                raise ValueError("PR96_HANDOFF_DEPLOYMENT_FILE_CHANGED:" + path)
        plan = json.loads(Path(control["plan"]).read_text())
        state_path = Path(plan["configs"]["state"])
        cfg = yaml.safe_load(state_path.read_text())
        payload = json.loads(Path(control["task_config"]).read_text())
        if (cfg["pr96"]["track"] != "flower" or cfg["pr96"]["stage"] != "state"
                or payload["task_flag"] != control["replacement_task_flag"]
                or payload["task_flag"] == control["source_task_flag"]
                or not payload["task_flag"].startswith("meteor_mechet_pr96_flower_")
                or payload["host_num"] != 1 or payload["host_gpu_num"] != 8
                or payload["GPUName"] != "H20" or payload["is_elasticity"]
                or payload["business_flag"] != "AI4LSLab_KEMO_ZJK_H20"
                or payload["location"] != "" or payload["quota_type"] != ""
                or payload["image_full_name"] != "mirrors.tencent.com/whaleywang/metabo:taiji6"):
            raise ValueError("PR96_HANDOFF_TASK_SCOPE_CHANGED")
        # Fail closed if a previous operator already created this replacement.
        existing = instance_rows(cli("instance_list", control["replacement_task_flag"]))
        if existing:
            raise ValueError("PR96_HANDOFF_REPLACEMENT_ALREADY_EXISTS")
        while True:
            source = sole_source_instance(cli("instance_list", control["source_task_flag"]),
                                          control["source_instance_id"])
            anchor = find_complete_anchor(cfg["output_dir"], control["source_identity_sha256"])
            if anchor is not None:
                break
            if source["state"] == "END":
                raise ValueError("PR96_HANDOFF_SOURCE_ENDED_WITHOUT_RESUMABLE_CHECKPOINT")
            progress = Path(cfg["output_dir"]) / "progress.json"
            record = json.loads(progress.read_text()) if progress.is_file() else {}
            emit("waiting-for-complete-checkpoint-no-gpus-reserved", source_state=source["state"],
                 current_step=record.get("global_step"), total_steps=record.get("total_steps"),
                 latest_checkpoint=record.get("latest_checkpoint"))
            time.sleep(45)
        for path, expected in control["frozen_files"].items():
            if sha(path) != expected:
                raise ValueError("PR96_HANDOFF_DEPLOYMENT_CHANGED_WHILE_WAITING:" + path)
        receipt = make_performance_resume_authorization(cfg, anchor)
        authorization = state_path.parent / "performance_resume_authorization.json"
        atomic_json(authorization, receipt)
        cfg["pr96"]["performance_resume_authorization"] = {"path": str(authorization), "sha256": sha(authorization)}
        cfg["pr96"]["performance_resume_anchor"] = str(anchor)
        temporary = state_path.with_suffix(".yaml.building")
        temporary.write_text(yaml.safe_dump(cfg, sort_keys=False))
        temporary.replace(state_path)
        verify_resume_checkpoint(cfg, anchor)
        source = sole_source_instance(cli("instance_list", control["source_task_flag"]),
                                      control["source_instance_id"])
        if instance_rows(cli("instance_list", control["replacement_task_flag"])):
            raise ValueError("PR96_HANDOFF_REPLACEMENT_CREATED_WHILE_WAITING_NO_SOURCE_STOP")
        resources = cli("business_resource_list", payload["business_flag"])
        emit("complete-checkpoint-verified-before-stop", checkpoint=str(anchor),
             step=receipt["source_step"], epoch=receipt["source_epoch"],
             authorization_sha256=sha(authorization), resource_snapshot_sha256=hashlib.sha256(resources.encode()).hexdigest())
        if source["state"] != "END":
            cli("stop", "--task_flag", control["source_task_flag"])
        deadline = time.monotonic() + 600
        while sole_source_instance(cli("instance_list", control["source_task_flag"]),
                                   control["source_instance_id"])["state"] != "END":
            if time.monotonic() > deadline:
                raise TimeoutError("PR96_HANDOFF_SOURCE_DID_NOT_STOP_NO_REPLACEMENT_SUBMITTED")
            emit("waiting-for-source-stop", checkpoint=str(anchor))
            time.sleep(15)
        pipeline_lock = (Path(plan["run_dir"]) / "pipeline.lock").open("a")
        fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_resume_checkpoint(cfg, anchor)
        fcntl.flock(pipeline_lock, fcntl.LOCK_UN)
        if instance_rows(cli("instance_list", control["replacement_task_flag"])):
            raise ValueError("PR96_HANDOFF_REPLACEMENT_CREATED_BEFORE_SUBMIT_NO_DUPLICATE")
        submitted_at = datetime.now(timezone.utc).isoformat()
        result = subprocess.run([sys.executable, control["submit_helper"], "submit",
            "--config", control["task_config"], "--donor-task", control["donor_task"],
            "--client", control["client"]], capture_output=True, text=True, timeout=240)
        if result.returncode:
            raise RuntimeError("PR96_HANDOFF_SUBMISSION_FAILED_CHECKPOINT_RETAINED")
        emit("replacement-submitted-not-yet-verified-running", checkpoint=str(anchor),
             source_step=receipt["source_step"], replacement_task_flag=control["replacement_task_flag"])
        while True:
            rows = instance_rows(cli("instance_list", control["replacement_task_flag"]))
            if len(rows) != 1:
                raise ValueError("PR96_HANDOFF_EXPECTED_ONE_REPLACEMENT_INSTANCE")
            current = rows[0]
            progress = Path(cfg["output_dir"]) / "progress.json"
            record = json.loads(progress.read_text()) if progress.is_file() else {}
            fresh = record.get("timestamp", "") > submitted_at
            # Counter advance alone is not enough: old records may be ahead of
            # the anchor. The resumed callback must name this exact checkpoint.
            resumed = (record.get("latest_checkpoint") == str(anchor)
                       and record.get("global_step", 0) > receipt["source_step"])
            if current["state"] == "TRAINING_RUNNING" and resumed and fresh:
                deployed_cfg = yaml.safe_load(state_path.read_text())
                preflight_path = Path(deployed_cfg["pr96"]["preflight_manifest"])
                if sha(preflight_path) != deployed_cfg["pr96"]["preflight_manifest_sha256"]:
                    raise ValueError("PR96_HANDOFF_DEPLOYED_PREFLIGHT_HASH_CHANGED")
                preflight = json.loads(preflight_path.read_text())
                from mechet.pr96_artifacts import validate_preflight
                validate_preflight(preflight, "flower")
                for rank in preflight["ranks"]:
                    layout = rank.get("performance_layout_probe", {})
                    if (layout.get("per_device_train_batch_size") != cfg["training"]["per_device_train_batch_size"]
                            or layout.get("gradient_checkpointing") != cfg["training"]["gradient_checkpointing"]
                            or not layout.get("finite_backward") or not layout.get("native_causal_conv1d")):
                        raise ValueError("PR96_HANDOFF_DEPLOYED_LAYOUT_NOT_VERIFIED")
                probe_code = (
                    "import glob,os,json,subprocess; workers=[]; "
                    "stdout=os.readlink('/proc/1/fd/1'); stderr=os.readlink('/proc/1/fd/2')\n"
                    "for name in glob.glob('/proc/[0-9]*/cmdline'):\n"
                    " try:\n"
                    "  parts=open(name,'rb').read().decode().split('\\0'); pid=name.split('/')[2]\n"
                    f"  if '-m' not in parts and {str(state_path)!r} in parts and any(p.endswith('/train_tool_sft.py') for p in parts):\n"
                    "   workers.append(os.readlink('/proc/'+pid+'/fd/1')==stdout and os.readlink('/proc/'+pid+'/fd/2')==stderr)\n"
                    " except (OSError,UnicodeError): pass\n"
                    "names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()\n"
                    "print(json.dumps({'gpu_names':names,'worker_count':len(workers),'all_workers_inherit_pid1_default_streams':len(workers)==8 and all(workers)}),flush=True)"
                )
                probe_text = cli("exec", "--tty=false", "--stdin=false", "--stderr=true", "--host-index", "0",
                                 control["replacement_task_flag"], current["instance_id"],
                                 "/root/miniconda3/envs/meteor/bin/python", "-c", probe_code)
                probes = [json.loads(line) for line in probe_text.splitlines() if line.startswith('{"gpu_names":')]
                if (len(probes) != 1 or not probes[0]["all_workers_inherit_pid1_default_streams"]
                        or len(probes[0]["gpu_names"]) != 8 or not all("H20" in n for n in probes[0]["gpu_names"])):
                    raise ValueError("PR96_HANDOFF_POD_HARDWARE_OR_DEFAULT_STREAMS_UNVERIFIED")
                emit("resumed-optimizer-updates-verified", replacement=current,
                     checkpoint=str(anchor), current_step=record["global_step"],
                     samples_per_second=record.get("samples_per_second"),
                     deployed_preflight=str(preflight_path), pod_verification=probes[0])
                return
            if current["state"] == "END":
                raise RuntimeError("PR96_HANDOFF_REPLACEMENT_ENDED_BEFORE_VERIFIED_RESUME")
            emit("replacement-waiting-for-verified-resume", replacement=current,
                 checkpoint=str(anchor), last_recorded_step=record.get("global_step"))
            time.sleep(45)
    except Exception as error:
        emit("handoff-failed-no-other-tasks-touched", error=f"{type(error).__name__}:{error}")
        raise


if __name__ == "__main__":
    main()
