"""Prepare a scheduler bundle for one durable trained-actor Reference arm."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

from scripts.pr117_confirmation import digest, read_pinned
from scripts.run_pr117_reference import CEPH, write_new


def prepare(args):
    parent, output = args.parent_launch.resolve(), args.output.resolve()
    if not (parent.is_relative_to(CEPH) and output.is_relative_to(CEPH)) or output.is_relative_to(parent):
        raise ValueError("Keep the trained and parent bundles in separate shared-storage directories")
    if output.exists():
        raise FileExistsError(output)
    source = Path(__file__).resolve().parent
    prepared = json.loads((parent / "prepared.json").read_bytes())
    manifest = read_pinned(dict(path=parent / "inputs.sha256", sha256=prepared["manifest_sha256"]))
    checkpoint = args.trained_checkpoint.resolve()
    complete = json.loads(read_pinned(dict(path=checkpoint, sha256=args.trained_checkpoint_sha256)))
    step = complete["optimizer_step"]

    with tempfile.TemporaryDirectory(prefix=".pr117-trained-", dir=output.parent) as directory:
        stage = Path(directory) / "bundle"
        stage.mkdir()
        for line in manifest.decode().splitlines():
            content_sha, relative = line.split("  ", 1)
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unexpected path in the parent bundle manifest")
            content = read_pinned(dict(path=parent / path, sha256=content_sha))
            if relative not in {"worker.sh", "inputs/deployment.json"}:
                (stage / path).parent.mkdir(parents=True, exist_ok=True)
                (stage / path).write_bytes(content)

        inputs = stage / "inputs"
        (inputs / "trained_checkpoint.path").write_text(str(checkpoint) + "\n")
        (inputs / "trained_checkpoint.sha256").write_text(args.trained_checkpoint_sha256 + "\n")
        (inputs / "parent_results.path").write_text(str(parent / "results") + "\n")
        (stage / "worker.sh").write_bytes((source / "pr117_confirmation_worker.sh").read_bytes())
        for module in ("pr117_actor_transfer", "pr117_actor_transfer_score", "pr117_public_registration",
                       "pr117_reference_source"):
            name = module + ".py"
            (stage / "code/scripts" / name).write_bytes((source / name).read_bytes())

        deployment = json.loads((parent / "inputs/deployment.json").read_bytes())
        confirmation = json.loads((inputs / "confirmation_registration.json").read_bytes())
        targets = confirmation["targets"]
        cap = confirmation["per_target_arm_request_cap"]
        task_flag = prepared["task_flag"].replace("reference96_", f"trained_reference96_u{step}_", 1)
        if task_flag == prepared["task_flag"]:
            raise ValueError("Expected the registered parent Reference confirmation task")
        commit = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
        write_new(inputs / "deployment.json", dict(deployment, task_flag=task_flag, execution_commit=commit,
                  start_command=shlex.join(["bash", str(output / "worker.sh")]),
                  arms=[confirmation["selected_arm"]], max_native_calls=targets * cap,
                  parent_task_flag=prepared["task_flag"], source_parent_manifest_sha256=digest(manifest),
                  trained_checkpoint_complete_sha256=args.trained_checkpoint_sha256))
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        text = "".join(f"{digest(path.read_bytes())}  {path.relative_to(stage)}\n" for path in files)
        (stage / "inputs.sha256").write_text(text)
        admission = subprocess.check_output(["bash", stage / "worker.sh", "--validate"])
        proof = json.loads(admission)
        if (proof["optimizer_step"] != step or proof["trained_checkpoint_complete_sha256"] != args.trained_checkpoint_sha256
                or proof["parent_confirmation_registration_sha256"] != prepared["confirmation_registration_sha256"]
                or proof["targets"] != targets or proof["per_target_arm_request_cap"] != cap
                or proof["total_request_ceiling"] != targets * cap):
            raise ValueError("Trained worker did not admit the selected checkpoint and registered comparison")
        (stage / "cpu_admission.json").write_bytes(admission)
        record = dict(status="READY_CPU_VERIFIED_AWAITING_SCHEDULED_GPU_WINDOW", optimizer_step=step,
                      task_flag=task_flag, execution_commit=commit, manifest_sha256=digest(text.encode()),
                      cpu_admission_sha256=digest(admission), parent_bundle=str(parent),
                      parent_confirmation_registration_sha256=prepared["confirmation_registration_sha256"],
                      trained_checkpoint_complete_sha256=args.trained_checkpoint_sha256,
                      trained_actor_sha256=proof["actor_sha256"], targets=targets, max_native_calls=targets * cap,
                      observed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                      gpu_jobs_submitted=0, native_calls=0)
        write_new(stage / "prepared.json", record)
        stage.rename(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-launch", "trained-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trained-checkpoint-sha256", required=True)
    print(json.dumps(prepare(parser.parse_args())))


if __name__ == "__main__":
    main()
