"""Prepare the scheduler workers for a finalized full Reference registration."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

from scripts.pr117_admit_allocation import verified_deployment
from scripts.pr117_confirmation import RUNNER_SOURCES, digest, read_pinned
from scripts.pr117_public_registration import finalized_confirmation_audit
from scripts.run_pr117_reference import CEPH, write_new

WINDOW = ("2026-11-29", "2026-12-03")
PUBLIC_SOURCES = (
    "pr117_public_batches.py", "pr117_public_registration.py", "pr117_reference_source.py",
    "pr117_actor_transfer.py", "pr117_actor_transfer_score.py", "pr117_matched_work.py",
)


def package_files(parent, registration, registration_raw, plan_raw, panel_raw):
    parent_receipt = json.loads((parent / "prepared.json").read_bytes())
    manifest = dict(line.split("  ", 1)[::-1] for line in read_pinned(dict(
        path=parent / "inputs.sha256", sha256=parent_receipt["manifest_sha256"])).decode().splitlines())
    base_names = ("inputs/runtime_contract.path", "inputs/confirmation_registration.json")
    files = {name: read_pinned(dict(path=parent / name, sha256=manifest[name])) for name in base_names}
    runtime_contract = Path(files[base_names[0]].decode().strip())
    read_pinned(dict(path=runtime_contract, sha256=registration["runtime_contract_sha256"]))
    source = Path(__file__).resolve().parent
    files.update({f"code/scripts/{name}": (source / name).read_bytes()
                  for name in (*RUNNER_SOURCES, *PUBLIC_SOURCES)})
    files.update({
        "worker.sh": (source / "pr117_confirmation_worker.sh").read_bytes(),
        "inputs/public_registration.json": registration_raw,
        "inputs/batch_plan.json": plan_raw,
        "inputs/panel.json": panel_raw,
    })
    return files


def prepare(args):
    parent, registration_path, plan_path, output = (p.resolve() for p in (
        args.parent_launch, args.public_registration, args.batch_plan, args.output))
    if (any(not path.is_relative_to(CEPH) for path in (parent, registration_path, plan_path, output))
            or output.is_relative_to(parent)):
        raise ValueError("Keep the full-benchmark scheduler package separate on shared storage")
    if output.exists():
        raise FileExistsError(output)
    base_deployment = verified_deployment(parent)
    registration_raw = read_pinned(dict(path=registration_path, sha256=args.public_registration_sha256))
    registration = json.loads(registration_raw)
    finalizer = registration.get("confirmation_finalizer")
    if (registration["phase"] != "reference_full_public" or finalizer is None
            or finalized_confirmation_audit(parent, finalizer) != registration["confirmation_audit"]):
        raise ValueError("Public workers require the finalized matching independent comparison")
    panel_raw = read_pinned(dict(path=registration_path.parent / "panel.json", sha256=registration["panel_sha256"]))
    plan_raw = read_pinned(dict(path=plan_path, sha256=args.batch_plan_sha256))
    plan = json.loads(plan_raw)
    targets = registration["targets"]
    partitions = plan["target_indexes"]
    indexes = [index for partition in partitions for index in partition]
    if (plan["public_target_lock_sha256"] != registration["target_lock"]["sha256"]
            or plan["registered_targets"] != targets or not all(partitions)
            or sorted(indexes) != list(range(targets))):
        raise ValueError("Worker batches must cover all registered targets exactly once")
    files = package_files(parent, registration, registration_raw, plan_raw, panel_raw)
    source = Path(__file__).resolve().parent
    commit = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    cap = registration["per_target_arm_request_cap"]
    if registration["total_request_ceiling"] != targets * cap:
        raise ValueError("Public registration and call ceiling disagree")
    batches = {}
    with tempfile.TemporaryDirectory(prefix=".pr117-public-", dir=output.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        for index, indices in enumerate(partitions, 1):
            name = f"batch-{index:02d}"
            launch = staging / name
            launch.mkdir()
            for path, contents in files.items():
                target = launch / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
            inputs = launch / "inputs"
            (inputs / "public_batch_index.txt").write_text(f"{index}\n")
            (inputs / "public_results_root.path").write_text(str(output / "results") + "\n")
            flag = f"meteor_mechet_pr117_reference1084_batch{index:02d}_1a100_20261129_01"
            budget = len(indices) * cap
            write_new(inputs / "deployment.json", dict(base_deployment, task_flag=flag,
                target_count=len(indices), planned_window_asia_shanghai=WINDOW, arms=registration["arms"],
                max_native_calls=budget, start_command=shlex.join(["bash", str(output / name / "worker.sh")]),
                execution_commit=commit))
            sealed = "".join(
                f"{digest(path.read_bytes())}  {path.relative_to(launch)}\n" for path in
                sorted(launch.rglob("*")) if path.is_file()).encode()
            (launch / "inputs.sha256").write_bytes(sealed)
            admission = subprocess.check_output(["bash", launch / "worker.sh", "--validate"])
            bound = json.loads(admission)["binding"]
            expected = dict(
                public_registration_sha256=args.public_registration_sha256, actor_sha256=registration["actor_sha256"],
                public_batch_plan_sha256=args.batch_plan_sha256, public_batch_index=index - 1,
                full_population_targets=targets, targets=len(indices), total_request_ceiling=budget,
            )
            if any(bound.get(key) != value for key, value in expected.items()):
                raise ValueError("A sealed worker did not admit its exact registered batch")
            (launch / "cpu_admission.json").write_bytes(admission)
            evidence = dict(
                task_flag=flag, targets=len(indices), max_native_calls=budget,
                manifest_sha256=digest(sealed), cpu_admission_sha256=digest(admission),
                gpu_jobs_submitted=0, native_calls=0,
            )
            write_new(launch / "prepared.json", evidence)
            batches[name] = evidence
        result = dict(
            status="FULL_REFERENCE_SCHEDULER_WORKERS_CPU_VERIFIED",
            observed_at=datetime.now().astimezone().isoformat(timespec="seconds"), execution_commit=commit,
            targets=targets, max_native_calls=targets * cap, planned_window_asia_shanghai=WINDOW,
            public_registration_sha256=args.public_registration_sha256, batch_plan_sha256=args.batch_plan_sha256,
            batches=batches, gpu_jobs_submitted=0, native_calls=0,
        )
        write_new(staging / "prepared.json", result)
        staging.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-launch", "public-registration", "batch-plan", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("public-registration-sha256", "batch-plan-sha256"):
        parser.add_argument(f"--{name}", required=True)
    print(json.dumps(prepare(parser.parse_args())))


if __name__ == "__main__":
    main()
