"""Prepare the two target-only CbyD workers from the sealed Reference launcher."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

from scripts.pr117_admit_allocation import verified_deployment
from scripts.pr117_confirmation import RUNNER_SOURCES, digest, read_pinned
from scripts.run_pr117_reference import CEPH, write_new


def prepare(args):
    parent, plan_path, output = (path.resolve() for path in (args.parent_launch, args.cbyd_plan, args.output))
    if any(not path.is_relative_to(CEPH) for path in (parent, plan_path, output)) or output.is_relative_to(parent):
        raise ValueError("Put the CbyD workers in a separate shared-storage directory")
    if output.exists():
        raise FileExistsError(output)
    base_deployment = verified_deployment(parent)
    parent_receipt = json.loads((parent / "prepared.json").read_bytes())
    plan = json.loads(read_pinned(dict(path=plan_path, sha256=args.cbyd_plan_sha256)))
    registration_sha = parent_receipt["confirmation_registration_sha256"]
    if plan["reference_confirmation_registration_sha256"] != registration_sha:
        raise ValueError("The CbyD worker must reuse the source Reference registration")
    manifest = dict(line.split("  ", 1)[::-1] for line in (parent / "inputs.sha256").read_text().splitlines())
    names = ["inputs/runtime_contract.path", "inputs/panel.json", "inputs/confirmation_registration.json",
             *(f"code/scripts/{name}" for name in RUNNER_SOURCES)]
    files = {name: read_pinned(dict(path=parent / name, sha256=manifest[name])) for name in names}
    source = Path(__file__).resolve().parent
    files["worker.sh"] = (source / "pr117_confirmation_worker.sh").read_bytes()
    files["code/scripts/pr117_cbyd.py"] = (source / "pr117_cbyd.py").read_bytes()
    commit = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    observed = datetime.now().astimezone().isoformat(timespec="seconds")
    window = plan["planned_window_asia_shanghai"]

    with tempfile.TemporaryDirectory(prefix=".pr117-cbyd-", dir=output.parent) as temporary:
        destination = Path(temporary) / "package"
        destination.mkdir()
        cohorts = {}
        for role in ("initial", "additional"):
            view = plan["views"][role]
            expected = json.loads(read_pinned(dict(path=plan_path.parent / view["admission"], sha256=view["admission_sha256"])))
            policy = read_pinned(dict(path=plan_path.parent / view["policy"], sha256=view["policy_sha256"]))
            launch = destination / role
            launch.mkdir()
            for name, contents in (files | {"inputs/cbyd_policy.json": policy}).items():
                path = launch / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            flag = f"meteor_mechet_pr117_cbyd_{role}_1a100_{window[0].replace('-', '')}_01"
            write_new(launch / "inputs/deployment.json", dict(base_deployment, task_flag=flag, target_count=view["targets"],
                planned_window_asia_shanghai=window, arms=expected["arms"], max_native_calls=view["calls_ceiling"],
                start_command=shlex.join(["bash", str(output / role / "worker.sh")]), execution_commit=commit))
            lines = [f"{digest(path.read_bytes())}  {path.relative_to(launch)}\n"
                     for path in sorted(launch.rglob("*")) if path.is_file()]
            sealed = "".join(lines).encode()
            (launch / "inputs.sha256").write_bytes(sealed)
            admission = subprocess.check_output(["bash", launch / "worker.sh", "--validate"])
            if (json.loads(admission) != expected or expected["targets"] != view["targets"]
                    or expected["total_request_ceiling"] != view["calls_ceiling"]):
                raise ValueError("Sealed CbyD worker differs from the admitted target-only view")
            (launch / "cpu_admission.json").write_bytes(admission)
            evidence = dict(task_flag=flag, targets=view["targets"], max_native_calls=view["calls_ceiling"],
                cbyd_policy_sha256=view["policy_sha256"], manifest_sha256=digest(sealed),
                cpu_admission_sha256=digest(admission), gpu_jobs_submitted=0, native_calls=0)
            write_new(launch / "prepared.json", evidence)
            cohorts[role] = evidence
        record = dict(status="TWO_CBYD_TARGET_ONLY_SCHEDULER_BUNDLES_CPU_VERIFIED", observed_at=observed,
            execution_commit=commit, cbyd_plan_sha256=args.cbyd_plan_sha256, planned_window_asia_shanghai=window,
            confirmation_registration_sha256=registration_sha, cohorts=cohorts, gpu_jobs_submitted=0, native_calls=0)
        write_new(destination / "prepared.json", record)
        destination.rename(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-launch", "cbyd-plan", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cbyd-plan-sha256", required=True)
    print(json.dumps(prepare(parser.parse_args())))


if __name__ == "__main__":
    main()
