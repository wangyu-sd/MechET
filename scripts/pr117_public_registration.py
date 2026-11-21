"""Bind the first full Reference benchmark condition to verified inputs."""

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts import pr117_controllers as controllers
from scripts.pr117_confirmation import digest, read_pinned, runner_source_hashes
from scripts.pr117_reference_source import PUBLIC_COHORTS


def source_hashes():
    root = Path(__file__).resolve().parent
    return dict(runner_source_hashes(), **{
        name: digest((root / name).read_bytes())
        for name in ("pr117_public_registration.py", "pr117_reference_source.py")
    })


def target_lock(ref):
    lock = json.loads(read_pinned(ref))
    counts = Counter(row["cohort"] for row in lock["targets"])
    identities = {(row["cohort"], row["target_id"]) for row in lock["targets"]}
    if (lock["status"] != "FULL_PUBLIC_REFERENCE_TARGETS_LOCKED" or lock["model_calls"] != 0
            or counts != PUBLIC_COHORTS or len(identities) != sum(PUBLIC_COHORTS.values())
            or lock["total_targets"] != len(identities) or set(lock["cohorts"]) != set(counts)
            or any(set(row) != {"cohort", "target_id", "target", "stock", "seed"}
                   or row["stock"] != lock["cohorts"][row["cohort"]]["stock"]
                   or row["stock"] not in lock["stocks"] for row in lock["targets"])):
        raise ValueError("Public evaluation must contain all official targets exactly once")
    return lock


def selected_method(method, confirmation):
    read_pinned(method)
    verified = json.loads(read_pinned(confirmation))
    binding = verified["binding"]
    arm = binding.get("primary_arm")
    known = {controllers.C0, controllers.C1, controllers.C2, controllers.C4}
    expected = {arm, controllers.C0, controllers.C4}
    count = binding["targets"]
    if (verified["fully_audited"] is not True or verified["final_summary_verified"] is not True
            or verified["fully_paired_targets"] != count
            or not 24 <= count <= 256 or arm not in known
            or set(binding["arms"]) != expected or len(binding["arms"]) != len(expected)
            or verified["audited"] != verified["expected"] or verified["expected"] != count * len(expected)
            or not binding.get("confirmation_registration_sha256")
            or binding["method_spec_sha256"] != method["sha256"]):
        raise ValueError("The selected method requires completed independent confirmation")
    return arm


def trained_actor(ref, runtime_sha, output_root, learner=0):
    checkpoint = Path(ref["path"]).resolve().parent
    if (Path(ref["path"]).name != "complete.json"
            or checkpoint.parent != (Path(output_root) / "checkpoints").resolve()):
        raise ValueError("A trained actor must be an original durable training checkpoint")
    complete = json.loads(read_pinned(ref))
    step = complete["optimizer_step"]
    if (type(step) is not int or step < 1 or checkpoint.name != f"round-{step:06d}"
            or complete["contract_sha256"] != runtime_sha
            or complete["round_metrics"][learner]["optimizer_updated"] is not True):
        raise ValueError("Trained actor does not bind to a completed optimizer update")
    adapter_files = {}
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        path = checkpoint / filename
        expected = complete["payload_sha256"][filename]
        read_pinned(dict(path=str(path), sha256=expected))
        adapter_files[filename] = expected
    return dict(checkpoint=str(checkpoint), optimizer_step=step,
                adapter_sha256=adapter_files["adapter_model.safetensors"], adapter_files=adapter_files)


def validate_registration(registration, panel, runtime_sha):
    if (registration["phase"] != "reference_full_public"
            or registration["runtime_contract_sha256"] != runtime_sha
            or registration["runner_source_sha256"] != source_hashes()):
        raise ValueError("Public evaluation runtime or launcher changed after registration")
    lock = target_lock(registration["target_lock"])
    if (panel["targets"] != lock["targets"] or panel["stocks"] != lock["stocks"]
            or panel["seed"] != lock["seed"] or registration["targets"] != lock["total_targets"]):
        raise ValueError("Public evaluation requires all official targets and original stock bindings")
    arm = selected_method(registration["method_spec"], registration["confirmation_audit"])
    if registration["arms"] != [arm] or registration["arms"] != panel["arms"]:
        raise ValueError("Public evaluation method does not match the confirmed primary method")
    cap = registration["per_target_arm_request_cap"]
    total = len(lock["targets"]) * cap
    if (type(cap) is not int or not 1 <= cap <= 128 or panel["request_ceiling_per_target_arm"] != cap
            or registration["total_request_ceiling"] != total):
        raise ValueError("Public evaluation request budget changed")
    if registration["actor_sha256"] != panel["actor"]["adapter_sha256"]:
        raise ValueError("Public evaluation actor changed")
    return dict(scope="full official public Reference benchmark", targets=len(lock["targets"]), arms=[arm],
                primary_arm=arm, total_request_ceiling=total, public_target_lock_sha256=registration["target_lock"]["sha256"],
                method_spec_sha256=registration["method_spec"]["sha256"],
                confirmation_audit_sha256=registration["confirmation_audit"]["sha256"],
                runner_source_sha256=registration["runner_source_sha256"])


def build_execution(source, method, confirmation, runtime_ref, *, cap, checkpoint=None):
    lock = target_lock(source)
    runtime = json.loads(read_pinned(runtime_ref))
    arm = selected_method(method, confirmation)
    actor = runtime["actor_lineage"]
    if checkpoint is not None:
        actor = trained_actor(checkpoint, runtime_ref["sha256"], runtime["output_dir"],
                              runtime.get("role_partition", {}).get("learner", 0))
    adapter_sha = actor["adapter_sha256"]
    panel = dict(seed=lock["seed"], targets=lock["targets"], stocks=lock["stocks"], arms=[arm],
                 actor=dict(adapter_sha256=adapter_sha), request_ceiling_per_target_arm=cap)
    payload = (json.dumps(panel, indent=2) + "\n").encode()
    registration = dict(phase="reference_full_public", target_lock=source, method_spec=method,
                        confirmation_audit=confirmation, runtime_contract_sha256=runtime_ref["sha256"],
                        runner_source_sha256=source_hashes(), panel_sha256=digest(payload), actor_sha256=adapter_sha,
                        trained_checkpoint=checkpoint, targets=lock["total_targets"], arms=[arm],
                        per_target_arm_request_cap=cap, total_request_ceiling=lock["total_targets"] * cap)
    validate_registration(registration, panel, runtime_ref["sha256"])
    return payload, registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target-lock", "method-spec", "confirmation-audit", "trained-checkpoint"):
        required = name != "trained-checkpoint"
        parser.add_argument("--" + name, type=Path, required=required)
        parser.add_argument("--" + name + "-sha256", required=required)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--request-cap", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.trained_checkpoint is None) != (args.trained_checkpoint_sha256 is None):
        parser.error("Provide both the trained complete receipt and its SHA256.")
    output = args.output.resolve()
    if not output.is_relative_to("/aaa/fionafyang/buddy1"):
        parser.error("Write the public registration on shared storage.")

    def ref(path, sha):
        return dict(path=str(path.resolve()), sha256=sha)

    runtime_path = args.runtime_contract
    panel, registration = build_execution(
        ref(args.target_lock, args.target_lock_sha256), ref(args.method_spec, args.method_spec_sha256),
        ref(args.confirmation_audit, args.confirmation_audit_sha256),
        ref(runtime_path, digest(runtime_path.read_bytes())), cap=args.request_cap,
        checkpoint=ref(args.trained_checkpoint, args.trained_checkpoint_sha256) if args.trained_checkpoint else None)
    output.mkdir(exist_ok=False)
    with (output / "panel.json").open("xb") as stream:
        stream.write(panel)
    path = output / "public_registration.json"
    with path.open("x") as stream:
        stream.write(json.dumps(registration, indent=2) + "\n")
    print(json.dumps(dict(registration=str(path), registration_sha256=digest(path.read_bytes()))))


if __name__ == "__main__":
    main()
