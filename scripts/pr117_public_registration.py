"""Bind the first full Reference benchmark condition to verified inputs."""

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts import pr117_controllers as controllers
from scripts.pr117_confirmation import comparison_conditions, digest, read_pinned, runner_source_hashes
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
    expected = set(comparison_conditions(arm))
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


def finalized_confirmation_audit(launch, evidence):
    """Use the completed finalizer output for the exact sealed Phase B comparison."""
    prepared = json.loads((launch / "prepared.json").read_bytes())
    registration_sha = prepared["confirmation_registration_sha256"]
    registration = json.loads(read_pinned(dict(path=launch / "inputs/confirmation_registration.json", sha256=registration_sha)))
    panel = json.loads(read_pinned(dict(path=launch / "inputs/panel.json", sha256=registration["panel_sha256"])))
    finalized = json.loads(read_pinned(evidence))
    expected_binding = {key: registration[key] for key in
                        ("targets", "arms", "actor_sha256", "panel_sha256", "per_target_arm_request_cap")}
    expected_binding.update(confirmation_registration_sha256=registration_sha,
                            method_spec_sha256=registration["method_spec"]["sha256"])
    count = registration["targets"] * len(registration["arms"])
    if (finalized["status"] != "COMPLETE_REPLAY_VERIFIED" or finalized["expected_arms"] != count
            or finalized["completed_arms"] != count or finalized["fully_paired_targets"] != registration["targets"]
            or any(finalized["binding"].get(k) != v for k, v in expected_binding.items())):
        raise ValueError("Public activation requires the finalized registered independent comparison")
    path = (Path(evidence["path"]).resolve().parent / "receipt_audit.json")
    ref = dict(path=str(path), sha256=finalized["artifact_sha256"]["receipt_audit.json"])
    audit = json.loads(read_pinned(ref))
    expected = {(r["cohort"], r["target_id"], arm) for r in panel["targets"] for arm in registration["arms"]}
    recorded = {(r["cohort"], r["target_id"], r["arm"]) for r in audit["rows"]}
    calls = [r["gpu_requests"] for r in audit["rows"]]
    if (audit["binding"] != finalized["binding"] or len(expected) != count or len(calls) != count or recorded != expected
            or any(not 0 <= n <= registration["per_target_arm_request_cap"] for n in calls)
            or any(finalized[k] != sum(calls) for k in ("issued_native_calls", "returned_native_calls", "audited_native_calls"))):
        raise ValueError("Finalized receipt audit must cover the registered target arms and physical calls")
    return ref


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


def paired_checkpoint(evidence, confirmation, checkpoint, actor_sha):
    pair = json.loads(read_pinned(evidence))
    baseline = json.loads(read_pinned(confirmation))
    expected = baseline["binding"]
    parent, trained = (pair[condition] for condition in ("parent", "trained"))
    parent_binding, trained_binding = parent["binding"], trained["binding"]
    count, arm = expected["targets"], expected["primary_arm"]
    if (parent["final_summary_verified"] is not True or trained["final_summary_verified"] is not True
            or pair["target_pairs"] != count or pair["same_initial_native_input_pairs"] != count
            or parent_binding != expected):
        raise ValueError("Public activation requires a completed paired actor evaluation on this confirmation")
    common = ("primary_arm", "targets", "runtime_contract_sha256", "method_spec_sha256",
              "confirmation_target_lock_sha256", "per_target_arm_request_cap", "stocks")
    if (any(trained_binding[key] != expected[key] for key in common)
            or trained_binding["arms"] != [arm]
            or trained_binding["parent_actor_sha256"] != expected["actor_sha256"]
            or trained_binding["parent_panel_sha256"] != expected["panel_sha256"]
            or trained_binding["parent_confirmation_registration_sha256"] != expected["confirmation_registration_sha256"]):
        raise ValueError("Paired actor evidence belongs to a different confirmed protocol")
    complete = json.loads(read_pinned(checkpoint))
    if (trained_binding["actor_sha256"] != actor_sha
            or trained_binding["trained_checkpoint_complete_sha256"] != checkpoint["sha256"]
            or trained_binding["optimizer_step"] != complete["optimizer_step"]):
        raise ValueError("Public activation checkpoint differs from the paired trained actor")
    locked = {(r["cohort"], r["target_id"]) for r in baseline["rows"] if r["arm"] == arm}
    reported = {(r["cohort"], r["target_id"]) for r in pair["rows"]}
    input_fields = ("first_request_id", "first_prompt_sha256")
    if (len(locked) != count or len(pair["rows"]) != count or reported != locked
            or any(not r["parent"][field] or r["parent"][field] != r["trained"][field]
                   for r in pair["rows"] for field in input_fields)):
        raise ValueError("Paired actor evaluation must cover all registered targets and matched native inputs")


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
    checkpoint = registration["trained_checkpoint"]
    paired = registration.get("paired_actor_evidence")
    if checkpoint is not None and registration.get("confirmation_finalizer") and paired is None:
        raise ValueError("Public activation of trained weights requires a completed paired actor evaluation")
    provenance = {}
    if paired is not None:
        if checkpoint is None:
            raise ValueError("Paired actor evidence requires the selected trained checkpoint")
        paired_checkpoint(paired, registration["confirmation_audit"], checkpoint, registration["actor_sha256"])
        provenance["paired_actor_evidence_sha256"] = paired["sha256"]
    return dict(scope="full official public Reference benchmark", targets=len(lock["targets"]), arms=[arm],
                primary_arm=arm, total_request_ceiling=total, public_target_lock_sha256=registration["target_lock"]["sha256"],
                method_spec_sha256=registration["method_spec"]["sha256"],
                confirmation_audit_sha256=registration["confirmation_audit"]["sha256"],
                runner_source_sha256=registration["runner_source_sha256"], **provenance)


def build_execution(source, method, confirmation, runtime_ref, *, cap, checkpoint=None, paired=None):
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
    if paired is not None:
        registration["paired_actor_evidence"] = paired
    validate_registration(registration, panel, runtime_ref["sha256"])
    return payload, registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target-lock", "method-spec", "trained-checkpoint"):
        required = name != "trained-checkpoint"
        parser.add_argument("--" + name, type=Path, required=required)
        parser.add_argument("--" + name + "-sha256", required=required)
    confirmation_input = parser.add_mutually_exclusive_group(required=True)
    confirmation_input.add_argument("--confirmation-audit", type=Path)
    confirmation_input.add_argument("--confirmation-evidence", type=Path)
    parser.add_argument("--confirmation-audit-sha256")
    parser.add_argument("--confirmation-evidence-sha256")
    parser.add_argument("--confirmation-launch", type=Path)
    parser.add_argument("--paired-actor-evidence", type=Path)
    parser.add_argument("--paired-actor-evidence-sha256")
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--request-cap", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.trained_checkpoint is None) != (args.trained_checkpoint_sha256 is None):
        parser.error("Provide both the trained complete receipt and its SHA256.")
    if (bool(args.paired_actor_evidence) != bool(args.paired_actor_evidence_sha256)
            or bool(args.trained_checkpoint) != bool(args.paired_actor_evidence)):
        parser.error("Trained weights require the completed paired actor evidence and its SHA256.")
    if (bool(args.confirmation_audit) != bool(args.confirmation_audit_sha256)
            or bool(args.confirmation_evidence) != bool(args.confirmation_evidence_sha256)
            or bool(args.confirmation_evidence) != bool(args.confirmation_launch)):
        parser.error("Supply the matching confirmation digest and the sealed launch for finalized evidence.")
    output = args.output.resolve()
    if not output.is_relative_to("/aaa/fionafyang/buddy1"):
        parser.error("Write the public registration on shared storage.")

    def ref(path, sha):
        return dict(path=str(path.resolve()), sha256=sha)

    runtime_path = args.runtime_contract
    finalized = ref(args.confirmation_evidence, args.confirmation_evidence_sha256) if args.confirmation_evidence else None
    audit = (finalized_confirmation_audit(args.confirmation_launch, finalized) if finalized else
             ref(args.confirmation_audit, args.confirmation_audit_sha256))
    panel, registration = build_execution(
        ref(args.target_lock, args.target_lock_sha256), ref(args.method_spec, args.method_spec_sha256), audit,
        ref(runtime_path, digest(runtime_path.read_bytes())), cap=args.request_cap,
        checkpoint=ref(args.trained_checkpoint, args.trained_checkpoint_sha256) if args.trained_checkpoint else None,
        paired=ref(args.paired_actor_evidence, args.paired_actor_evidence_sha256) if args.paired_actor_evidence else None)
    if finalized is not None:
        registration["confirmation_finalizer"] = finalized
    output.mkdir(exist_ok=False)
    with (output / "panel.json").open("xb") as stream:
        stream.write(panel)
    path = output / "public_registration.json"
    with path.open("x") as stream:
        stream.write(json.dumps(registration, indent=2) + "\n")
    print(json.dumps(dict(registration=str(path), registration_sha256=digest(path.read_bytes()))))


if __name__ == "__main__":
    main()
