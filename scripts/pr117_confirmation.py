"""Authenticate the independent Reference target and execution locks."""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from scripts import pr117_controllers as controllers


RUNNER_SOURCES = (
    "pr117_confirmation.py", "run_pr117_reference.py", "pr117_controllers.py",
    "pr117_reaction_first.py", "pr117_actor_backend.py", "pr117_native_cuda.py",
)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def runner_source_hashes():
    source = Path(__file__).resolve().parent
    return {name: digest((source / name).read_bytes()) for name in RUNNER_SOURCES}


def read_pinned(ref):
    path = Path(ref["path"])
    raw = path.read_bytes()
    if digest(raw) != ref["sha256"]:
        raise ValueError(f"Registration digest changed: {path}")
    return raw


def read_target_lock(target_ref, inventory_ref):
    lock = json.loads(read_pinned(target_ref))
    inventory = json.loads(read_pinned(inventory_ref))
    if (lock["status"] != "INDEPENDENT_REFERENCE_TARGETS_LOCKED" or lock["model_calls"] != 0
            or inventory_ref["sha256"] != lock["source_inventory_sha256"]):
        raise ValueError("Confirmation does not use the independent preregistered inventory")
    rows = {(item["cohort"], item["target_id"]): item for item in inventory["candidates"]}
    rank = [(item["cohort"], item["target_id"]) for item in lock["ranked_candidate_ids"]]
    if (len(rank) != len(set(rank)) or set(rank) != set(rows)
            or len(rows) != len(inventory["candidates"]) or len(rows) != lock["eligible_inventory_size"]
            or len({item["target"] for item in rows.values()}) != len(rows)):
        raise ValueError("Confirmation inventory changed or duplicates targets")
    ordered = [rows[key] for key in rank]
    if (any(row["rank_sha256"] != meta["rank_sha256"]
            for row, meta in zip(ordered, lock["ranked_candidate_ids"], strict=True))
            or ordered[:lock["primary_targets"]] != lock["targets"]):
        raise ValueError("Confirmation target rank changed")
    return lock, ordered


def validate_registration(registration, panel, runtime_sha):
    if (registration["phase"] != "reference_confirmation"
            or registration["runtime_contract_sha256"] != runtime_sha):
        raise ValueError("Confirmation runtime changed after registration")
    if registration["runner_source_sha256"] != runner_source_hashes():
        raise ValueError("Confirmation launcher or controller changed after registration")
    lock, ordered = read_target_lock(registration["target_lock"], registration["inventory"])
    read_pinned(registration["method_spec"])

    targets = registration["targets"]
    if (type(targets) is not int or not 24 <= targets <= 256
            or str(targets) not in lock["candidate_order_prefix_strata"]
            or Counter(row["cohort"] for row in panel["targets"])
                != lock["candidate_order_prefix_strata"][str(targets)]
            or panel["targets"] != ordered[:targets]):
        raise ValueError("Confirmation must use the registered target prefix")
    if panel["stocks"] != lock["stocks"]:
        raise ValueError("Registered confirmation stock binding changed")

    arms = registration["arms"]
    primary = registration["selected_arm"]
    known = {controllers.C0, controllers.C1, controllers.C2, controllers.C4}
    if (arms != panel["arms"] or len(arms) != len(set(arms)) or set(arms) - known
            or set(arms) != {primary, controllers.C0, controllers.C4}):
        raise ValueError("Confirmation needs the selected, independent and reaction-first conditions")
    cap = registration["per_target_arm_request_cap"]
    if (type(cap) is not int or not 1 <= cap <= 128
            or panel["request_ceiling_per_target_arm"] != cap
            or registration["total_request_ceiling"] != targets * len(arms) * cap):
        raise ValueError("Confirmation request allowance changed")
    return dict(targets=targets, arms=arms, total_request_ceiling=registration["total_request_ceiling"],
                primary_arm=primary, scope="independent preregistered confirmation",
                confirmation_target_lock_sha256=registration["target_lock"]["sha256"],
                method_spec_sha256=registration["method_spec"]["sha256"],
                runner_source_sha256=registration["runner_source_sha256"])


def build_execution(target_lock, inventory, method_spec, runtime_ref, *, targets, selected_arm, cap):
    lock, ordered = read_target_lock(target_lock, inventory)
    runtime = json.loads(read_pinned(runtime_ref))
    actor = runtime["actor_lineage"]["adapter_sha256"]
    arms = list(dict.fromkeys((controllers.C0, selected_arm, controllers.C4)))
    panel = dict(seed=runtime["seed"], targets=ordered[:targets], stocks=lock["stocks"],
                 actor=dict(adapter_sha256=actor), arms=arms, request_ceiling_per_target_arm=cap)
    payload = (json.dumps(panel, indent=2) + "\n").encode()
    registration = dict(phase="reference_confirmation", target_lock=target_lock, inventory=inventory,
                        method_spec=method_spec, runtime_contract_sha256=runtime_ref["sha256"],
                        runner_source_sha256=runner_source_hashes(),
                        panel_sha256=digest(payload), actor_sha256=actor, targets=targets, arms=arms,
                        selected_arm=selected_arm, per_target_arm_request_cap=cap,
                        total_request_ceiling=targets * len(arms) * cap)
    validate_registration(registration, panel, runtime_ref["sha256"])
    return payload, registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lock", required=True, type=Path)
    parser.add_argument("--target-lock-sha256", required=True)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--runtime-contract", required=True, type=Path)
    parser.add_argument("--method-spec", required=True, type=Path)
    parser.add_argument("--method-spec-sha256", required=True)
    parser.add_argument("--selected-arm", required=True)
    parser.add_argument("--request-cap", required=True, type=int)
    parser.add_argument("--target-count", type=int, default=96)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to("/aaa/fionafyang/buddy1"):
        parser.error("Confirmation registrations must be written to shared storage")
    target = dict(path=str(args.target_lock.resolve()), sha256=args.target_lock_sha256)
    lock = json.loads(read_pinned(target))
    inventory = dict(path=str(args.inventory.resolve()), sha256=lock["source_inventory_sha256"])
    spec = dict(path=str(args.method_spec.resolve()), sha256=args.method_spec_sha256)
    runtime = dict(path=str(args.runtime_contract.resolve()), sha256=digest(args.runtime_contract.read_bytes()))
    payload, registration = build_execution(target, inventory, spec, runtime, targets=args.target_count,
                                            selected_arm=args.selected_arm, cap=args.request_cap)
    output.mkdir(exist_ok=False)
    with (output / "panel.json").open("xb") as stream:
        stream.write(payload)
    path = output / "confirmation_registration.json"
    with path.open("x") as stream:
        stream.write(json.dumps(registration, indent=2) + "\n")
    print(json.dumps(dict(panel=str(output / "panel.json"), registration=str(path),
                          confirmation_registration_sha256=digest(path.read_bytes()))))


if __name__ == "__main__":
    main()
