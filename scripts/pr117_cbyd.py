"""Run the frozen native policy on a target-only CbyD cohort."""
import argparse
import json
from pathlib import Path
import re

from scripts.pr117_confirmation import read_pinned
from scripts.run_pr117_reference import bind_runtime, run_live, sha

STOCK = "n1-n5-stock"


def split_policy_views(rows, source_by_policy, initial_source_ids):
    reverse = {source: policy for policy, source in source_by_policy.items()}
    by_id = {row["target_id"]: row for row in rows}
    selected = set(initial_source_ids)
    if (len(reverse) != len(source_by_policy) or len(by_id) != len(rows)
            or len(selected) != len(initial_source_ids) or set(by_id) != set(source_by_policy)
            or not selected < set(reverse)):
        raise ValueError("Expected unique and complete CbyD cohort bindings")
    initial = [by_id[reverse[uid]] for uid in initial_source_ids]
    remaining = [row for row in rows if source_by_policy[row["target_id"]] not in selected]
    return initial, remaining


def policy_panel(view, reference, stock, primary, cap):
    if set(view) - {"seed", "targets", "source_target_lock_sha256", "status"}:
        raise ValueError("Expected a target-only policy view")
    if type(view["seed"]) is not int or not re.fullmatch(r"[0-9a-f]{64}", view["source_target_lock_sha256"]):
        raise ValueError("Invalid frozen target-only view provenance")
    targets, ids, structures = [], set(), set()
    for row in view["targets"]:
        if (set(row) != {"target_id", "target", "seed"} or type(row["seed"]) is not int
                or not isinstance(row["target"], str) or not row["target"]
                or not re.fullmatch(r"[0-9a-f]{24}", row["target_id"])):
            raise ValueError("Expected target-only rows with opaque IDs and fixed seeds")
        if row["target"] in stock:
            raise ValueError("A registered CbyD target is already in stock")
        if row["target_id"] in ids or row["target"] in structures:
            raise ValueError("Duplicate CbyD policy ID or target")
        ids.add(row["target_id"])
        structures.add(row["target"])
        targets.append(dict(row, cohort="cbyd", stock=STOCK))
    if not targets:
        raise ValueError("Empty registered CbyD cohort")
    return dict(actor=reference["actor"], stocks={STOCK: reference["stocks"][STOCK]}, seed=view["seed"],
                arms=[primary], request_ceiling_per_target_arm=cap, targets=targets)


def bind_cbyd(args):
    cfg, reference, all_stocks, admitted = bind_runtime(args)
    from mechet.route_training_contract import chemical_identity

    view = json.loads(read_pinned({"path": args.cbyd_policy, "sha256": args.cbyd_policy_sha256}))
    stock = all_stocks[STOCK]
    cap = admitted["per_target_arm_request_cap"]
    panel = policy_panel(view, reference, stock, admitted["primary_arm"], cap)
    if any(chemical_identity(row["target"]) != row["target"] for row in panel["targets"]):
        raise ValueError("Policy target differs from the canonical executor root")
    verification = dict(
        scope="CbyD frozen target-only inference with separate literature evaluation",
        panel_sha256=sha(json.dumps(panel, sort_keys=True, separators=(",", ":")).encode()),
        runtime_contract_sha256=admitted["runtime_contract_sha256"], actor_sha256=admitted["actor_sha256"],
        method_spec_sha256=admitted["method_spec_sha256"], arms=panel["arms"], primary_arm=panel["arms"][0],
        per_target_arm_request_cap=cap, targets=len(panel["targets"]), total_request_ceiling=cap*len(panel["targets"]),
        stocks={STOCK:len(stock)}, stock_source_sha256=panel["stocks"][STOCK]["source_sha256"],
        reference_confirmation_registration_sha256=admitted["confirmation_registration_sha256"],
        cbyd_policy_sha256=args.cbyd_policy_sha256, cbyd_source_target_lock_sha256=view["source_target_lock_sha256"],
    )
    return cfg, panel, {STOCK: stock}, verification


def add_inputs(parser):
    for name in ("runtime-contract", "panel", "confirmation-registration", "cbyd-policy"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("confirmation-registration-sha256", "cbyd-policy-sha256"):
        parser.add_argument(f"--{name}", required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--allocation-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.backend = "hf"
    args.vllm_native_driver = False
    cfg, panel, stocks, verification = bind_cbyd(args)
    if args.validate:
        print(json.dumps(verification))
        return
    if args.output is None or args.allocation_receipt is None or not 1 <= args.workers <= 16:
        parser.error("Live inference requires an output directory, a GPU allocation receipt and 1–16 workers")
    run_live(args, cfg, panel, stocks, verification)


if __name__ == "__main__":
    main()
