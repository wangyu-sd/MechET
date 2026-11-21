"""Execute and independently combine batches of one full Reference registration."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.pr117_actor_transfer_score import published_bytes, replay_arm
from scripts.pr117_confirmation import read_pinned
from scripts.pr117_matched_work import solution_cost
from scripts.pr117_public_registration import target_lock
from scripts.run_pr117_reference import CEPH, bind_runtime, run_live, write_new


def partition_indexes(rows, batch_count):
    keys = {(r["cohort"], r["target_id"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Full benchmark target identities must be unique")
    if type(batch_count) is not int or not 1 <= batch_count <= len(rows):
        raise ValueError("Batch count must give nonempty parts of the complete benchmark")
    return [list(range(index, len(rows), batch_count)) for index in range(batch_count)]


def batch_plan(lock, lock_sha256, batch_count):
    return dict(public_target_lock_sha256=lock_sha256, registered_targets=len(lock["targets"]),
                target_indexes=partition_indexes(lock["targets"], batch_count))


def bind_batches(args):
    cfg, panel, stocks, full = bind_runtime(args)
    if full.get("public_registration_sha256") != args.public_registration_sha256:
        raise ValueError("Batch execution requires the genuine confirmed full public registration")
    plan = json.loads(read_pinned({"path": args.batch_plan, "sha256": args.batch_plan_sha256}))
    expected = batch_plan(panel, full["public_target_lock_sha256"], len(plan["target_indexes"]))
    if plan != expected or plan["registered_targets"] != full["targets"]:
        raise ValueError("Batches must cover the full frozen benchmark exactly once")
    return cfg, panel, stocks, full, plan


def selected_batch(panel, full, plan, plan_sha, index):
    if type(index) is not int or not 0 <= index < len(plan["target_indexes"]):
        raise ValueError("Batch index is outside the registered plan")
    targets = [panel["targets"][i] for i in plan["target_indexes"][index]]
    cap = full["per_target_arm_request_cap"]
    proof = dict(full, scope="one execution batch of the full public Reference benchmark",
                 public_batch_plan_sha256=plan_sha, public_batch_index=index,
                 public_batch_count=len(plan["target_indexes"]), full_population_targets=full["targets"],
                 targets=len(targets), total_request_ceiling=len(targets)*cap)
    # The controller's seed, target rows, stock and original panel binding all stay identical.
    return dict(panel, targets=targets), proof


def budget_counts(rows, cap):
    limits = sorted({cap, *(n for n in (10, 20, 30, 64) if n < cap)})
    return {str(limit): dict(targets=len(rows), closed=sum(r["first_solution"] is not None
            and r["first_solution"]["native_calls"] <= limit for r in rows)) for limit in limits}


def merge_verified(expected, batch_outcomes, cap, *, all_workers_verified=True, exposure=None):
    if exposure is not None and set(exposure) != expected:
        raise ValueError("Training exposure annotations must cover the full registered target population")
    outcomes = {}
    for batch in batch_outcomes:
        if set(outcomes).intersection(batch):
            raise ValueError("Completed batches duplicate a target outcome")
        if not set(batch) <= expected or any(row["native_calls"] > cap for row in batch.values()):
            raise ValueError("Batch outcome exceeds the registered target population or budget")
        if any(row["stock_closed"] != (row["first_solution"] is not None) for row in batch.values()):
            raise ValueError("Recorded closure lacks its corresponding physical first-solution evidence")
        outcomes.update(batch)
    complete = set(outcomes) == expected and all_workers_verified
    merged = dict(status="COMPLETE_REFERENCE_EVIDENCE" if complete else "PARTIAL_REFERENCE_EVIDENCE",
                  evaluated_targets=len(outcomes), registered_targets=len(expected),
                  actual_calls_in_replayed_targets=sum(r["native_calls"] for r in outcomes.values()))
    if complete:
        merged["solved_by_budget"] = budget_counts(list(outcomes.values()), cap)
        merged["solved_by_cohort_and_budget"] = {
            cohort: budget_counts([r for (c, _), r in outcomes.items() if c == cohort], cap)
            for cohort in sorted({c for c, _ in outcomes})}
        if exposure is not None:
            groups = dict(exact_train=[], train_connectivity_only=[], no_train_connectivity=[])
            independent = []
            for key, row in outcomes.items():
                annotation = exposure[key]
                if annotation["exact_train_sources"]:
                    groups["exact_train"].append(row)
                elif annotation["connectivity_train_sources"]:
                    groups["train_connectivity_only"].append(row)
                else:
                    groups["no_train_connectivity"].append(row)
                    if not annotation["reused_development"]:
                        independent.append(row)
            merged["solved_by_training_exposure_and_budget"] = {
                group: budget_counts(rows, cap) for group, rows in groups.items()}
            merged["solved_on_independent_targets_by_budget"] = budget_counts(independent, cap)
    return merged


def public_case(raw, replay):
    return dict(stock_closed=replay["solved"], first_solution=solution_cost(raw, replay), status=replay["status"],
                native_calls=replay["gpu_requests"], generated_tokens=replay["generated_tokens"],
                terminal_seconds=raw["seconds"], executor_calls=replay["executor_calls"],
                executed_flow_transitions=replay["executed_flows"], distinct_flow_transitions=replay["distinct_flow_transitions"],
                committed_reactions=replay["committed_reactions"], source_sha256=replay["source_sha256"])


def fully_published(path):
    return published_bytes(path) is not None


def merge_batches(args, cfg, panel, stocks, full, plan):
    batches, scopes = [], []
    arm, = panel["arms"]
    for index in range(len(plan["target_indexes"])):
        branch, proof = selected_batch(panel, full, plan, args.batch_plan_sha256, index)
        path = args.results_root / f"batch-{index+1:02d}"
        if not path.exists():
            scopes.append(dict(batch=index+1, status="UNSTARTED", targets=proof["targets"], observed_dispatches=0))
        elif not (path / "native_generation.jsonl").is_file() and not fully_published(path / "screen_result.json"):
            scopes.append(dict(batch=index+1, status="INCOMPLETE", targets=proof["targets"], observed_dispatches=None))
        else:
            outcomes, evidence = replay_arm(path, cfg, branch, stocks, proof, arm, measure=public_case, allow_partial=True)
            batches.append(outcomes)
            status = "REPLAY_VERIFIED" if evidence["final_summary_verified"] else "INCOMPLETE_RECEIPTS_VERIFIED"
            scopes.append(dict(batch=index+1, status=status, targets=proof["targets"],
                               evaluated_targets=len(outcomes), **evidence))
    expected = {(row["cohort"], row["target_id"]) for row in panel["targets"]}
    lock = json.loads(read_pinned({"path": args.public_registration, "sha256": args.public_registration_sha256}))
    annotations = target_lock(lock["target_lock"])["target_annotations"]
    exposure = {(cohort, key): facts for cohort, targets in annotations.items() for key, facts in targets.items()}
    merged = merge_verified(expected, batches, full["per_target_arm_request_cap"], exposure=exposure,
                            all_workers_verified=all(s["status"] == "REPLAY_VERIFIED" for s in scopes))
    rows = [dict(cohort=k[0], target_id=k[1], training_exposure=exposure[k], **v)
            for part in batches for k, v in part.items()]
    issued = sum(s["observed_dispatches"] or 0 for s in scopes)
    if issued > full["total_request_ceiling"]:
        raise ValueError("Native calls exceeded the full public registration")
    confirmed = [s for s in scopes if s["status"] == "REPLAY_VERIFIED"]
    return dict(merged, bound_to=full, batch_plan_sha256=args.batch_plan_sha256, batches=scopes, rows=rows,
                observed_native_dispatches_lower_bound=issued,
                dispatch_counts_known_for_started_batches=all(s["observed_dispatches"] is not None for s in scopes),
                actual_calls_in_verified_batches=sum(s["observed_dispatches"] for s in confirmed),
                gpu_hours_of_verified_workers=sum(s["gpu_hours"] for s in confirmed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "run", "merge"))
    for name in ("runtime-contract", "panel", "public-registration", "batch-plan"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("public-registration-sha256", "batch-plan-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--batch-index", type=int, help="1-based batch in the complete benchmark plan")
    parser.add_argument("--allocation-receipt", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    cfg, panel, stocks, full, plan = bind_batches(args)
    if args.mode == "validate":
        if args.batch_index is not None:
            _, proof = selected_batch(panel, full, plan, args.batch_plan_sha256, args.batch_index-1)
            print(json.dumps(dict(binding=proof)))
            return
        batches = [selected_batch(panel, full, plan, args.batch_plan_sha256, i)[0]["targets"]
                   for i in range(len(plan["target_indexes"]))]
        print(json.dumps(dict(binding=full, batch_sizes=[len(x) for x in batches],
                              cohort_counts=[dict(Counter(row["cohort"] for row in batch)) for batch in batches])))
        return
    if args.results_root is None or not args.results_root.resolve().is_relative_to(CEPH):
        parser.error("Use a benchmark results root on shared storage")
    if args.mode == "merge":
        if (args.write is None or not args.write.resolve().is_relative_to(CEPH)
                or args.write.resolve().is_relative_to(args.results_root.resolve())):
            parser.error("Write evidence outside the raw benchmark outputs on shared storage")
        report = merge_batches(args, cfg, panel, stocks, full, plan)
        write_new(args.write, report)
        print(json.dumps({k: report[k] for k in ("status", "evaluated_targets", "registered_targets")}))
        return
    if args.batch_index is None or args.allocation_receipt is None:
        parser.error("Live execution requires one registered batch and a separate GPU allocation receipt")
    branch, proof = selected_batch(panel, full, plan, args.batch_plan_sha256, args.batch_index-1)
    args.results_root.mkdir(exist_ok=True)
    args.output = args.results_root / f"batch-{args.batch_index:02d}"
    if args.output.exists():
        raise FileExistsError(f"This batch already has an execution directory: {args.output}")
    args.backend, args.workers, args.vllm_native_driver = "hf", 6, False
    run_live(args, cfg, branch, stocks, proof)


if __name__ == "__main__":
    main()
