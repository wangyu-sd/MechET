"""Independently score parent and learned weights at the same native call prefixes."""
import argparse
import json
from pathlib import Path

from scripts.pr117_actor_transfer import bind_trained_comparison
from scripts.pr117_matched_work import solution_cost
from scripts.run_pr117_reference import CEPH, bind_runtime, sha, write_new


def native_endpoints(raw, replay):
    first_solution = solution_cost(raw, replay)
    receipts = [r for r in raw["result"]["decision_receipts"] if r["generated_token_ids"]]
    positions = {r["request_id"]: i for i, r in enumerate(receipts, 1)}
    if len(positions) != len(receipts):
        raise ValueError("Duplicate physical receipt identity")
    roots = [positions[r["receipt"]] for r in replay["commits"] if r["occurrence"] == "0"]
    first = receipts[0] if receipts else None
    first_accepted = first is not None and any(
        event["request_id"] == first["request_id"] and event.get("accepted") is True
        for attempt in raw["result"]["independent_attempts"] for event in attempt["records"])
    return dict(cohort=raw["cohort"], first_action_accepted=first_accepted,
                first_root_reaction_call=min(roots, default=None),
                first_solution=first_solution, stock_closed=replay["solved"], status=replay["status"],
                native_calls=len(receipts), generated_tokens=replay["generated_tokens"], terminal_seconds=raw["seconds"],
                distinct_flow_transitions=replay["distinct_flow_transitions"],
                first_request_id=first["request_id"] if first else None,
                first_prompt_sha256=sha(json.dumps(first["prompt_token_ids"]).encode()) if first else None)


def published_bytes(path):
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    return raw if raw.endswith(b"\n") else None


def replay_arm(results, cfg, panel, stocks, binding, arm, *, measure=native_endpoints, allow_partial=False):
    from audit_reference_receipts import audit_arm, journal_index

    admission = json.loads((results / "admission.json").read_bytes())
    raw_summary = published_bytes(results / "screen_result.json")
    if raw_summary is None and not allow_partial:
        raise ValueError("Actor replay requires a completed worker summary")
    summary = json.loads(raw_summary) if raw_summary is not None else None
    if admission["inference_backend"] != "hf" or any(admission.get(k) != v for k, v in binding.items()):
        raise ValueError("Actor run does not match its frozen inference binding")
    target_rows = {(row["cohort"], row["target_id"]): row for row in panel["targets"]}
    actual = {}
    if summary is not None:
        actual = {(r["cohort"], r["target_id"], r["arm"]): r for r in summary["rows"]}
        if any(summary.get(k) != v for k, v in binding.items()):
            raise ValueError("Final summary differs from the admitted actor and panel")
        if (len(actual) != len(summary["rows"])
                or set(actual) != {(*key, condition) for key in target_rows for condition in panel["arms"]}):
            raise ValueError("Actor run lacks the full declared target-condition population")

    requests, raw_rows = set(), {}
    for (cohort, target), case in target_rows.items():
        result_path = results / "targets" / f"{cohort}-{target}-{arm}.json"
        payload = published_bytes(result_path)
        if payload is None:
            if summary is None:
                continue
            raise ValueError("Completed actor run lacks a fully published target")
        raw = json.loads(payload)
        if ((raw["cohort"], raw["target_id"], raw["arm"], raw["target"], raw["stock"], raw["uid"])
                != (cohort, target, arm, case["target"], case["stock"], f"pr117/{cohort}/{target}/{case['seed']}/{arm}")):
            raise ValueError("Worker target does not match its preregistered identity")
        if summary is not None:
            tabular = {k: v for k, v in actual[cohort, target, arm].items() if k != "native_generation_requests"}
            if {k: v for k, v in raw.items() if k != "result"} != tabular:
                raise ValueError("Worker target differs from its final summary row")
        raw_rows[cohort, target] = dict(raw, source_sha256=sha(payload))
        ids = [r["request_id"] for r in raw["result"]["decision_receipts"]]
        if len(ids) != len(set(ids)) or requests.intersection(ids):
            raise ValueError("Independent target attempts reused a policy receipt")
        requests.update(ids)

    req, ret, rec, journal = journal_index(results / "native_generation.jsonl", requests, "hf")
    cfg = dict(cfg, behavior_version="validation000000:" + binding["actor_sha256"], seed=panel["seed"])
    cap = binding["per_target_arm_request_cap"]
    outcomes = {}
    for key, raw in raw_rows.items():
        replay = audit_arm(raw, target_rows[key], cfg, stocks[raw["stock"]], cap, (req, ret, rec))
        if summary is not None and replay["gpu_requests"] != actual[(*key, arm)]["native_generation_requests"]:
            raise ValueError("Physical actor calls disagree with the worker result")
        outcomes[key] = measure(raw, replay)

    arm_calls = sum(row["native_calls"] for row in outcomes.values())
    observed = journal.get("gpu_requests", 0)
    selected_arm_calls = journal["gpu_requests_by_arm"].get(arm, 0)
    if arm_calls > selected_arm_calls or observed > binding["total_request_ceiling"]:
        raise ValueError("Independent replay or observed dispatches exceeded the registered work")
    if summary is not None:
        total = summary["actual_native_generation_requests"]
        if (arm_calls != selected_arm_calls
                or any(total != journal.get(k, 0) for k in ("gpu_requests", "gpu_returns", "behavior_receipts"))
                or sum(row["stock_closed"] for row in outcomes.values()) != summary["solved"][arm]):
            raise ValueError("Actor physical journal or full-cohort closure count disagrees with replay")
    return outcomes, dict(summary_sha256=sha(raw_summary) if raw_summary is not None else None, binding=binding,
                          gpu_model=admission["allocation"]["gpu_model"], inference_backend="hf",
                          final_summary_verified=summary is not None, observed_dispatches=observed,
                          worker_wall_seconds=summary["single_gpu_wall_seconds"] if summary is not None else None,
                          gpu_hours=summary["gpu_hours"] if summary is not None else None)


def evaluate(args):
    from analyze_paired_reference import estimate

    parent_cfg, parent_panel, stocks, parent_proof = bind_runtime(args)
    trained_cfg, trained_panel, trained_stocks, learned_proof = bind_trained_comparison(args)
    arm = learned_proof["primary_arm"]
    parent, parent_scope = replay_arm(args.parent_results, parent_cfg, parent_panel, stocks, parent_proof, arm)
    trained, trained_scope = replay_arm(args.trained_results, trained_cfg, trained_panel, trained_stocks,
                                       learned_proof, arm)
    if len(parent) != learned_proof["targets"] or set(parent) != set(trained):
        raise ValueError("The model comparison must retain all registered target pairs")
    pairs = [dict(parent=parent[key], trained=trained[key]) for key in parent]
    endpoints = {
        "full_inventory_closed": estimate(pairs, "trained", "parent", lambda r: r["stock_closed"]),
        "first_native_action_accepted": estimate(pairs, "trained", "parent", lambda r: r["first_action_accepted"]),
    }
    for cap in (10, 30, learned_proof["per_target_arm_request_cap"]):
        endpoints[f"productive_root_by_{cap}_calls"] = estimate(
            pairs, "trained", "parent", lambda r: r["first_root_reaction_call"] is not None
            and r["first_root_reaction_call"] <= cap)
    rows = [dict(cohort=key[0], target_id=key[1], parent=parent[key], trained=trained[key]) for key in parent]
    paired_initial_input = sum(r["parent"]["first_request_id"] is not None
                               and r["parent"]["first_request_id"] == r["trained"]["first_request_id"]
                               and r["parent"]["first_prompt_sha256"] == r["trained"]["first_prompt_sha256"]
                               for r in rows)
    return dict(parent=parent_scope, trained=trained_scope, target_pairs=len(rows),
                same_initial_native_input_pairs=paired_initial_input, effects=endpoints, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-contract", "panel", "confirmation-registration", "trained-checkpoint",
                 "parent-results", "trained-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("confirmation-registration-sha256", "trained-checkpoint-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(CEPH) or any(output.is_relative_to(path.resolve()) for path in
                                               (args.parent_results, args.trained_results)):
        parser.error("Write the paired evidence to shared storage outside both raw runs")
    result = evaluate(args)
    write_new(output, result)
    print(json.dumps(result["effects"]))


if __name__ == "__main__":
    main()
