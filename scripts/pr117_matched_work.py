"""Pair complete Reference outcomes using verified native calls, tokens and wall time."""
import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path


def solution_cost(source, audit):
    """Recover physical first-proof costs from the independently replayed trace."""
    receipts = source["result"]["decision_receipts"]
    tokens = [len(row["generated_token_ids"]) for row in receipts]
    dispatched = [row.get("resource_error") != "PHYSICAL_CONTEXT_LIMIT" for row in receipts]
    if len(receipts) != audit["requests"] or sum(dispatched) != audit["gpu_requests"]:
        raise ValueError("Trace and audited dispatch counts disagree")
    if sum(tokens) != audit["generated_tokens"] or any(
            is_native != bool(count) for is_native, count in zip(dispatched, tokens, strict=True)):
        raise ValueError("Trace and audited generated-token counts disagree")
    seconds = source["seconds"]
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid target-arm monotonic elapsed time")
    attempt = audit["first_solution_requests"]
    prefix_tokens = audit["first_solution_tokens"]
    if not audit["solved"]:
        if attempt is not None or prefix_tokens is not None:
            raise ValueError("Unsolved arm has first-solution evidence")
        return None
    if (audit["status"] != "stock_solved" or not 1 <= attempt <= len(receipts)
            or not dispatched[attempt - 1] or sum(tokens[:attempt]) != prefix_tokens):
        raise ValueError("Inconsistent independently replayed first solution")
    return dict(native_calls=sum(dispatched[:attempt]), generated_tokens=prefix_tokens,
                terminal_seconds=seconds)


def native_call_cutoffs(cap):
    return sorted({cap, *(cut for cut in (10, 20, 30) if cut < cap)})


def paired_prefix_counts(rows, arms, cutoffs):
    outcomes = {(False, False): "neither", (True, False): "left_only",
                (False, True): "right_only", (True, True): "both"}
    comparisons = {}
    for left, right in combinations(arms, 2):
        effects = {}
        for metric, limits in cutoffs.items():
            for limit in limits:
                counts = dict(n=len(rows), neither=0, left_only=0, right_only=0, both=0)
                for evidence in rows.values():
                    a, b = evidence[left], evidence[right]
                    solved_left = a is not None and a[metric] <= limit
                    solved_right = b is not None and b[metric] <= limit
                    counts[outcomes[solved_left, solved_right]] += 1
                effects[f"{metric}<={limit}"] = counts
        comparisons[f"{left}/{right}"] = effects
    return comparisons


def matched_work(panel_path, audit_path, results):
    panel_raw = panel_path.read_bytes()
    audit_raw = audit_path.read_bytes()
    report_raw = (results / "screen_result.json").read_bytes()
    panel, audit, report = map(json.loads, (panel_raw, audit_raw, report_raw))
    panel_sha = hashlib.sha256(panel_raw).hexdigest()
    if (audit["fully_audited"] is not True or audit["final_summary_verified"] is not True
            or report["panel_sha256"] != panel_sha or audit["binding"]["panel_sha256"] != panel_sha
            or any(report.get(k) != v for k, v in audit["binding"].items())):
        raise ValueError("Expected complete, independently replayed results for the frozen panel")

    arms = panel["arms"]
    rows = {(row["cohort"], row["target_id"]): {} for row in panel["targets"]}
    if len(rows) != len(panel["targets"]) or len(arms) != len(set(arms)) or len(arms) < 2:
        raise ValueError("Expected a unique paired population with at least two arms")
    expected = {(*key, arm) for key in rows for arm in arms}
    primary = {(row["cohort"], row["target_id"], row["arm"]): row for row in report["rows"]}
    replay = {(row["cohort"], row["target_id"], row["arm"]): row for row in audit["rows"]}
    if (set(primary) != expected or set(replay) != expected
            or len(primary) != len(report["rows"]) or len(replay) != len(audit["rows"])):
        raise ValueError("Paired results do not cover each frozen target and arm exactly once")

    fields = ("cohort", "target_id", "arm", "uid", "target", "stock", "status", "solved",
              "seconds", "executor_calls", "request_attempts", "generated_tokens")
    target_rows = []
    for row in panel["targets"]:
        cohort, target = row["cohort"], row["target_id"]
        for arm in arms:
            summary = primary[cohort, target, arm]
            verified = replay[cohort, target, arm]
            name = f"{cohort}-{target}-{arm}.json"
            if Path(name).name != name:
                raise ValueError("Invalid target-arm evidence filename")
            raw = (results / "targets" / name).read_bytes()
            source = json.loads(raw)
            if (hashlib.sha256(raw).hexdigest() != verified["source_sha256"]
                    or any(source[field] != summary[field] for field in fields)
                    or any(source[field] != verified[field] for field in ("uid", "status", "solved"))
                    or summary["native_generation_requests"] != verified["gpu_requests"]
                    or (source["target"], source["stock"]) != (row["target"], row["stock"])):
                raise ValueError(f"Target-arm evidence does not match its audits: {name}")
            cost = solution_cost(source, verified)
            rows[cohort, target][arm] = cost
            target_rows.append(dict(cohort=cohort, target_id=target, arm=arm, status=source["status"],
                                    solved=source["solved"], terminal_seconds=source["seconds"],
                                    native_calls=verified["gpu_requests"], generated_tokens=verified["generated_tokens"],
                                    first_verified_solution=cost))

    cap = panel["request_ceiling_per_target_arm"]
    if any(row["native_calls"] > cap for row in target_rows):
        raise ValueError("A target arm exceeded the frozen physical call ceiling")
    cutoffs = {"native_calls": native_call_cutoffs(cap),
               "generated_tokens": (1500, 3000, 4500, 9000, 18000),
               "terminal_seconds": (120, 240, 480, 900)}
    return dict(panel_sha256=panel_sha, replay_audit_sha256=hashlib.sha256(audit_raw).hexdigest(),
                native_summary_sha256=hashlib.sha256(report_raw).hexdigest(), arms=arms, targets=len(rows),
                scope=report["scope"], physical_call_ceiling=cap, cutoffs=cutoffs,
                time_scope="Worker-monotonic arm start through verified terminal route, including queue and execution. Excludes model startup. All time and token prefixes remain under the frozen native call ceiling.",
                paired=paired_prefix_counts(rows, arms, cutoffs), rows=target_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("panel", "audit", "results", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    output = matched_work(args.panel, args.audit, args.results)
    with args.output.open("x") as f:
        f.write(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
