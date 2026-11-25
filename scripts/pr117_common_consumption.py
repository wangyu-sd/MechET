"""Compare executed chemistry at each target's shared observed native-call count."""
import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from scripts.pr117_matched_work import matched_work

ARMS = ("C0_original", "C2_early_FLOW_and_backjump", "C4_completed_reaction_first")
CHEMISTRY = ("distinct_flow_transitions", "distinct_committed_edges",
             "distinct_root_precursor_sets")
METRICS = (*CHEMISTRY, "verified_stock_closures")


def summarize(rows, arms):
    per_arm = {arm: {metric: sum(row["by_arm"][arm][metric] for row in rows)
                     for metric in METRICS} for arm in arms}
    paired = {}
    for left, right in combinations(arms, 2):
        counts = {}
        for metric in METRICS:
            deltas = [row["by_arm"][left][metric] - row["by_arm"][right][metric] for row in rows]
            counts[metric] = dict(left_more=sum(delta > 0 for delta in deltas),
                                  right_more=sum(delta < 0 for delta in deltas),
                                  tie=sum(delta == 0 for delta in deltas))
        paired[f"{left}/{right}"] = counts
    return dict(targets=len(rows),
                shared_physical_calls_per_arm=sum(row["shared_physical_calls"] for row in rows),
                arms=per_arm, paired=paired)


def compare_prefixes(targets, prefixes, work, arms):
    details = []
    strata = defaultdict(list)
    for cohort, target_id in targets:
        key = (cohort, target_id)
        shared = min(work[(*key, arm)]["native_calls"] for arm in arms)
        by_arm = {}
        for arm in arms:
            arm_work = work[(*key, arm)]
            values = prefixes[(*key, arm)][str(shared)] if shared else dict.fromkeys(CHEMISTRY, 0)
            if shared and (values["native_calls"] != shared
                           or any(not 0 <= values[metric] <= shared for metric in CHEMISTRY)):
                raise ValueError(f"Invalid chemistry prefix for {key}, {arm} at {shared} calls")
            cost = arm_work["first_verified_solution"]
            by_arm[arm] = {**{metric: values[metric] for metric in CHEMISTRY},
                           "native_calls_observed": arm_work["native_calls"],
                           "first_verified_solution_call": cost["native_calls"] if cost else None,
                           "verified_stock_closures": int(cost is not None and cost["native_calls"] <= shared)}
        row = dict(cohort=cohort, target_id=target_id, shared_physical_calls=shared, by_arm=by_arm)
        details.append(row)
        strata[cohort].append(row)
    by_stratum = {cohort: summarize(rows, arms) for cohort, rows in sorted(strata.items())}
    return dict(by_stratum={**by_stratum, "all": summarize(details, arms)}, targets_detail=details)


def replay_comparison(panel_path, audit_path, prefix_path, results_path):
    matched = matched_work(panel_path, audit_path, results_path)
    if not set(ARMS).issubset(matched["arms"]):
        raise ValueError("The panel must include all three preregistered search controllers")
    base_raw = audit_path.read_bytes()
    prefix_raw = prefix_path.read_bytes()
    base, prefix = json.loads(base_raw), json.loads(prefix_raw)
    if matched["replay_audit_sha256"] != sha256(base_raw).hexdigest():
        raise ValueError("Original receipt audit changed while comparing the results")
    if {key: value for key, value in prefix.items() if key != "rows"} != {
            key: value for key, value in base.items() if key != "rows"}:
        raise ValueError("Enriched audit has different provenance from the original receipt replay")

    def work_key(row):
        return row["cohort"], row["target_id"], row["arm"]

    stripped = [{key: value for key, value in row.items() if key != "chemistry_by_native_call"}
                for row in prefix["rows"]]
    if stripped != base["rows"]:
        raise ValueError("Enriched audit rows differ from the original receipt replay")

    panel_raw = panel_path.read_bytes()
    if matched["panel_sha256"] != sha256(panel_raw).hexdigest():
        raise ValueError("Frozen panel changed while comparing the results")
    panel = json.loads(panel_raw)
    targets = [(row["cohort"], row["target_id"]) for row in panel["targets"]]
    prefixes = {work_key(row): row["chemistry_by_native_call"] for row in prefix["rows"]}
    work = {work_key(row): row for row in matched["rows"]}
    comparison = compare_prefixes(targets, prefixes, work, ARMS)
    return dict(scope=matched["scope"], arms=ARMS, targets=len(targets),
                physical_call_ceiling=matched["physical_call_ceiling"],
                panel_sha256=matched["panel_sha256"], native_summary_sha256=matched["native_summary_sha256"],
                replay_audit_sha256=matched["replay_audit_sha256"],
                chemistry_prefix_sha256=sha256(prefix_raw).hexdigest(), **comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("panel", "audit", "prefix-audit", "results", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    result = replay_comparison(args.panel, args.audit, args.prefix_audit, args.results)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
