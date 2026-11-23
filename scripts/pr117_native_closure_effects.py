"""Run the sealed Reference paired estimator at native first-solution call limits."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from scripts.pr117_matched_work import matched_work, native_call_cutoffs


def measure(rows, arms, comparisons, cutoffs, estimate):
    targets = defaultdict(dict)
    for row in rows:
        key = (row["cohort"], row["target_id"])
        arm = row["arm"]
        if arm in targets[key]:
            raise ValueError(f"Repeated target and arm: {key}, {arm}")
        targets[key][arm] = row
    if not targets or any(set(target) != set(arms) for target in targets.values()):
        raise ValueError("Keep every registered target and arm in the paired analysis")

    effects = {}
    for left, right in comparisons:
        limits = {}
        for cut in cutoffs:
            def solved(row):
                cost = row["first_verified_solution"]
                return cost is not None and cost["native_calls"] <= cut

            limits[f"native_calls<={cut}"] = estimate(list(targets.values()), left, right, solved)
        effects[f"{left}/{right}"] = limits
    return effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("panel", "audit", "results", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--estimator-sha256", required=True,
                        help="SHA-256 of analyze_paired_reference.py from the sealed bundle on PYTHONPATH.")
    args = parser.parse_args()

    import analyze_paired_reference as estimator

    estimator_sha = sha256(Path(estimator.__file__).read_bytes()).hexdigest()
    if estimator_sha != args.estimator_sha256:
        parser.error("The analysis module differs from the pinned paired estimator")

    verified = matched_work(args.panel, args.audit, args.results)
    # These three comparisons use the controller selection fixed before confirmation.
    comparisons = estimator.confirmation_comparisons(estimator.C0)
    cap = verified["physical_call_ceiling"]
    limits = native_call_cutoffs(cap)
    effects = measure(verified["rows"], verified["arms"], comparisons, limits, estimator.estimate)
    result = {key: verified[key] for key in (
        "scope", "targets", "panel_sha256", "replay_audit_sha256", "native_summary_sha256",
        "physical_call_ceiling")}
    result.update(analysis_module_sha256=estimator_sha, cutoffs=limits, paired_effects=effects,
                  estimand="Left minus right in percent of registered targets whose verified route "
                           "reaches stock by the observed native call limit. Each target appears once "
                           "in a pair. Unsolved and resource-limited targets remain in the denominator.")
    with args.output.open("x") as file:
        file.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
