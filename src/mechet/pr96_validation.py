"""Predeclared PR96 VALID cohorts and engineering promotion gate.

No test-set tuning, candidate reranking, or learned value model lives here.
"""
from collections import defaultdict
import hashlib
import math


def _length_bucket(n):
    return str(n) if n < 3 else ("3-4" if n < 5 else "5+")


def frozen_cohorts(records, *, seed=17):
    records = list(records)
    ids = [str(r["source_id"]) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_VALID_ID")
    by_tool = defaultdict(list)
    for row in records:
        tool = row["first_tool"]
        if tool not in {"import_fragments", "apply_electron_flow"}:
            raise ValueError("INVALID_ROOT_TOOL")
        by_tool[tool].append(row)
    if len(by_tool["apply_electron_flow"]) < 8 or len(by_tool["import_fragments"]) < 120:
        raise ValueError("INSUFFICIENT_VALID_STRATA_FOR_TWO_DISJOINT_64_COHORTS")

    def ordered(rows):
        strata = defaultdict(list)
        for r in rows:
            roles = r.get("import_roles", {})
            key = (_length_bucket(int(r["flow_successors_match_source"])),
                   bool(roles.get("electron_participant")), bool(roles.get("endpoint_context")))
            strata[key].append(str(r["source_id"]))
        groups = [sorted(strata[k], key=lambda rid: hashlib.sha256(f"{seed}:{rid}".encode()).digest())
                  for k in sorted(strata)]
        result = []
        while any(groups):
            for group in groups:
                if group:
                    result.append(group.pop(0))
        return result

    flow = ordered(by_tool["apply_electron_flow"])
    imports = ordered(by_tool["import_fragments"])
    return {
        "seed": seed, "selection": "root_tool_then_length_import_role_round_robin_v1",
        "paired_ids": sorted(flow[:4] + imports[:60]),
        "independent_ids": sorted(flow[4:8] + imports[60:120]),
        "paired_root_flow": 4, "paired_root_import": 60,
        "independent_root_flow": 4, "independent_root_import": 60,
    }


def stage_one_gate(metrics):
    """Fail closed on missing results, bad denominators, or protocol problems."""
    ratios = {"parse": .99, "root_import_tool": .8, "root_import_fragment": .25,
              "tf_flow_successor": .6, "tf_finish": .8}
    rates, checks = {}, {}
    for name, threshold in ratios.items():
        value = metrics.get(name, {})
        correct, total = value.get("correct", 0), value.get("total", 0)
        valid = (isinstance(correct, int) and isinstance(total, int)
                 and 0 <= correct <= total and total > 0)
        rate = correct / total if valid else None
        rates[name] = rate
        checks[name] = rate is not None and math.isfinite(rate) and rate >= threshold
    checks["full_frozen_64"] = (metrics.get("closed_loop_completed") == 64
                                and metrics.get("closed_loop_expected") == 64)
    flow_expected = metrics.get("expected_tf_flow_decisions", 0)
    expected_totals = {"root_import_tool": 60, "root_import_fragment": 60,
                       "tf_finish": 64, "tf_flow_successor": flow_expected}
    checks["all_reference_decisions"] = (
        isinstance(flow_expected, int) and flow_expected >= 64
        and all(metrics.get(k, {}).get("total") == n for k, n in expected_totals.items())
    )
    hits = metrics.get("structural_hits", 0)
    checks["structural_hits"] = isinstance(hits, int) and 4 <= hits <= 64
    counts = metrics.get("root_predicted_tools", {})
    checks["no_root_collapse"] = (sum(counts.values()) == 64
                                  and counts.get("import_fragments", 0) > 0
                                  and counts.get("apply_electron_flow", 0) > 0)
    checks["no_truncation"] = metrics.get("truncated_calls") == 0
    checks["no_gt_future_leak"] = metrics.get("gt_future_leak_detected") is False
    return {"go": all(checks.values()), "checks": checks, "rates": rates,
            "failed": sorted(k for k, v in checks.items() if not v),
            "gate_version": "pr96_stage_one_fixed_engineering_gate_v1"}
