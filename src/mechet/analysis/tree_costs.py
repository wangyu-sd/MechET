"""Observed per-target charged work, not inferred global GPU accounting."""
from __future__ import annotations

import math

WORK_FIELDS = ["sampled_tokens", "prompt_tokens", "raw_candidates", "node_expansions", "executor_calls",
    "execution_cache_hits", "policy_request_batches", "model_forward_batches_participated", "shared_policy_cache_hits",
    "reused_output_tokens", "unique_chemical_successors", "cpu_executor_wall_seconds", "cpu_executor_thread_seconds"]


def observed_work_metrics(entries):
    def number(value):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
    def ratio(a, b):
        return a / b if a is not None and b is not None and b > 0 else None
    rows = []
    for entry in entries:
        report = entry["report"]
        work = report.get("work", {})
        values = {key: number(work.get(key)) for key in WORK_FIELDS}
        values["wall_seconds"] = number(report.get("wall_seconds"))
        calls, cache = values["executor_calls"], values["execution_cache_hits"]
        rows.append({"target_id": entry["target_id"], "tree_id": entry.get("tree_id"),
            "report_path": entry.get("report_path"), "report_index": entry.get("report_index"),
            "report_sha256": entry.get("report_sha256"), "values": values,
            "ratios": {"execution_cache_hit_fraction": ratio(cache, calls + cache if calls is not None and cache is not None else None),
                "executor_wall_seconds_per_call": ratio(values["cpu_executor_wall_seconds"], calls),
                "executor_thread_seconds_per_call": ratio(values["cpu_executor_thread_seconds"], calls)}})
    def quantile(values, fraction):
        if not values:
            return None
        position = (len(values) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    fields = {}
    for key in [*WORK_FIELDS, "wall_seconds"]:
        values = sorted(row["values"][key] for row in rows if row["values"][key] is not None)
        fields[key] = {"observed_n": len(values), "missing_n": len(rows) - len(values),
            "zero_n": values.count(0), "median": quantile(values, .5), "p90": quantile(values, .9),
            "min": min(values) if values else None, "max": max(values) if values else None,
            "instrumentation_note": "all-zero participation field does not establish zero model work" if key == "model_forward_batches_participated" and values and not any(values) else None}
    return {"observed_targets": len(rows), "fields": fields, "per_target": rows,
        "counting_unit": "source-reported per-target charged work; shared cache reuse and batch participation are not globally additive GPU work",
        "quantile_method": "linear interpolation between sorted observed values", "global_GPU_work_inferred": False,
        "matched_compute_gain_claim": False, "unavailable": ["GPU utilization", "model-generation wall time decomposition", "work-to-first exact endpoint when no time-indexed witness is recorded"]}
