#!/usr/bin/env python3
"""Aggregate distributed vNext vLLM benchmark reports."""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    grouped = {}
    for filename in glob.glob(str(args.input_dir / "*.rank*.json")):
        row = json.loads(Path(filename).read_text())
        grouped.setdefault(row["mode"], []).append(row)
    if not grouped:
        raise ValueError("no runtime benchmark reports")
    report = {"artifact_type": "vnext_vllm_runtime_summary_v1", "modes": {}}
    for mode, rows in sorted(grouped.items()):
        n = sum(int(r["n_states"]) for r in rows)
        max_wall = max(float(r["wall_seconds"]) for r in rows)
        if n <= 0 or max_wall <= 0:
            raise ValueError(f"invalid benchmark accounting for {mode}")
        def weighted(key):
            return sum(float(r[key]) * int(r["n_states"]) for r in rows) / n
        exact_rows = [r for r in rows if r.get("exact_action_vs_eager_prefix") is not None]
        exact = None
        if exact_rows:
            denom = sum(int(r["n_states"]) for r in exact_rows)
            exact = sum(float(r["exact_action_vs_eager_prefix"]) * int(r["n_states"]) for r in exact_rows) / denom
        report["modes"][mode] = {
            "ranks": len(rows),
            "n_states": n,
            "max_rank_wall_seconds": max_wall,
            "parallel_states_per_second": n / max_wall,
            "parse_success": weighted("parse_success"),
            "handle_valid": weighted("handle_valid"),
            "exact_action_vs_eager_prefix": exact,
        }
    eager = report["modes"].get("eager_prefix")
    if eager is None:
        raise ValueError("runtime benchmark requires eager_prefix reference")
    eager_rate = max(float(eager["parallel_states_per_second"]), 1e-12)
    for mode, values in report["modes"].items():
        values["speedup_vs_eager_prefix"] = (
            float(values["parallel_states_per_second"]) / eager_rate
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
