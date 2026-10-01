#!/usr/bin/env python3
"""Paired summary for matched vNext product-start search arms."""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path
from typing import Any


def read_glob(pattern: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    for filename in files:
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = str(row["source_id"])
                if rid in rows:
                    raise ValueError(f"duplicate source_id {rid} in {pattern}")
                rows[rid] = row
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_delta(
    baseline: list[int], pointer: list[int], *, seed: int, draws: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(baseline)
    deltas = []
    paired = [p - b for b, p in zip(baseline, pointer, strict=True)]
    for _ in range(draws):
        deltas.append(sum(paired[rng.randrange(n)] for _ in range(n)) / n)
    deltas.sort()
    lo = deltas[int(0.025 * (draws - 1))]
    hi = deltas[int(0.975 * (draws - 1))]
    return lo, hi


def exact_mcnemar(baseline: list[int], pointer: list[int]) -> dict[str, Any]:
    b_only = sum(b == 1 and p == 0 for b, p in zip(baseline, pointer, strict=True))
    p_only = sum(b == 0 and p == 1 for b, p in zip(baseline, pointer, strict=True))
    discordant = b_only + p_only
    if discordant == 0:
        pvalue = 1.0
    else:
        k = min(b_only, p_only)
        tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
        pvalue = min(1.0, 2.0 * tail)
    return {
        "baseline_only": b_only,
        "pointer_only": p_only,
        "discordant": discordant,
        "two_sided_exact_p": pvalue,
    }


def summarize(
    baseline_rows: dict[str, dict[str, Any]],
    pointer_rows: dict[str, dict[str, Any]],
    *,
    expected_rows: int,
    seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    if set(baseline_rows) != set(pointer_rows):
        missing_pointer = sorted(set(baseline_rows) - set(pointer_rows))
        missing_baseline = sorted(set(pointer_rows) - set(baseline_rows))
        raise ValueError(
            f"matched arm IDs differ: pointer_missing={missing_pointer[:5]}, "
            f"baseline_missing={missing_baseline[:5]}"
        )
    ids = sorted(baseline_rows)
    if len(ids) != expected_rows:
        raise ValueError(f"expected {expected_rows} matched reactions, observed {len(ids)}")
    baseline = [int(bool(baseline_rows[rid]["top1_exact"])) for rid in ids]
    pointer = [int(bool(pointer_rows[rid]["top1_exact"])) for rid in ids]
    b_pass = [int(bool(baseline_rows[rid]["pass_at_beam"])) for rid in ids]
    p_pass = [int(bool(pointer_rows[rid]["pass_at_beam"])) for rid in ids]
    invalid = sum(int(pointer_rows[rid].get("pointer_invalid_handles", 0)) for rid in ids)
    lo, hi = bootstrap_delta(baseline, pointer, seed=seed, draws=bootstrap_draws)
    pass_lo, pass_hi = bootstrap_delta(b_pass, p_pass, seed=seed + 1, draws=bootstrap_draws)
    return {
        "artifact_type": "vnext_pointer_product_start_paired_summary_v1",
        "n": len(ids),
        "baseline_top1": mean(baseline),
        "pointer_top1": mean(pointer),
        "delta_top1": mean(pointer) - mean(baseline),
        "delta_top1_ci95_low": lo,
        "delta_top1_ci95_high": hi,
        "baseline_pass_at_beam": mean(b_pass),
        "pointer_pass_at_beam": mean(p_pass),
        "delta_pass_at_beam": mean(p_pass) - mean(b_pass),
        "delta_pass_at_beam_ci95_low": pass_lo,
        "delta_pass_at_beam_ci95_high": pass_hi,
        "pointer_invalid_handles": invalid,
        "mcnemar_top1": exact_mcnemar(baseline, pointer),
        "scope": "matched_product_start_validation_not_test",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="glob for baseline shard JSONL files")
    parser.add_argument("--pointer", required=True, help="glob for pointer shard JSONL files")
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.expected_rows < 1 or args.bootstrap_draws < 1000:
        raise ValueError("invalid row/bootstrap count")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize(
        read_glob(args.baseline),
        read_glob(args.pointer),
        expected_rows=args.expected_rows,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
