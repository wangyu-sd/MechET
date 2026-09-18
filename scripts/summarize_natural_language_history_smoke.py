#!/usr/bin/env python3
"""Summarize a matched State-SFT versus compact-history closed-loop smoke."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_shards(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for shard in sorted(path.glob("results.shard-*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = dict(json.loads(line))
                identifier = str(row["id"])
                if identifier in rows:
                    raise ValueError(f"duplicate result ID: {identifier}")
                rows[identifier] = row
    return rows


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    rejected: Counter[str] = Counter()
    for row in values:
        rejected.update(
            {str(key): int(value) for key, value in dict(row.get("rejected") or {}).items()}
        )
    n = len(values)
    exact = sum(bool(row["top1_exact"]) for row in values)
    passed = sum(bool(row["pass_at_beam"]) for row in values)
    terminal = sum(bool(row["top_terminal"]) for row in values)
    return {
        "n_reactions": n,
        "top1_exact_count": exact,
        "top1_exact_rate": exact / max(n, 1),
        "pass_at_beam_count": passed,
        "pass_at_beam_rate": passed / max(n, 1),
        "formal_terminal_count": terminal,
        "formal_terminal_rate": terminal / max(n, 1),
        "mean_selected_actions": (
            mean(int(row["n_actions"]) for row in values) if values else 0.0
        ),
        "rejected_candidates": dict(rejected.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-sft", type=Path, required=True)
    parser.add_argument("--trajectory-sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = read_shards(args.state_sft)
    trajectory = read_shards(args.trajectory_sft)
    if set(state) != set(trajectory):
        raise ValueError("matched smoke conditions do not contain identical reaction IDs")
    state_report = summarize(state)
    trajectory_report = summarize(trajectory)
    report = {
        "artifact_type": "natural_language_history_closed_loop_smoke_v1",
        "protocol": {
            "product_only": True,
            "gold_history_visible": False,
            "same_reaction_ids": True,
            "same_search_budget": True,
            "history_runtime_reconstructed": True,
        },
        "state_sft": state_report,
        "trajectory_sft": trajectory_report,
        "trajectory_minus_state": {
            "top1_exact_rate": (
                trajectory_report["top1_exact_rate"] - state_report["top1_exact_rate"]
            ),
            "pass_at_beam_rate": (
                trajectory_report["pass_at_beam_rate"] - state_report["pass_at_beam_rate"]
            ),
            "formal_terminal_rate": (
                trajectory_report["formal_terminal_rate"]
                - state_report["formal_terminal_rate"]
            ),
        },
        "reaction_ids": sorted(state),
        "warning": "Development smoke only; not a full validation or test result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
