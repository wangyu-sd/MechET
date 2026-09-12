#!/usr/bin/env python3
"""Audit GT-independent factorized electron-event coverage on frozen traces."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.transactional_event_space import audit_reference_event


def _summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    def percentile(q: float) -> int:
        return ordered[min(int((len(ordered) - 1) * q), len(ordered) - 1)]
    return {
        "n": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def run(args: argparse.Namespace) -> dict:
    reactions = events = covered = 0
    moves = covered_moves = 0
    reasons: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    arities: Counter[int] = Counter()
    source_counts: list[int] = []
    polar_source_counts: list[int] = []
    radical_source_counts: list[int] = []
    sink_counts: list[int] = []
    examples: list[dict] = []

    with args.data.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            reactions += 1
            steps = ((row.get("metadata") or {}).get("trace_plan") or {}).get("steps") or []
            for step in steps:
                if args.limit_events and events >= args.limit_events:
                    break
                gold = list(step.get("moves") or [])
                result = audit_reference_event(str(step.get("state_before") or ""), gold)
                events += 1
                arities[len(gold)] += 1
                covered += int(result["covered"])
                reasons[str(result["reason"])] += 1
                source_counts.append(int(result["source_candidates"]))
                polar_source_counts.append(int(result["polar_source_candidates"]))
                radical_source_counts.append(int(result["radical_source_candidates"]))
                sink_counts.extend(int(value) for value in result["gold_conditioned_sink_candidates"])
                for item in result["move_coverage"]:
                    moves += 1
                    covered_moves += int(item["covered"])
                    signatures["->".join(item["signature"])] += 1
                if not result["covered"] and len(examples) < args.max_failure_examples:
                    examples.append(
                        {
                            "id": row.get("id"),
                            "step_index": step.get("step_index"),
                            "reason": result["reason"],
                            "moves": gold,
                            "move_coverage": result["move_coverage"],
                        }
                    )
            if args.limit_events and events >= args.limit_events:
                break

    return {
        "artifact_type": "gt_independent_transactional_event_space_audit_v1",
        "data": str(args.data),
        "reactions_read": reactions,
        "events": events,
        "events_covered": covered,
        "event_coverage": covered / events if events else 0.0,
        "moves": moves,
        "moves_covered": covered_moves,
        "move_coverage": covered_moves / moves if moves else 0.0,
        "event_arities": dict(sorted(arities.items())),
        "move_signatures": dict(sorted(signatures.items())),
        "failure_reasons": dict(sorted(reasons.items())),
        "source_candidate_count": _summary(source_counts),
        "polar_source_candidate_count": _summary(polar_source_counts),
        "radical_source_candidate_count": _summary(radical_source_counts),
        "gold_conditioned_sink_candidate_count": _summary(sink_counts),
        "failure_examples": examples,
        "claim_boundary": (
            "Candidates are derived only from each executor state. Reference moves are "
            "used solely after construction to measure coverage. This audits factorized "
            "polar-arrow support, not full-event ranking or endpoint accuracy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--max-failure-examples", type=int, default=8)
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
