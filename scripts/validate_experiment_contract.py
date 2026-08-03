#!/usr/bin/env python3
"""Validate matched scientific conditions before training or evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import (
    condition_contract_summary,
    read_jsonl,
    validate_alignment,
)


def parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must be NAME=PATH")
    return name, Path(path)


def relative_gap(left: int, right: int) -> float:
    return abs(left - right) / max(left, right, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, type=parse_condition)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--assistant-budget-tolerance",
        type=float,
        default=0.02,
        help="maximum relative supervised-character gap across matched conditions",
    )
    parser.add_argument(
        "--context-budget-tolerance",
        type=float,
        default=0.0,
        help="maximum relative evidence-context gap for conditions that contain textbook text",
    )
    parser.add_argument("--strict-assistant-budget", action="store_true")
    parser.add_argument("--strict-context-budget", action="store_true")
    args = parser.parse_args()

    conditions: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for name, path in args.condition:
        if name in conditions:
            raise ValueError(f"duplicate condition name: {name}")
        if not path.exists():
            raise FileNotFoundError(path)
        conditions[name] = read_jsonl(path)
        sources[name] = str(path)

    validate_alignment(conditions)
    summaries = {
        name: condition_contract_summary(rows)
        for name, rows in conditions.items()
    }

    assistant_values = {
        name: int(summary["assistant_characters"])
        for name, summary in summaries.items()
    }
    assistant_gap = 0.0
    for left_name, left in assistant_values.items():
        for right_name, right in assistant_values.items():
            if left_name < right_name:
                assistant_gap = max(assistant_gap, relative_gap(left, right))

    evidence_values = {
        name: int(summary["textbook_context_characters"])
        for name, summary in summaries.items()
        if int(summary["textbook_context_characters"]) > 0
    }
    context_gap = 0.0
    for left_name, left in evidence_values.items():
        for right_name, right in evidence_values.items():
            if left_name < right_name:
                context_gap = max(context_gap, relative_gap(left, right))

    violations = []
    if args.strict_assistant_budget and assistant_gap > args.assistant_budget_tolerance:
        violations.append(
            {
                "code": "ASSISTANT_BUDGET_MISMATCH",
                "observed_relative_gap": assistant_gap,
                "tolerance": args.assistant_budget_tolerance,
            }
        )
    if args.strict_context_budget and context_gap > args.context_budget_tolerance:
        violations.append(
            {
                "code": "CONTEXT_BUDGET_MISMATCH",
                "observed_relative_gap": context_gap,
                "tolerance": args.context_budget_tolerance,
            }
        )

    report = {
        "scientific_contract": "causal_compositional_electron_flow_v1",
        "sources": sources,
        "conditions": summaries,
        "same_stable_ids": len({v["stable_ids_sha256"] for v in summaries.values()}) == 1,
        "assistant_budget_relative_gap": assistant_gap,
        "context_budget_relative_gap": context_gap,
        "violations": violations,
        "passed": not violations,
        "notes": [
            "Character budgets are an early deterministic contract check.",
            "Final reports must additionally record tokenizer-specific input and supervised token counts.",
            "Direct-answer and trace-owned outputs need not have identical output syntax; any budget matching strategy must be declared before training.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
