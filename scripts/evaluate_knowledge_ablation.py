#!/usr/bin/env python3
"""Evaluate trace, endpoint and knowledge-use metrics across matched conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import (
    condition_metrics,
    matched_intersection,
    read_jsonl,
)


def parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must be NAME=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = {name: read_jsonl(path) for name, path in args.condition}
    identifiers, matched = matched_intersection(loaded)
    metrics = {name: condition_metrics(rows) for name, rows in matched.items()}
    result = {
        "n_matched_ids": len(identifiers),
        "conditions": metrics,
        "required_interpretation": {
            "textbook_gain": "textbook RAG minus trace-only under matched IDs",
            "irrelevant_context_control": "textbook RAG minus length-matched irrelevant context",
            "structured_gain": "structured anchors minus trace-only",
            "combined_gain": "textbook plus anchors minus each individual condition",
            "faithfulness_gate": "trace_bound_rate must remain 1.0 in main-method conditions",
            "knowledge_reward_gate": "knowledge_direct_reward_violations must remain zero",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
