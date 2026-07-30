#!/usr/bin/env python3
"""Evaluate proof-only generations by executing them to derive precursors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_program import verify_proof


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gold(row: dict) -> str:
    metadata = row.get("metadata") or {}
    return str(
        metadata.get("derived_precursor")
        or metadata.get("initial_reactants")
        or ""
    )


def _aggregate(cases: list[dict]) -> dict:
    n = len(cases)
    keys = ("format_ok", "execute_ok", "endpoint_exact")
    return {
        "n": n,
        **{
            f"{key}_rate": (
                sum(bool(case.get(key)) for case in cases) / n if n else 0.0
            )
            for key in keys
        },
        "mean_states": (
            sum(int(case.get("n_states") or 0) for case in cases) / n
            if n
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = _load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    predictions = {
        str(row.get("id")): row
        for row in _load_jsonl(args.predictions)
    }

    cases: list[dict] = []
    by_topology: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    for row in rows:
        prediction_row = predictions.get(str(row.get("id")))
        if prediction_row is None:
            missing += 1
            continue
        prediction = str(prediction_row.get("prediction") or "")
        score = verify_proof(
            prediction,
            expected_precursor=_gold(row),
        )
        case = {
            "id": row.get("id"),
            "topology": (row.get("metadata") or {}).get("topology"),
            **score,
        }
        cases.append(case)
        by_topology[str(case.get("topology") or "unknown")].append(case)

    report = {
        "mode": "proof_execution",
        "data": str(args.data),
        "predictions": str(args.predictions),
        "missing_predictions": missing,
        "overall": _aggregate(cases),
        "by_topology": {
            name: _aggregate(group)
            for name, group in sorted(by_topology.items())
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
