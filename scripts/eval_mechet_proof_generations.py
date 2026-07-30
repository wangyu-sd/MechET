#!/usr/bin/env python3
"""Evaluate executable proofs, partial-order equivalence, and local repair."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_diagnostics import repair_proof_once
from mechet.proof_equivalence import composition_signature, proofs_equivalent
from mechet.proof_program import ProofProgramError, verify_proof


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


def _gold_proof(row: dict) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _aggregate(cases: list[dict]) -> dict:
    n = len(cases)
    keys = (
        "format_ok",
        "execute_ok_before_repair",
        "repair_changed",
        "execute_ok",
        "endpoint_exact",
        "proof_equivalent_to_gold",
        "composition_match",
    )
    return {
        "n": n,
        **{
            f"{key}_rate": (
                sum(bool(case.get(key)) for case in cases) / n
                if n
                else 0.0
            )
            for key in keys
        },
        "mean_states": (
            sum(int(case.get("n_states") or 0) for case in cases) / n
            if n
            else 0.0
        ),
    }


def _equivalence_metrics(prediction: str, gold_proof: str) -> tuple[bool, bool]:
    if not prediction or not gold_proof:
        return False, False
    try:
        equivalent = proofs_equivalent(prediction, gold_proof)
        composition_match = (
            composition_signature(prediction)
            == composition_signature(gold_proof)
        )
        return equivalent, composition_match
    except ProofProgramError:
        return False, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attempt-local-repair", action="store_true")
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
        score_before = verify_proof(
            prediction,
            expected_precursor=_gold(row),
        )
        repair_changed = False
        repair_code = ""
        prediction_used = prediction
        if args.attempt_local_repair and not score_before.get("execute_ok"):
            repair = repair_proof_once(prediction)
            repair_changed = repair.changed
            prediction_used = repair.repaired_text
            if repair.certificate is not None:
                repair_code = repair.certificate.code

        score = verify_proof(
            prediction_used,
            expected_precursor=_gold(row),
        )
        equivalent, composition_match = _equivalence_metrics(
            prediction_used,
            _gold_proof(row),
        )
        case = {
            "id": row.get("id"),
            "topology": (row.get("metadata") or {}).get("topology"),
            "execute_ok_before_repair": bool(score_before.get("execute_ok")),
            "repair_changed": repair_changed,
            "repair_code": repair_code,
            "proof_equivalent_to_gold": equivalent,
            "composition_match": composition_match,
            **score,
        }
        cases.append(case)
        by_topology[str(case.get("topology") or "unknown")].append(case)

    report = {
        "mode": "proof_execution_partial_order",
        "data": str(args.data),
        "predictions": str(args.predictions),
        "attempt_local_repair": args.attempt_local_repair,
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
