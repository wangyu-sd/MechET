#!/usr/bin/env python3
"""Attach external plausibility evidence to formally executable hypotheses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.plausibility import PlausibilityEvidence, combine_evidence, load_oracle
from mechet.proof_program import execute_proof


def candidates(row: dict) -> list[dict]:
    values = list(row.get("hypotheses") or row.get("candidates") or [])
    return values or ([row] if row.get("proof") or row.get("prediction") else [])


def evidence_from_item(item: dict) -> PlausibilityEvidence:
    payload = dict(item.get("evidence") or {})
    for key in (
        "precedent_score",
        "condition_score",
        "energy_score",
        "expert_score",
        "uncertainty",
        "sources",
    ):
        if key in item and key not in payload:
            payload[key] = item[key]
    if "sources" in payload:
        payload["sources"] = tuple(payload["sources"] or [])
    return PlausibilityEvidence(**payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle", default="", help="optional module:function")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    oracle = load_oracle(args.oracle) if args.oracle else None
    weights = json.loads(args.weights.read_text(encoding="utf-8")) if args.weights else None
    output_rows = []
    for row in rows:
        scored = []
        for item in candidates(row):
            proof = str(item.get("proof") or item.get("prediction") or "")
            formal_ok = execute_proof(proof).ok
            evidence = None
            combined = None
            if formal_ok:
                evidence = oracle({"row": row, "candidate": item}) if oracle else evidence_from_item(item)
                combined = combine_evidence(evidence, weights=weights)
            scored_item = dict(item)
            scored_item["formal_execute_ok"] = formal_ok
            scored_item["plausibility_evidence"] = evidence.to_dict() if evidence else None
            scored_item["plausibility_score"] = combined
            scored.append(scored_item)
        scored.sort(
            key=lambda item: (
                int(bool(item.get("formal_execute_ok"))),
                float(item.get("plausibility_score")) if item.get("plausibility_score") is not None else float("-inf"),
                float(item.get("model_logprob") or 0.0),
            ),
            reverse=True,
        )
        output_rows.append({"id": row.get("id"), "candidates": scored})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "n_targets": len(output_rows),
        "oracle": args.oracle or "precomputed_evidence_only",
        "warning": "plausibility scores rank executable proofs; they do not replace formal verification",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
