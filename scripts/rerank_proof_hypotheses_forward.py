#!/usr/bin/env python3
"""Rerank executable MechET proof hypotheses with the compact forward expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.forward_expert import ForwardElectronExpert, score_reaction
from mechet.proof_program import parse_proof_program


def rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    args = parser.parse_args()
    model = ForwardElectronExpert.load(args.checkpoint, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows(args.predictions):
            scored = []
            for hypothesis in row.get("hypotheses") or []:
                item = dict(hypothesis)
                if not item.get("execute_ok") or not item.get("proof"):
                    item["forward_evidence"] = None
                    item["forward_rank_score"] = float("-inf")
                    scored.append(item)
                    continue
                precursor = str(
                    item.get("derived_core_precursor")
                    or item.get("derived_precursor")
                    or ""
                )
                try:
                    target = parse_proof_program(item["proof"]).target_smiles
                    evidence = score_reaction(model, precursor, target)
                    item["forward_evidence"] = evidence.to_dict()
                    item["plausibility_score"] = evidence.target_score or 0.0
                    item["forward_rank_score"] = (
                        float(item.get("model_logprob") or 0.0)
                        + args.forward_weight * float(evidence.target_score or 0.0)
                        - args.uncertainty_weight
                        * float(evidence.uncertainty or 0.0)
                    )
                except Exception as exc:
                    item["forward_evidence"] = {
                        "verdict": "UNKNOWN",
                        "error": str(exc),
                    }
                    item["forward_rank_score"] = float("-inf")
                scored.append(item)
            scored.sort(
                key=lambda item: (
                    int(bool(item.get("execute_ok"))),
                    int(
                        (item.get("forward_evidence") or {}).get("target_rank")
                        == 1
                    ),
                    float(item.get("forward_rank_score") or float("-inf")),
                ),
                reverse=True,
            )
            row["hypotheses"] = scored
            row["forward_reranking"] = {
                "checkpoint": str(args.checkpoint),
                "forward_weight": args.forward_weight,
                "uncertainty_weight": args.uncertainty_weight,
                "selectivity": (
                    "requires explicit competitor products; not inferred from "
                    "alternative precursors"
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
