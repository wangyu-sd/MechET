#!/usr/bin/env python3
"""Rank executable proof candidates for external energy or experiment oracles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.reaction_network import ReactionNetwork, rank_frontier


def candidate_rows(row: dict) -> list[dict]:
    values = list(row.get("hypotheses") or row.get("candidates") or [])
    return values or ([row] if row.get("proof") or row.get("prediction") else [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--novelty-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--plausibility-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    network = ReactionNetwork.from_dict(json.loads(args.network.read_text(encoding="utf-8")))
    rows = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    temporary = ReactionNetwork()
    executable = []
    for row in rows:
        for index, item in enumerate(candidate_rows(row)):
            edge = temporary.add_proof(
                str(item.get("proof") or item.get("prediction") or ""),
                model_score=float(item.get("model_logprob", item.get("model_score", 0.0)) or 0.0),
                plausibility_score=float(item.get("plausibility_score") or 0.0),
                energy=item.get("energy"),
                uncertainty=float(item.get("uncertainty") or 0.0),
                metadata={"source_id": row.get("id"), "source_index": index},
            )
            if edge is not None:
                executable.append(edge)
    ranked = rank_frontier(
        executable,
        network,
        novelty_weight=args.novelty_weight,
        uncertainty_weight=args.uncertainty_weight,
        plausibility_weight=args.plausibility_weight,
        energy_weight=args.energy_weight,
    )[: args.top_k]
    payload = {
        "n_candidates": sum(len(candidate_rows(row)) for row in rows),
        "n_executable": len(executable),
        "top_k": args.top_k,
        "weights": {
            "novelty": args.novelty_weight,
            "uncertainty": args.uncertainty_weight,
            "plausibility": args.plausibility_weight,
            "energy": args.energy_weight,
        },
        "frontier": ranked,
        "warning": "frontier score prioritizes external evaluation; it is not a feasibility proof",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("n_candidates", "n_executable", "top_k")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
