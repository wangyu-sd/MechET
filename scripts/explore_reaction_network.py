#!/usr/bin/env python3
"""Build an executable proof hypergraph from generated hypotheses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.reaction_network import ReactionNetwork, find_species_cycles, network_digest


def candidate_rows(row: dict) -> list[dict]:
    values = list(row.get("hypotheses") or row.get("candidates") or [])
    if not values and (row.get("proof") or row.get("prediction")):
        values = [row]
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cycle-length", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    network = ReactionNetwork()
    rejected = 0
    for row in rows:
        for index, item in enumerate(candidate_rows(row)):
            proof = str(item.get("proof") or item.get("prediction") or "")
            edge = network.add_proof(
                proof,
                edge_id=str(item.get("edge_id") or ""),
                reversible=bool(item.get("reversible")),
                model_score=float(item.get("model_logprob", item.get("model_score", 0.0)) or 0.0),
                plausibility_score=float(item.get("plausibility_score") or 0.0),
                energy=item.get("energy"),
                uncertainty=float(item.get("uncertainty") or 0.0),
                metadata={"source_id": row.get("id"), "source_index": index},
            )
            rejected += int(edge is None)
    payload = network.to_dict()
    payload["cycles"] = find_species_cycles(network, max_length=args.max_cycle_length)
    payload["network_digest"] = network_digest(network)
    payload["summary"]["rejected_non_executable"] = rejected
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
