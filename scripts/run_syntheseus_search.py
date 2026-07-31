#!/usr/bin/env python3
"""Run Syntheseus search over an offline MechET hypothesis pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.syntheseus_adapter import MechETBackwardReactionModel, MechETCandidatePool


def read_smiles(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["retro_star", "breadth_first"], default="retro_star")
    parser.add_argument("--num-results", type=int, default=20)
    parser.add_argument("--max-routes", type=int, default=25)
    parser.add_argument("--reaction-model-calls", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--time-limit-s", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pool = MechETCandidatePool.from_jsonl(args.candidate_pool)
    targets = read_smiles(args.targets)
    inventory = read_smiles(args.inventory)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "algorithm": args.algorithm,
                    "n_pool_targets": pool.n_targets,
                    "n_pool_candidates": pool.n_candidates,
                    "n_search_targets": len(targets),
                    "n_inventory": len(inventory),
                    "num_results": args.num_results,
                    "reaction_model_calls": args.reaction_model_calls,
                    "time_limit_s": args.time_limit_s,
                },
                indent=2,
            )
        )
        return 0

    try:
        from syntheseus import Molecule
        from syntheseus.search.analysis.route_extraction import iter_routes_time_order
        from syntheseus.search.mol_inventory import SmilesListInventory
    except ImportError as exc:
        raise RuntimeError("install mechet[planning]") from exc

    model = MechETBackwardReactionModel(
        pool,
        default_num_results=args.num_results,
        use_cache=True,
    )
    mol_inventory = SmilesListInventory(smiles_list=inventory)
    if args.algorithm == "retro_star":
        from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
        from syntheseus.search.node_evaluation.common import (
            ConstantNodeEvaluator,
            ReactionModelLogProbCost,
        )

        search = RetroStarSearch(
            reaction_model=model,
            mol_inventory=mol_inventory,
            limit_iterations=args.iterations,
            limit_reaction_model_calls=args.reaction_model_calls,
            time_limit_s=args.time_limit_s,
            value_function=ConstantNodeEvaluator(0.0),
            and_node_cost_fn=ReactionModelLogProbCost(),
        )
    else:
        from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch

        search = AndOr_BreadthFirstSearch(
            reaction_model=model,
            mol_inventory=mol_inventory,
            limit_iterations=args.iterations,
            limit_reaction_model_calls=args.reaction_model_calls,
            time_limit_s=args.time_limit_s,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, target in enumerate(targets):
        search.reset()
        graph, _ = search.run_from_mol(Molecule(target))
        routes = list(iter_routes_time_order(graph, max_routes=args.max_routes))
        graph_path = args.output_dir / f"target_{index:05d}.pkl"
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle)
        summaries.append(
            {
                "index": index,
                "target": target,
                "n_graph_nodes": len(graph),
                "n_routes": len(routes),
                "solved": bool(routes),
                "graph": graph_path.name,
            }
        )
    payload = {
        "algorithm": args.algorithm,
        "candidate_pool": str(args.candidate_pool),
        "n_targets": len(targets),
        "solved": sum(item["solved"] for item in summaries),
        "targets": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
